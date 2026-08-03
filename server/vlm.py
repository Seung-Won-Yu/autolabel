"""VLM 문맥 심판: 판정 기준(rubric) 문서 + 박스 crop → pass/fail/unsure 예비 판정.

검출 모델은 "어디에 있나"만 답한다. "이 박스가 기준에 맞나"(예: 일반 차량이
아니라 사고 연루 차량인가)는 문맥 판정이라 픽셀 유사도로는 못 푼다 — 기준
텍스트를 읽고 이미지를 보는 VLM만 자동 제안이 가능한 부류다. 사람은 fail과
unsure만 확인하면 되므로 리뷰가 전수 판독에서 예외 확인으로 바뀐다.

제공자는 플러그인 (자동 감지 순서):
- anthropic: ANTHROPIC_API_KEY가 있고 `pip install anthropic` 된 경우 (기본
  모델 claude-opus-5 — AUTOLABEL_VLM_MODEL로 변경, 예: claude-haiku-4-5로
  비용 절감). 가장 빠름, 종량 과금.
- claude-code: Claude Code CLI가 설치돼 있으면 헤드리스(`claude -p`)로 호출 —
  **구독(Pro/Team/Max)에 포함되어 추가 비용 없음**. 박스당 수 초로 느리지만
  개인 규모엔 충분. API 키 없이 쓰는 기본 경로.
- ollama: localhost:11434에 비전 모델이 떠 있는 경우 (기본 qwen2.5vl —
  AUTOLABEL_OLLAMA_MODEL로 변경). 오프라인·무료.
- 없으면 기능만 비활성 — 도구는 정상 기동. AUTOLABEL_VLM으로 강제 가능.

판정 결과는 어노테이션 meta.vlm에 저장하고 rubric 해시를 함께 남긴다 —
같은 기준으로 재실행하면 캐시를 재사용해 비용이 다시 들지 않는다.
"""
import base64
import hashlib
import io
import json
import os
import re
import threading

from PIL import Image, ImageDraw

from server import jobs
from server.db import get_db, row_to_dict

OLLAMA_URL = os.environ.get("AUTOLABEL_OLLAMA_URL", "http://localhost:11434")
CROP_MARGIN = 0.25   # 박스 주변 문맥이 판정 근거다 (사고 차량 = 주변 상황)
CROP_MAX = 1024      # 비전 토큰 비용 상한 — 판정에 이 이상 해상도는 불필요

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail", "unsure"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

_jobs: dict[int, dict] = {}


def rubric_sha(rubric: str) -> str:
    return hashlib.sha256(rubric.strip().encode()).hexdigest()[:12]


def _anthropic_ready() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def _ollama_ready() -> bool:
    try:
        import httpx

        return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.0).status_code == 200
    except Exception:
        return False


def _claude_code_ready() -> bool:
    import shutil

    return shutil.which("claude") is not None


def provider() -> str | None:
    """사용할 VLM 제공자. AUTOLABEL_VLM으로 강제(anthropic|claude-code|ollama|off)."""
    forced = os.environ.get("AUTOLABEL_VLM")
    if forced == "off":
        return None
    if forced in ("anthropic", "claude-code", "ollama"):
        return forced
    if _anthropic_ready():
        return "anthropic"
    # Claude Code 구독은 추가 비용이 없고 품질이 로컬 모델보다 높다 —
    # API 키가 없을 때의 기본 경로
    if _claude_code_ready():
        return "claude-code"
    if _ollama_ready():
        return "ollama"
    return None


PROVIDER_HINT = ("VLM 제공자가 없습니다 — Claude Code CLI 설치(구독으로 무료), "
                 "ANTHROPIC_API_KEY 설정 후 `pip install anthropic`, 또는 "
                 "Ollama 비전 모델(예: ollama pull qwen2.5vl) 중 하나가 필요합니다")


def _crop_png(image: Image.Image, bbox: list[float]) -> bytes:
    """박스 + 여백 crop, 박스 위치를 빨간 사각형으로 표시, PNG 바이트."""
    x, y, w, h = bbox
    mx, my = w * CROP_MARGIN, h * CROP_MARGIN
    x1 = max(0, int(x - mx)); y1 = max(0, int(y - my))
    x2 = min(image.width, int(x + w + mx)); y2 = min(image.height, int(y + h + my))
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    lw = max(2, int(max(crop.size) / 300))
    draw.rectangle([x - x1, y - y1, x + w - x1, y + h - y1], outline=(255, 40, 40), width=lw)
    if max(crop.size) > CROP_MAX:
        crop.thumbnail((CROP_MAX, CROP_MAX))
    buf = io.BytesIO()
    crop.save(buf, "PNG")
    return buf.getvalue()


def _prompt(rubric: str, class_name: str) -> str:
    return (
        "당신은 데이터 라벨링 검수자다. 아래 판정 기준에 따라, 이미지에서 빨간 "
        f"사각형으로 표시된 객체가 '{class_name}' 라벨로 올바른지 판정하라.\n\n"
        f"판정 기준:\n{rubric}\n\n"
        "verdict는 pass(기준에 부합), fail(부합하지 않음), unsure(이미지만으로 "
        "판단 불가) 중 하나. reason은 한 문장의 한국어 근거. "
        'JSON으로만 답하라: {"verdict": ..., "reason": ...}'
    )


def _judge_anthropic(img_b64: str, prompt: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.environ.get("AUTOLABEL_VLM_MODEL", "claude-opus-5"),
        max_tokens=1024,
        # 박스 하나의 기준 부합 여부는 분류 문제 — 낮은 effort로 충분하고
        # 수백 장 배치에서 비용을 지배한다. 필요하면 환경변수로 올릴 것.
        output_config={
            "effort": os.environ.get("AUTOLABEL_VLM_EFFORT", "low"),
            "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
        },
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": img_b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    if resp.stop_reason == "refusal":
        return {"verdict": "unsure", "reason": "안전 분류기가 요청을 거부했습니다"}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text)


def _judge_claude_code(img_png: bytes, prompt: str) -> dict:
    """Claude Code CLI 헤드리스 — 구독(Pro/Team/Max)으로 호출, API 키 불필요.

    crop을 임시 파일로 두고 Read 도구만 허용해 읽게 한다. 세션을 매번 띄우므로
    박스당 수 초 — 개인 규모용. 대량이면 anthropic 제공자가 빠르다.
    """
    import subprocess
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".png", prefix="vlm_judge_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(img_png)
        r = subprocess.run(
            ["claude", "-p",
             f"{path} 이미지를 Read 도구로 읽어라. 그 다음 아래 지시에 따라 "
             f"JSON 한 줄로만 답하라 (다른 말 금지).\n\n{prompt}",
             "--output-format", "json", "--allowedTools", "Read",
             "--max-turns", "3"],
            capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise RuntimeError(f"claude CLI 실패 (exit {r.returncode}): {r.stderr[:200]}")
        text = json.loads(r.stdout).get("result", "")
        m = re.search(r"\{.*\}", text, re.S)  # 앞뒤 산문 방어
        if not m:
            raise ValueError(f"응답에 JSON 없음: {text[:200]}")
        return json.loads(m.group(0))
    finally:
        os.unlink(path)


def _judge_ollama(img_b64: str, prompt: str) -> dict:
    import httpx

    r = httpx.post(f"{OLLAMA_URL}/api/chat", timeout=120.0, json={
        "model": os.environ.get("AUTOLABEL_OLLAMA_MODEL", "qwen2.5vl"),
        "stream": False,
        "format": VERDICT_SCHEMA,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
    })
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def judge_box(image: Image.Image, bbox: list[float], class_name: str,
              rubric: str, prov: str) -> dict:
    """박스 하나 판정. 반환: {verdict, reason}. 실패는 unsure로 강등 — 배치가
    한 건 때문에 죽으면 안 되고, 판정 불능은 사람이 보라는 신호가 맞다."""
    png = _crop_png(image, bbox)
    prompt = _prompt(rubric, class_name)
    try:
        if prov == "anthropic":
            out = _judge_anthropic(base64.standard_b64encode(png).decode(), prompt)
        elif prov == "claude-code":
            out = _judge_claude_code(png, prompt)
        else:
            out = _judge_ollama(base64.standard_b64encode(png).decode(), prompt)
        if out.get("verdict") not in ("pass", "fail", "unsure"):
            return {"verdict": "unsure", "reason": f"판정 형식 오류: {out}", "error": True}
        return {"verdict": out["verdict"], "reason": str(out.get("reason", ""))[:500]}
    except Exception as e:
        # error 플래그를 남긴다 — 이게 없으면 일시적 장애(429·키 만료·네트워크)가
        # unsure로 영구 캐시되어 같은 기준으론 영원히 재판정되지 않는다
        return {"verdict": "unsure", "reason": f"판정 실패: {e}", "error": True}


def job_status(pid: int) -> dict:
    return _jobs.get(pid) or jobs.get("vlm", pid)


def _image_path(im: dict):
    from server.main import _row_image_path
    return _row_image_path(im)


def _run_judge(pid: int, rubric: str, image_ids: list[int], prov: str):
    job = _jobs[pid]
    sha = rubric_sha(rubric)
    counts = {"pass": 0, "fail": 0, "unsure": 0, "cached": 0, "stale": 0}
    conn = get_db()
    # 박스 단위 진행률 — 판정은 박스당 수 초~수십 초라 이미지 단위(done)만
    # 보여주면 박스 많은 이미지에서 수십 분째 그대로로 보인다 (실측: 멈춤과
    # 저속을 구분 못 해 죽었는지 확인하러 옴)
    qmarks = ",".join("?" * len(image_ids))
    total_boxes = conn.execute(
        f"SELECT COUNT(*) FROM annotations WHERE image_id IN ({qmarks})",
        image_ids).fetchone()[0] if image_ids else 0
    done_boxes = 0
    job.update(total_boxes=total_boxes, done_boxes=0)
    try:
        for n, iid in enumerate(image_ids, 1):
            im = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
            if not im:
                continue
            anns = [row_to_dict(a) for a in conn.execute(
                "SELECT * FROM annotations WHERE image_id=?", (iid,))]
            img = None
            for a in anns:
                prev = (a.get("meta") or {}).get("vlm")
                box_key = [a["bbox"], a["class_name"]]
                # 같은 기준·같은 박스로 이미 판정된 것만 캐시로 인정.
                # error 판정(일시 장애)은 제외 — 아니면 429 한 번에 영구 unsure.
                # box 스냅샷 불일치(판정 후 박스 수정)도 재판정 대상이다.
                done_boxes += 1
                if (prev and prev.get("rubric_sha") == sha
                        and not prev.get("error") and prev.get("box") == box_key):
                    counts["cached"] += 1
                    counts[prev["verdict"]] = counts.get(prev["verdict"], 0) + 1
                    job.update(done_boxes=done_boxes)
                    continue
                if img is None:
                    path = _image_path(im)
                    if not path.exists():
                        break
                    img = Image.open(path).convert("RGB")
                v = judge_box(img, a["bbox"], a["class_name"], rubric, prov)
                # 판정하는 수 초 사이 사용자가 저장하면 행이 DELETE+INSERT로
                # 재생성되고 id가 재사용될 수 있다 — 낡은 meta 사본을 통째로
                # 덮지 말고, 정체성(id+이미지+클래스+박스)이 그대로일 때만
                # 현재 meta에 vlm 키 하나를 병합한다. 불일치면 0행 매치로 무해.
                patch = {**v, "rubric_sha": sha, "provider": prov, "box": box_key}
                # json_patch는 RFC-7386 병합 — 패치에 없는 키는 남는다. 이전
                # 실패의 error 플래그가 성공 판정 뒤에도 살아남아 캐시를 영영
                # 막지 않게, 성공 시 null(=키 삭제)을 명시한다.
                patch.setdefault("error", None)
                cur = conn.execute(
                    "UPDATE annotations SET meta=json_patch(meta, ?) "
                    "WHERE id=? AND image_id=? AND class_name=? AND bbox=?",
                    (json.dumps({"vlm": patch}, ensure_ascii=False),
                     a["id"], iid, a["class_name"], json.dumps(a["bbox"])))
                if cur.rowcount == 0:
                    counts["stale"] += 1  # 판정 중 편집·삭제됨 — 결과 폐기
                else:
                    counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
                # 박스마다 즉시 커밋 — 다음 판정(수 초~수십 초)까지 쓰기
                # 트랜잭션을 쥐고 있으면 그동안 다른 쓰기(배치 오토라벨,
                # 프로젝트 생성)가 전부 "database is locked"로 죽는다 (실측)
                conn.commit()
                job.update(done_boxes=done_boxes, **counts)
            conn.commit()
            job.update(done=n, **counts)
            jobs.update("vlm", pid, done=n, **counts)
        judged = counts["pass"] + counts["fail"] + counts["unsure"]
        advice = (f"판정 {judged}건: 부합 {counts['pass']} · 위반 {counts['fail']} · "
                  f"불확실 {counts['unsure']}"
                  + (f" (캐시 재사용 {counts['cached']})" if counts["cached"] else "")
                  + (f" (편집으로 무효화 {counts['stale']})" if counts["stale"] else "")
                  + " — 위반·불확실만 확인하면 됩니다")
        job.update(status="completed", advice=advice, **counts)
        jobs.update("vlm", pid, status="completed", advice=advice, **counts)
    except Exception as e:
        job.update(status="failed", error=str(e))
        jobs.update("vlm", pid, status="failed", error=str(e))
    finally:
        conn.close()


_start_lock = threading.Lock()


def start_judge(pid: int, rubric: str, image_ids: list[int]) -> dict:
    # provider()는 Ollama HTTP 프로브로 최대 1초 걸린다 — 잠금 밖에서 확인
    prov = provider()
    if not prov:
        return {"status": "failed", "error": PROVIDER_HINT}
    # 검사-등록을 원자화한다. 더블클릭·중복 탭이 검사를 동시에 통과하면 같은
    # 배치가 두 번 판정되어 API 비용이 2배가 되고 잡 상태가 서로를 덮는다.
    with _start_lock:
        if _jobs.get(pid, {}).get("status") == "running":
            return _jobs[pid]
        _jobs[pid] = {"status": "running", "done": 0, "total": len(image_ids),
                      "provider": prov}
        jobs.start("vlm", pid, done=0, total=len(image_ids))
        threading.Thread(target=_run_judge, args=(pid, rubric, image_ids, prov),
                         daemon=True).start()
    return _jobs[pid]

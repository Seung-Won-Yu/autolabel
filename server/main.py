"""오토라벨 도구 백엔드 — FastAPI.

실행: .venv/bin/python -m uvicorn server.main:app --port 8899 --reload
"""
import io
import hashlib
import json
import math
import os
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from server import ensemble, foundation, importer, jobs, ml, train, video, vlm
from server.db import get_db, init_db, row_to_dict

DATA_ROOT = Path(os.environ.get("AUTOLABEL_DATA")
                 or Path(__file__).parent.parent / "data")
DATA_DIR = DATA_ROOT / "uploads"
MODELS_DIR = DATA_ROOT / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="autolabel")
app.add_middleware(
    CORSMiddleware,
    # 파일 경로 임포트처럼 로컬 권한을 쓰는 API가 있다. 서버가 loopback에만
    # 떠도 CORS를 전부 열면 사용자가 방문한 외부 페이지가 localhost API를
    # 호출할 수 있으므로, 개발·E2E 포트를 포함한 로컬 origin만 허용한다.
    allow_origins=[],
    allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)
init_db()
# 이전 프로세스와 함께 죽은 인프로세스 잡을 interrupted로 정리한다.
# 안 하면 프론트가 사라진 기록을 "완료"로 읽어, 절반만 처리된 데이터를 두고
# 사용자에게 끝났다고 알린다.
jobs.sweep_stale()

IMAGE_STATUSES = {"unlabeled", "prelabeled", "approved", "rejected"}
AUTOLABEL_PROFILES = {"balanced", "recall"}
MODEL_BUNDLE_SCHEMA = 1
MODEL_QUALITY_FLOOR = 0.30
MODEL_CLASS_QUALITY_FLOOR = 0.10
MIN_CLASS_TEST_INSTANCES = 2
MAX_MODEL_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_BUNDLE_MEMBERS = 20


def _valid_status(value) -> str:
    if not isinstance(value, str) or value not in IMAGE_STATUSES:
        raise HTTPException(400, f"status는 {', '.join(sorted(IMAGE_STATUSES))} 중 하나여야 합니다")
    return value


def _autolabel_options(body: dict | None) -> tuple[str, float]:
    """오토라벨 프로필 검증.

    balanced는 온톨로지 임계값 그대로, recall은 전용 모델에 낮은 후보 임계값과
    증강 추론을 적용한다. 후보 임계값은 자동 승인 기준과 별개다.
    """
    body = body or {}
    profile = body.get("profile", "balanced")
    if profile not in AUTOLABEL_PROFILES:
        raise HTTPException(400, f"profile은 {', '.join(sorted(AUTOLABEL_PROFILES))} 중 하나여야 합니다")
    try:
        candidate_conf = float(body.get("candidate_conf", 0.10))
    except (TypeError, ValueError):
        raise HTTPException(400, "candidate_conf는 0~1 사이 숫자여야 합니다")
    if not math.isfinite(candidate_conf) or not 0 < candidate_conf < 1:
        raise HTTPException(400, "candidate_conf는 0~1 사이 숫자여야 합니다")
    return profile, candidate_conf


def _scoped_image_ids(conn, pid: int, raw) -> list[int]:
    """클라이언트가 넘긴 이미지 id가 전부 해당 프로젝트 소속인지 확인한다."""
    if not isinstance(raw, list) or not raw or not all(isinstance(i, int) for i in raw):
        raise HTTPException(400, "image_ids는 하나 이상의 정수 배열이어야 합니다")
    ids = list(dict.fromkeys(raw))
    marks = ",".join("?" * len(ids))
    found = {r["id"] for r in conn.execute(
        f"SELECT id FROM images WHERE project_id=? AND id IN ({marks})", (pid, *ids))}
    if found != set(ids):
        raise HTTPException(400, "다른 프로젝트 또는 존재하지 않는 image_id가 포함됐습니다")
    return ids


def _acceptance_params(body: dict) -> tuple[float, float, int | None]:
    try:
        target = float(body.get("target_error_rate", 0.05))
        confidence = float(body.get("confidence", 0.95))
        max_defects = (None if body.get("max_defects") is None
                       else int(body["max_defects"]))
    except (TypeError, ValueError):
        raise HTTPException(400, "검수 통계 파라미터 형식이 잘못됐습니다")
    if (not math.isfinite(target) or not 0 < target < 1
            or not math.isfinite(confidence) or not 0 < confidence < 1
            or (max_defects is not None and max_defects < 0)):
        raise HTTPException(400, "오류율·신뢰도는 0~1, 허용 불량은 0 이상이어야 합니다")
    return target, confidence, max_defects


# ---------- 프로젝트 ----------

@app.post("/api/projects")
def create_project(body: dict):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO projects (name, ontology) VALUES (?, ?)",
        (body["name"], json.dumps(body.get("ontology", []))))
    conn.commit()
    pid = cur.lastrowid
    row = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    conn.close()
    return row_to_dict(row)


@app.get("/api/capabilities")
def capabilities():
    """설치된 모델 역량 — UI가 SAM 3 유무 등을 안내하는 데 쓴다."""
    from pathlib import Path

    return {
        "sam3": ml.sam3_available(),
        "sam3_hint": "models/sam3.pt 를 두면 텍스트 검출이 SAM 3로 승급됩니다 "
                     "(huggingface.co/facebook/sam3 접근 승인 후 다운로드)",
        "sam_encoder": not ml.NO_MODELS and Path(ml.SAM_CKPT).exists(),
        "device": ml.DEVICE,
        "models_disabled": ml.NO_MODELS,
        # 문맥 심판(rubric 기반 박스 판정)에 쓸 VLM 제공자 — 없으면 null
        "vlm": vlm.provider(),
        "vlm_hint": vlm.PROVIDER_HINT,
    }


@app.get("/api/projects")
def list_projects():
    conn = get_db()
    rows = [row_to_dict(r) for r in conn.execute(
        "SELECT p.*, (SELECT COUNT(*) FROM images i WHERE i.project_id=p.id) AS image_count, "
        "(SELECT COUNT(*) FROM images i WHERE i.project_id=p.id AND i.status='approved') "
        "AS approved_count FROM projects p ORDER BY p.id DESC")]
    conn.close()
    return rows


@app.put("/api/projects/{pid}/rubric")
def update_rubric(pid: int, body: dict):
    """VLM 문맥 심판의 판정 기준 문서 저장."""
    conn = get_db()
    cur = conn.execute("UPDATE projects SET rubric=? WHERE id=?",
                       (body.get("rubric", ""), pid))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "프로젝트 없음")
    return {"ok": True}


@app.post("/api/projects/{pid}/vlm-judge")
def vlm_judge(pid: int, body: dict | None = None):
    """VLM 문맥 심판 시작 — 리뷰 대기(prelabeled) 이미지의 박스를 rubric으로 판정.

    같은 rubric으로 판정된 박스는 캐시를 재사용하므로 재실행 비용이 없다.
    image_ids로 대상을 좁힐 수 있다.
    """
    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if not proj:
        conn.close()
        raise HTTPException(404, "프로젝트 없음")
    rubric = (proj["rubric"] or "").strip() if "rubric" in proj.keys() else ""
    if not rubric:
        conn.close()
        raise HTTPException(400, "판정 기준(rubric)을 먼저 작성하세요")
    if body is not None and "image_ids" in body:
        try:
            ids = _scoped_image_ids(conn, pid, body["image_ids"])
        except Exception:
            conn.close()
            raise
    else:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM images WHERE project_id=? AND status='prelabeled'", (pid,))]
    conn.close()
    if not ids:
        return {"status": "completed", "done": 0, "total": 0,
                "advice": "판정할 이미지가 없습니다 — 리뷰 대기(prelabeled) 이미지가 없습니다"}
    st = vlm.start_judge(pid, rubric, ids)
    if st.get("status") == "failed":
        raise HTTPException(503, st["error"])
    return st


@app.get("/api/projects/{pid}/vlm-judge/status")
def vlm_judge_status(pid: int):
    return vlm.job_status(pid)


@app.put("/api/projects/{pid}/ontology")
def update_ontology(pid: int, body: dict):
    conn = get_db()
    cur = conn.execute("UPDATE projects SET ontology=? WHERE id=?",
                       (json.dumps(body["ontology"]), pid))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "프로젝트 없음")
    return {"ok": True}


@app.get("/api/projects/{pid}")
def get_project(pid: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404)
    return row_to_dict(row)


# ---------- 이미지 ----------

@app.post("/api/projects/{pid}/images")
async def upload_images(pid: int, files: list[UploadFile]):
    conn = get_db()
    if not conn.execute("SELECT id FROM projects WHERE id=?", (pid,)).fetchone():
        conn.close()
        raise HTTPException(404, "프로젝트 없음")
    saved, failed = [], []
    pdir = DATA_DIR / str(pid)
    pdir.mkdir(exist_ok=True)
    try:
        for f in files:
            data = await f.read()
            # 경로 구분자가 든 파일명은 저장 경로를 의도 밖으로 끌고 간다
            # (없는 디렉터리로 배치 전체 실패 + 고아 파일) — basename만 쓴다
            fname = Path(f.filename or "").name or "upload"
            iid = None
            try:
                im = Image.open(io.BytesIO(data))
                w, h = im.size
                cur = conn.execute(
                    "INSERT INTO images (project_id, file_name, width, height) VALUES (?,?,?,?)",
                    (pid, fname, w, h))
                iid = cur.lastrowid
                # 파일명 충돌 회피 — id 프리픽스 저장
                (pdir / f"{iid}_{fname}").write_bytes(data)
                saved.append(iid)
            except Exception:
                # 조용히 스킵하면 프론트가 전량 성공으로 보고한다 — 목록으로 알린다
                if iid is not None:
                    conn.execute("DELETE FROM images WHERE id=?", (iid,))
                failed.append(fname)
        conn.commit()
    finally:
        conn.close()
    return {"saved": saved, "failed": failed}


@app.post("/api/projects/{pid}/video")
async def upload_video(pid: int, file: UploadFile, stride: int = 5,
                       max_frames: int = video.MAX_FRAMES_DEFAULT):
    """비디오 업로드 → 프레임 추출 + SAM 3 전파 트래킹 (백그라운드).

    프레임은 일반 이미지로 등록되어 리뷰·학습·익스포트 레인을 그대로 탄다.
    """
    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    conn.close()
    if not proj:
        raise HTTPException(404, "프로젝트 없음")
    ontology = json.loads(proj["ontology"])
    if not ontology:
        raise HTTPException(400, "클래스를 먼저 정의하세요 — 트래킹 프롬프트로 씁니다")
    fname = Path(file.filename or "").name or "video.mp4"
    pdir = DATA_DIR / str(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    dst = pdir / f"src_{fname}"
    # 통째로 read()하면 영상 크기만큼 RAM 스파이크 + 이벤트 루프 블로킹 —
    # 1MB 청크 스트리밍 복사
    with dst.open("wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    return video.start(pid, dst, ontology, DATA_DIR, stride=stride, max_frames=max_frames)


@app.get("/api/projects/{pid}/video/status")
def video_status(pid: int):
    return video.job_status(pid)


@app.get("/api/projects/{pid}/images")
def list_images(pid: int, status: str | None = None):
    conn = get_db()
    # 상관 서브쿼리 3개는 이미지마다 annotations를 3번 훑는다 — 집계 조인
    # 1패스로. (목록은 프로젝트 열기·배치 후마다 도는 핫패스)
    q = ("SELECT i.*, COALESCE(s.ann_count, 0) AS ann_count, s.min_conf, "
         "COALESCE(s.vlm_flags, 0) AS vlm_flags "
         "FROM images i LEFT JOIN ("
         "  SELECT image_id, COUNT(*) AS ann_count, MIN(confidence) AS min_conf, "
         "  SUM(json_extract(meta, '$.vlm.verdict') IN ('fail','unsure')) AS vlm_flags "
         "  FROM annotations GROUP BY image_id"
         ") s ON s.image_id = i.id WHERE i.project_id=?")
    args: list = [pid]
    if status:
        q += " AND i.status=?"
        args.append(status)
    rows = [row_to_dict(r) for r in conn.execute(q + " ORDER BY i.id", args)]
    conn.close()
    return rows


def _attachment(filename: str) -> str:
    """다운로드 파일명을 Content-Disposition 헤더로 안전하게 만든다.

    HTTP 헤더는 latin-1만 담을 수 있어 한글 프로젝트명을 그대로 넣으면
    UnicodeEncodeError로 500이 난다 — 한글 이름 프로젝트는 익스포트가 통째로
    막혔다. RFC 6266대로 ASCII 대체본과 UTF-8 원본을 함께 보낸다.
    """
    from urllib.parse import quote

    ascii_name = filename.encode("ascii", "replace").decode("ascii").replace("?", "_")
    return f"attachment; filename={ascii_name}; filename*=UTF-8''{quote(filename)}"


def _row_image_path(row) -> Path:
    """이미지 행 하나의 실제 디스크 경로.

    연결 임포트된 이미지는 복사본이 없고 src_path에 원본 경로만 있다.
    업로드 경로만 보면 조용히 건너뛰게 되므로(익스포트·학습 사고) 항상 이걸 쓴다.
    """
    src = row["src_path"] if "src_path" in row.keys() else None
    if src:
        return Path(src)
    return DATA_DIR / str(row["project_id"]) / f"{row['id']}_{row['file_name']}"


def _image_path(iid: int) -> Path:
    conn = get_db()
    row = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404)
    return _row_image_path(row)


@app.get("/api/images/{iid}/file")
def image_file(iid: int):
    return FileResponse(_image_path(iid))


THUMB_DIR = DATA_DIR.parent / "thumbs"
THUMB_MAX = 128  # 목록 썸네일은 44px로 표시 — 레티나 여유까지 이 정도면 충분


@app.get("/api/images/{iid}/thumb")
def image_thumb(iid: int, size: int = 96):
    """목록용 축소 이미지. 디스크에 캐시한다.

    예전엔 목록이 원본을 그대로 받아 44x44로 줄여 그렸다. signature 143장은
    스크롤 한 번에 9.3MB, 2MB 사진 1만 장이면 20GB를 썸네일 표시에만 쓴다.
    """
    size = max(32, min(size, THUMB_MAX))
    src = _image_path(iid)
    if not src.exists():
        raise HTTPException(404, "원본 이미지 없음")
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMB_DIR / f"{iid}_{size}.jpg"
    # 원본이 바뀌면(재임포트 등) 다시 만든다
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        im = Image.open(src)
        im.draft("RGB", (size * 2, size * 2))  # JPEG는 디코딩부터 축소해 훨씬 빠르다
        im = im.convert("RGB")
        im.thumbnail((size, size))
        im.save(out, "JPEG", quality=72)
    return FileResponse(out, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.put("/api/images/{iid}/status")
def set_image_status(iid: int, body: dict):
    status = _valid_status(body.get("status"))
    conn = get_db()
    cur = conn.execute("UPDATE images SET status=? WHERE id=?", (status, iid))
    conn.commit()
    row = conn.execute("SELECT project_id FROM images WHERE id=?", (iid,)).fetchone()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "이미지 없음")
    # 승인 누적 시 자동 파인튜닝 트리거 (조건 미달이면 no-op)
    trained = None
    if status == "approved" and row:
        trained = train.maybe_start_training(row["project_id"], debounce=True)
    return {"ok": True, "train": trained}


# ---------- 어노테이션 ----------

@app.get("/api/images/{iid}/annotations")
def get_annotations(iid: int):
    conn = get_db()
    rows = [row_to_dict(r) for r in conn.execute(
        "SELECT * FROM annotations WHERE image_id=? ORDER BY id", (iid,))]
    conn.close()
    return rows


@app.put("/api/images/{iid}/annotations")
def replace_annotations(iid: int, body: dict):
    """이미지의 어노테이션 스냅샷 저장. 기존 행 id는 가능한 한 유지한다.

    DELETE+INSERT 전체 교체는 저장할 때마다 id를 바꾼다. 프론트가 이전 id를
    들고 한 번 더 저장하면, 백그라운드 VLM이 새 행에 기록한 판정이 다음 저장에서
    유실된다. 기존 행은 UPDATE하고 새 행만 INSERT한 뒤 빠진 행만 삭제한다.
    """
    conn = get_db()
    image = conn.execute("SELECT id FROM images WHERE id=?", (iid,)).fetchone()
    if not image:
        conn.close()
        raise HTTPException(404, "이미지 없음")
    incoming = body.get("annotations")
    if not isinstance(incoming, list):
        conn.close()
        raise HTTPException(400, "annotations는 배열이어야 합니다")

    existing = {r["id"]: row_to_dict(r) for r in conn.execute(
        "SELECT * FROM annotations WHERE image_id=?", (iid,))}
    # 부모가 스냅샷에서 빠졌는데 자식만 남으면 dangling FK가 된다. 현재 UI는
    # 새 계층 링크를 만들지 않으므로, 같은 이미지에서 함께 유지되는 기존 id만 허용.
    incoming_ids = {a.get("id") for a in incoming
                    if isinstance(a, dict) and a.get("id") in existing}
    saved_ids = []
    try:
        for a in incoming:
            if not isinstance(a, dict):
                raise HTTPException(400, "각 annotation은 객체여야 합니다")
            bbox = a.get("bbox")
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox)
                    or bbox[2] <= 0 or bbox[3] <= 0):
                raise HTTPException(400, "bbox는 숫자 4개의 배열이어야 합니다")
            if not isinstance(a.get("class_name"), str) or not a["class_name"].strip():
                raise HTTPException(400, "class_name은 비어 있을 수 없습니다")

            old = existing.get(a.get("id"))
            if a.get("meta") is not None and not isinstance(a["meta"], dict):
                raise HTTPException(400, "meta는 객체여야 합니다")
            meta = dict(a.get("meta") or {})
            identity = [bbox, a["class_name"]]
            # 화면을 연 뒤 VLM 판정이 도착한 경우 클라이언트 사본에는 vlm이 없다.
            # 같은 객체 스냅샷일 때만 현재 DB 판정을 병합하고, 박스/클래스가
            # 바뀌었으면 낡은 판정은 즉시 제거한다.
            old_vlm = (old.get("meta") or {}).get("vlm") if old else None
            sent_vlm = meta.get("vlm") if isinstance(meta.get("vlm"), dict) else None
            if sent_vlm and sent_vlm.get("box") is not None and sent_vlm.get("box") != identity:
                meta.pop("vlm", None)
            elif not sent_vlm and old_vlm and old_vlm.get("box") == identity:
                meta["vlm"] = old_vlm

            parent_id = a.get("parent_annotation_id")
            if parent_id not in incoming_ids:
                parent_id = None
            values = (
                a["class_name"].strip(), json.dumps(bbox),
                json.dumps(a.get("segmentation")) if a.get("segmentation") else None,
                a.get("confidence"), parent_id, a.get("source", "human"),
            )
            meta_json = json.dumps(meta, ensure_ascii=False)
            if old:
                same_identity = [old["bbox"], old["class_name"]] == identity
                if not sent_vlm and same_identity:
                    # SELECT 이후 UPDATE 직전에 VLM 스레드가 판정을 기록해도 같은
                    # SQL 문 안에서 현재 meta.vlm을 다시 병합한다. Python에서 읽은
                    # old_vlm만 쓰면 이 짧은 경합 창에서 유료 판정이 유실된다.
                    conn.execute(
                        "UPDATE annotations SET class_name=?, bbox=?, segmentation=?, "
                        "confidence=?, parent_annotation_id=?, source=?, "
                        "meta=CASE WHEN json_type(meta, '$.vlm') IS NOT NULL "
                        "THEN json_set(?, '$.vlm', json_extract(meta, '$.vlm')) ELSE ? END "
                        "WHERE id=? AND image_id=?",
                        (*values, meta_json, meta_json, old["id"], iid))
                else:
                    conn.execute(
                        "UPDATE annotations SET class_name=?, bbox=?, segmentation=?, "
                        "confidence=?, parent_annotation_id=?, source=?, meta=? "
                        "WHERE id=? AND image_id=?",
                        (*values, meta_json, old["id"], iid))
                saved_ids.append(old["id"])
            else:
                cur = conn.execute(
                    "INSERT INTO annotations (image_id, class_name, bbox, segmentation, "
                    "confidence, parent_annotation_id, source, meta) VALUES (?,?,?,?,?,?,?,?)",
                    (iid, *values, meta_json))
                saved_ids.append(cur.lastrowid)

        removed = set(existing) - set(saved_ids)
        if removed:
            marks = ",".join("?" * len(removed))
            conn.execute(f"DELETE FROM annotations WHERE id IN ({marks})", tuple(removed))
        conn.commit()
        rows = []
        for aid in saved_ids:
            rows.append(row_to_dict(conn.execute(
                "SELECT * FROM annotations WHERE id=?", (aid,)).fetchone()))
        return {"ok": True, "annotations": rows}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------- SAM 임베딩 (브라우저 디코더용) ----------

@app.get("/api/images/{iid}/embed")
def embed(iid: int):
    # 모델이 꺼져 있으면 503으로 분명히 알린다. 프론트는 임베딩 실패를 조용히
    # 넘기므로(SAM 클릭만 비활성) 라벨링 자체는 계속된다.
    if ml.NO_MODELS:
        raise HTTPException(503, "모델 로딩이 꺼져 있습니다 (AUTOLABEL_NO_MODELS=1)")
    return ml.embed_image(_image_path(iid).read_bytes())


# ---------- 오토라벨 ----------

def _detect_auto(pid: int, image: Image.Image, ontology: list, engine: str = "auto",
                 profile: str = "balanced", candidate_conf: float = 0.10,
                 class_routes: dict[str, str] | None = None):
    """엔진 라우팅: 활성 학생 모델 우선, 없으면 파운데이션(GDINO).

    온톨로지에 '부모.자식' 표기가 있으면 part 캐스케이드까지 수행한다.
    반환: (검출, 사용 엔진)
    """
    from server import parts, tiling

    parent_onto, parts_by_parent = parts.parse_ontology(ontology)
    # 고해상도 이미지는 타일링으로 작은 객체 회수율을 올린다 (SAHI 패턴)
    use_tiles = tiling.should_tile(image)
    student = train.active_model(pid) if engine in ("auto", "student") else None
    if student:
        recall = profile == "recall"
        student_ontology = ([{**c, "threshold": min(float(c.get("threshold", 0.35)),
                                                     candidate_conf)} for c in ontology]
                            if recall else ontology)
        detect_student = lambda im: ml.detect_student(  # noqa: E731
            im, student, student_ontology, augment=recall)
        dets = (tiling.detect_tiled(image, detect_student)
                if use_tiles else detect_student(image))
        used = f"student(mAP50 {student['map50']})" + ("+tiled" if use_tiles else "")
        if recall:
            used += f"+recall(conf {candidate_conf:g},TTA)"
        # 학생 모델이 part까지 학습했으면 캐스케이드 불필요
        if parts_by_parent and not any("." in d["class_name"] for d in dets):
            dets = dets + parts.detect_with_parts(image, ontology, dets)
            used += "+parts"
        for det in dets:
            det["meta"] = {**(det.get("meta") or {}), "model_id": student["id"]}
        return dets, used

    # 파운데이션 추론·장애 폴백·클래스별 경로 선택은 한 모듈에서 관리한다.
    # main은 학생 모델/part/저장 파이프라인만 조정한다.
    onto_use = parent_onto or ontology
    dets, used = foundation.detect(
        image, onto_use, engine=engine, class_routes=class_routes)
    if parts_by_parent:
        dets = dets + parts.detect_with_parts(image, ontology, dets)
        used += "+parts"
    return dets, used


@app.post("/api/images/{iid}/autolabel")
def autolabel_one(iid: int, body: dict | None = None):
    """단일 이미지 오토라벨 (프리뷰용). 결과는 저장하지 않고 반환만."""
    body = body or {}
    conn = get_db()
    img = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
    if not img:
        conn.close()
        raise HTTPException(404, "이미지 없음")
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (img["project_id"],)).fetchone()
    ontology = body.get("ontology") or json.loads(proj["ontology"])
    if not ontology:
        conn.close()
        raise HTTPException(400, "온톨로지가 비어 있음")
    class_routes = foundation.useful_routes(
        foundation.build_profile(conn, img["project_id"], ontology))
    conn.close()
    profile, candidate_conf = _autolabel_options(body)
    image = Image.open(_image_path(iid)).convert("RGB")
    dets, engine = _detect_auto(img["project_id"], image, ontology,
                                body.get("engine", "auto"), profile, candidate_conf,
                                class_routes=class_routes)
    for d in dets:
        d["meta"] = {**(d.get("meta") or {}), "engine": engine,
                     "profile": profile, "ontology": ontology}
    if body.get("masks", True):
        rles = ml.boxes_to_masks(image, [d["bbox"] for d in dets])
        for d, r in zip(dets, rles):
            d["segmentation"] = r
    return {"detections": dets, "engine": engine, "profile": profile,
            "agreement_counts": ensemble.agreement_counts(dets)}


def _batch_verdict(job: dict, total: int, ontology: list[str] | None = None) -> dict:
    """배치 결과를 읽고 다음에 뭘 해야 하는지 정한다.

    제로샷이 특수 도메인에서 거의 못 잡는 건 정상이다(README 참조). 문제는
    그때 앱이 아무 말도 안 해서, 사용자가 빈 캔버스를 보며 도구가 고장난
    줄 안다는 것이다. 검출률을 근거로 다음 수를 지정해준다.
    """
    hit, found = job.get("hit", 0), job.get("found", 0)
    class_counts = job.get("class_counts") or {}
    large_boxes = job.get("large_boxes", 0)
    ontology = ontology or []
    rate = hit / max(total, 1)
    diagnostics = {
        "class_counts": class_counts,
        "missing_classes": [name for name in ontology if not class_counts.get(name)],
        "large_boxes": large_boxes,
        "agreement_counts": job.get("agreement_counts") or {},
        "ensemble_pilot": job.get("ensemble_pilot") or {},
    }
    agreement = diagnostics["agreement_counts"]
    ensemble_total = sum(agreement.values())
    consensus = agreement.get("consensus", 0)
    solo = agreement.get("sam3_only", 0) + agreement.get("gdino_only", 0)
    ensemble_advice = (f" 두 모델 합의 {consensus}개 · 단독 후보 {solo}개이며, "
                       "단독 후보부터 검수하세요." if ensemble_total else "")
    pilot = diagnostics["ensemble_pilot"]
    pilot_advice = (f" 초기 {pilot.get('images', 0)}장 교차 시험에서 합의가 낮아 "
                    "나머지는 SAM3로 자동 전환했습니다."
                    if pilot.get("decision") == "sam3" else "")
    if hit == 0:
        return {"verdict": "empty", **diagnostics, "advice":
                "한 장도 못 찾았습니다. 검출 프롬프트를 영어로 바꾸거나 "
                "'프롬프트 실험'으로 후보를 비교해 보세요. 그래도 안 되면 직접 "
                "몇 장 그린 뒤 전용 모델을 학습시키는 편이 빠릅니다."}

    # '무언가를 찾은 비율'은 정확도가 아니다. 실제 철강 표면 테스트에서는
    # 22/30장 검출로 종전 판정은 good이었지만, 정답과 클래스+IoU50가 맞은
    # 박스는 7/55뿐이었다. 정답 라벨 없이도 드러나는 붕괴 신호를 함께 본다.
    warnings = []
    enough_to_compare = found >= max(10, len(ontology) * 2)
    if enough_to_compare and diagnostics["missing_classes"]:
        names = ", ".join(diagnostics["missing_classes"][:4])
        suffix = " 외" if len(diagnostics["missing_classes"]) > 4 else ""
        warnings.append(f"한 번도 나온 적 없는 클래스: {names}{suffix}")
    if found >= 10 and large_boxes / max(found, 1) >= 0.15:
        warnings.append(f"화면의 80% 이상을 덮는 큰 박스 {large_boxes}개")
    if enough_to_compare and len(class_counts) >= 1 and len(ontology) >= 2:
        dominant_name, dominant_count = max(class_counts.items(), key=lambda item: item[1])
        if dominant_count / max(sum(class_counts.values()), 1) >= 0.75:
            warnings.append(f"{dominant_name} 한 클래스에 예측이 과도하게 몰림")
    if ensemble_total >= 10 and consensus / ensemble_total < 0.25:
        warnings.append(f"SAM3·GDINO 합의가 {consensus}/{ensemble_total}개로 낮음")

    if rate >= 0.7 and not warnings:
        return {"verdict": "good", **diagnostics,
                "advice": f"{total}장 중 {hit}장에서 검출 · 박스 {found}개."
                          f"{pilot_advice}{ensemble_advice} 리뷰를 시작하세요."}

    if warnings:
        return {"verdict": "partial", **diagnostics, "warnings": warnings,
                "advice": f"{total}장 중 {hit}장에서 {found}개를 찾았지만 품질 경고가 있습니다: "
                          f"{' · '.join(warnings)}. 검출률만으로 정확하다고 보지 말고, "
                          "클래스별 대표 이미지를 먼저 확인한 뒤 프롬프트를 조정하거나 "
                          f"직접 라벨로 전용 모델을 학습하세요.{pilot_advice}{ensemble_advice}"}
    # 중간 커버리지는 단정하지 않는다 — 대상이 없는 이미지가 섞인 데이터셋에선
    # 낮은 검출률이 정답이다 (실측: 개 9장+비개 6장에서 9/15 검출 = 만점인데
    # "제로샷 약함"으로 오판해 프롬프트 실험·수동 라벨로 오도했다)
    return {"verdict": "partial", "advice":
            f"{total}장 중 {hit}장에서 검출({found}개). 못 찾은 {total - hit}장에 실제로 "
            "대상이 없다면 정상입니다 — 빈 이미지 몇 장을 열어 누락인지 확인하세요. "
            "누락이 많으면 '프롬프트 실험'으로 표현을 고르거나, 직접 라벨 수십 장으로 "
            f"전용 모델을 학습시키면 급격히 좋아집니다.{pilot_advice}{ensemble_advice}", **diagnostics}


def _box_area_ratio(bbox: list[float], image: Image.Image) -> float:
    """DB 표준 bbox([x, y, width, height])가 이미지에서 차지하는 비율."""
    _x, _y, width, height = bbox
    return max(0, width) * max(0, height) / max(image.width * image.height, 1)


def _run_batch(pid: int, image_ids: list[int], ontology: list[dict], masks: bool,
               profile: str = "balanced", candidate_conf: float = 0.10):
    found = hit = saved = suppressed = large_boxes = 0
    class_counts: dict[str, int] = {}
    agreement_totals = {"consensus": 0, "sam3_only": 0, "gdino_only": 0}
    has_student = train.active_model(pid) is not None
    conn = get_db()
    learned_profile = foundation.build_profile(conn, pid, ontology)
    class_routes = foundation.useful_routes(learned_profile)
    foundation_engine = ("auto" if has_student else
                         "routed" if class_routes else
                         "ensemble" if ml.sam3_available() else "foundation")
    pilot_images = 0
    pilot_decision = ("student" if has_student else
                      "calibrated" if class_routes else "testing")
    try:
        for n, iid in enumerate(image_ids, 1):
            image = Image.open(_image_path(iid)).convert("RGB")
            dets, used_engine = _detect_auto(
                pid, image, ontology, engine=foundation_engine,
                profile=profile, candidate_conf=candidate_conf,
                class_routes=class_routes if foundation_engine == "routed" else None)
            # 사람이 후보를 삭제하거나 박스를 고쳐도 원본 예측은 감사 테이블에
            # 남는다. 승인이 끝난 표본부터 다음 배치의 클래스별 경로에 반영된다.
            foundation.replace_audit(conn, pid, iid, dets, used_engine)
            raw_agreement = ensemble.agreement_counts(dets)
            for key, value in raw_agreement.items():
                agreement_totals[key] += value
            if foundation_engine == "ensemble":
                pilot_images += 1
                keep_ensemble = ensemble.pilot_should_continue(
                    agreement_totals, pilot_images)
                if keep_ensemble is False:
                    foundation_engine = "sam3"
                    pilot_decision = "sam3"
                elif keep_ensemble is True:
                    pilot_decision = "ensemble"
            detected_count = len(dets)
            # 재실행은 낡은 모델 초안만 교체한다. 사람이 직접 만들거나 고친
            # 같은 클래스 박스는 정답으로 보고, 겹치는 새 초안을 추가하지 않는다.
            from server import tiling
            trusted = [row_to_dict(r) for r in conn.execute(
                "SELECT * FROM annotations WHERE image_id=? AND source!='model'", (iid,))]
            dets, n_suppressed = tiling.suppress_trusted_overlaps(dets, trusted)
            suppressed += n_suppressed
            for det in dets:
                name = det["class_name"]
                class_counts[name] = class_counts.get(name, 0) + 1
                large_boxes += int(_box_area_ratio(det["bbox"], image) >= 0.80)
            rles = ml.boxes_to_masks(image, [d["bbox"] for d in dets]) if masks else []
            conn.execute(
                "DELETE FROM annotations WHERE image_id=? AND source='model'", (iid,))
            # 부모 먼저 저장하고 그 id를 part에 연결 (계층 라벨)
            parent_ids: dict[int, int] = {}
            for i, d in enumerate(dets):
                cur = conn.execute(
                    "INSERT INTO annotations (image_id, class_name, bbox, segmentation, "
                    "confidence, parent_annotation_id, source, meta) VALUES (?,?,?,?,?,?,?,?)",
                    (iid, d["class_name"], json.dumps(d["bbox"]),
                     json.dumps(rles[i]) if i < len(rles) else None,
                     d["confidence"],
                     parent_ids.get(d.get("_parent_index")) if "_parent_index" in d else None,
                     "model",
                     json.dumps({**(d.get("meta") or {}), "engine": used_engine,
                                 "profile": profile, "ontology": ontology})))
                if "_parent_index" not in d:
                    parent_ids[i] = cur.lastrowid
            # 명시 재실행으로 승인 이미지를 골랐더라도 새 모델 초안을 승인 상태로
            # 남기면 학습 정답으로 즉시 오염된다. 처리한 이미지는 항상 다시 검수한다.
            conn.execute("UPDATE images SET status='prelabeled' WHERE id=?", (iid,))
            conn.commit()
            found += detected_count
            hit += 1 if detected_count else 0
            saved += len(dets)
            jobs.update("autolabel", pid, done=n, total=len(image_ids),
                        found=found, saved=saved, hit=hit,
                        class_counts=class_counts, large_boxes=large_boxes,
                        agreement_counts=agreement_totals,
                        ensemble_pilot={"images": pilot_images, "decision": pilot_decision},
                        suppressed_human_overlap=suppressed)
        verdict = _batch_verdict(
            {"found": found, "hit": hit, "class_counts": class_counts,
             "large_boxes": large_boxes, "agreement_counts": agreement_totals,
             "ensemble_pilot": {"images": pilot_images, "decision": pilot_decision}},
            len(image_ids), [item["name"] for item in ontology])
        if suppressed:
            verdict["advice"] += f" 사람 라벨과 겹친 중복 {suppressed}개는 저장하지 않았습니다."
        jobs.update("autolabel", pid, status="completed", profile=profile,
                    found=found, saved=saved, hit=hit,
                    foundation_profile=foundation.build_profile(conn, pid, ontology),
                    suppressed_human_overlap=suppressed, **verdict)
    except Exception as e:  # 잡 실패를 상태로 노출
        jobs.update("autolabel", pid, status="failed", error=str(e))
    finally:
        conn.close()


@app.post("/api/projects/{pid}/autolabel")
def autolabel_batch(pid: int, body: dict):
    """배치 오토라벨 — 백그라운드 스레드 (MVP)."""
    if jobs.get("autolabel", pid).get("status") == "running":
        raise HTTPException(409, "이미 실행 중인 잡 있음")
    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if not proj:
        conn.close()
        raise HTTPException(404, "프로젝트 없음")
    try:
        profile, candidate_conf = _autolabel_options(body)
    except Exception:
        conn.close()
        raise
    ontology = body.get("ontology") or json.loads(proj["ontology"])
    # 기본 대상은 리뷰 전 이미지만. 승인 이미지를 포함하면 사람이 검토한 라벨을
    # 무검토 검출로 교체하면서 status는 approved로 남아, 오염된 라벨이 승인
    # 데이터로 둔갑해 학습셋에 들어간다. 재실행이 필요하면 image_ids로 명시.
    if "image_ids" in body:
        try:
            ids = _scoped_image_ids(conn, pid, body["image_ids"])
        except Exception:
            conn.close()
            raise
    else:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM images WHERE project_id=? "
            "AND status IN ('unlabeled','prelabeled')", (pid,))]
    conn.close()
    if not ids:
        return {"status": "completed", "done": 0, "total": 0,
                "advice": "라벨할 이미지가 없습니다 — 리뷰 전(unlabeled/prelabeled) "
                          "이미지가 없습니다. 승인·거부된 라벨은 덮어쓰지 않습니다."}
    ok, st = jobs.try_start(
        "autolabel", pid, done=0, total=len(ids), profile=profile)
    if not ok:
        raise HTTPException(409, "이미 실행 중인 잡 있음")
    threading.Thread(
        target=_run_batch,
        args=(pid, ids, ontology, body.get("masks", True), profile, candidate_conf),
        daemon=True).start()
    return st


@app.get("/api/projects/{pid}/autolabel/status")
def autolabel_status(pid: int):
    # jobs가 메모리 캐시 → 디스크 순으로 본다 — 서버 재시작 후에도 중단
    # (interrupted)을 알리고, 기록이 없으면 진짜 미실행(idle)이다.
    return jobs.get("autolabel", pid)


@app.get("/api/projects/{pid}/foundation-profile")
def foundation_profile(pid: int):
    """승인된 교차 시험 표본으로 계산한 클래스별 SAM3/GDINO 선택 근거."""
    conn = get_db()
    project = conn.execute("SELECT ontology FROM projects WHERE id=?", (pid,)).fetchone()
    if not project:
        conn.close()
        raise HTTPException(404, "프로젝트 없음")
    result = foundation.build_profile(conn, pid, json.loads(project["ontology"]))
    conn.close()
    return result


MAX_LAB_PROMPTS = 8  # 표본 5장 기준 이 이상은 대기가 너무 길어진다


def _unique_prompts(raw: list, limit: int = MAX_LAB_PROMPTS) -> list[str]:
    """후보 프롬프트 정리 — 공백 제거·중복 제거·상한. 입력 순서는 유지.

    같은 프롬프트를 두 번 돌리는 건 추론 시간 낭비고, 결과표에 똑같은 줄이
    두 개 뜨면 어느 쪽을 고르라는 건지 알 수 없다.
    """
    seen = dict.fromkeys(p.strip() for p in raw if isinstance(p, str) and p.strip())
    return list(seen)[:limit]


@app.post("/api/projects/{pid}/prompt-lab")
def prompt_lab(pid: int, body: dict):
    """후보 프롬프트들을 표본 이미지에 돌려 비교한다.

    제로샷 품질은 프롬프트가 좌우하는데, 지금까지는 프롬프트를 바꿔 전체를
    다시 돌려보는 것 말고 비교할 방법이 없었다. 몇 장만으로 후보를 견주면
    전체 배치 전에 고를 수 있다.
    """
    prompts = _unique_prompts(body.get("prompts", []))
    if not prompts:
        raise HTTPException(400, "비교할 프롬프트가 없음")
    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if not proj:
        conn.close()
        raise HTTPException(404, "프로젝트 없음")
    onto = json.loads(proj["ontology"])
    cls = body.get("class_name") or (onto[0]["name"] if onto else "object")
    thr = float(body.get("threshold", 0.25))
    n = max(1, min(int(body.get("n_images", 5)), 20))
    # 라벨이 이미 있는 이미지를 우선 표본으로 — 정답과 대조할 수 있다
    rows = conn.execute(
        "SELECT i.*, (SELECT COUNT(*) FROM annotations a WHERE a.image_id=i.id) AS ann_count "
        "FROM images i WHERE i.project_id=? ORDER BY ann_count DESC, i.id LIMIT ?",
        (pid, n)).fetchall()
    conn.close()
    if not rows:
        raise HTTPException(400, "이미지가 없음")

    images = [(r, Image.open(_row_image_path(r)).convert("RGB")) for r in rows
              if _row_image_path(r).exists()]
    results = []
    for p in prompts:
        one = [{"name": cls, "prompt": p, "threshold": thr}]
        hit = found = 0
        confs = []
        for _row, img in images:
            dets = ml.detect(img, one)
            found += len(dets)
            hit += 1 if dets else 0
            confs += [d["confidence"] for d in dets]
        results.append({
            "prompt": p,
            "images_with_detection": hit,
            "detections": found,
            "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
            # 장당 박스가 지나치게 많으면 과검출 — 개수만 보고 고르면 속는다
            "per_image": round(found / max(len(images), 1), 2),
        })
    # 커버리지 우선, 같으면 확신도. 과검출은 per_image로 사용자가 판단한다
    results.sort(key=lambda r: (-r["images_with_detection"], -r["avg_confidence"]))
    return {"class_name": cls, "sampled_images": len(images),
            "threshold": thr, "results": results, "best": results[0]["prompt"]}


# ---------- 시각 예시 검출 ----------

@app.post("/api/images/{iid}/exemplar")
def exemplar(iid: int, body: dict):
    """예시 박스 1개 → 같은 이미지에서 유사 객체 전부 검출.

    전용 모델이 있으면 그 후보를 예시 유사도로 거르는 경로가 훨씬 정확하다
    (특징맵 피크 방식은 미세 객체에서 무력). 없으면 피크 탐색으로 폴백.
    """
    import numpy as np

    pil = Image.open(_image_path(iid)).convert("RGB")
    image = np.array(pil)
    cls = body.get("class_name", "")

    conn = get_db()
    im = conn.execute("SELECT project_id FROM images WHERE id=?", (iid,)).fetchone()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (im["project_id"],)).fetchone()
    conn.close()
    student = train.active_model(im["project_id"])

    mode = "peak"
    if student and body.get("use_model", True):
        ontology = [{**c, "threshold": 0.15} for c in json.loads(proj["ontology"])]
        cands = ml.detect_student(pil, student, ontology)
        # 예시 자신과 겹치는 후보는 제외 (이미 아는 것)
        dets = ml.exemplar_rerank(image, body["bbox"], cands,
                                  sim_thr=float(body.get("sim_thr", 0.55)))
        mode = "model+exemplar"
        if not dets:  # 모델이 아무것도 못 주면 피크 탐색으로 폴백
            dets = ml.exemplar_detect(image, body["bbox"],
                                      topk=int(body.get("topk", 20)),
                                      sim_thr=float(body.get("sim_thr", 0.6)))
            mode = "peak(fallback)"
    else:
        dets = ml.exemplar_detect(
            image, body["bbox"],
            topk=int(body.get("topk", 20)),
            sim_thr=float(body.get("sim_thr", 0.6)))

    for d in dets:
        if mode.startswith("peak"):
            d["class_name"] = cls
        elif cls:
            d["class_name"] = cls  # 사용자가 지정한 클래스로 통일
    return {"detections": dets, "mode": mode}


# ---------- 데이터셋 연결 임포트 ----------

@app.post("/api/import/preview")
def import_preview(body: dict):
    """폴더 경로만 주면 이미지 수·라벨 형식·클래스를 미리 알려준다."""
    if not body.get("images_dir"):
        raise HTTPException(400, "images_dir 필수")
    return importer.preview(body["images_dir"], body.get("labels_dir"), body.get("coco_json"))


@app.post("/api/projects/{pid}/import")
def import_dataset(pid: int, body: dict):
    """복사 없이 폴더를 연결하고 기존 라벨을 가져온다 (대용량 데이터셋용)."""
    if not body.get("images_dir"):
        raise HTTPException(400, "images_dir 필수")
    return importer.start_import(pid, body)


@app.get("/api/projects/{pid}/import/status")
def import_status(pid: int):
    return importer.job_status(pid)  # jobs가 메모리→디스크 폴백을 이미 담당


# ---------- 외부 모델 임포트 ----------

def _model_metric(value, field: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{field}는 0~1 사이 숫자여야 합니다")
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise HTTPException(400, f"{field}는 0~1 사이 숫자여야 합니다")
    return result


def _model_class_metrics(value) -> dict | None:
    """검증 번들의 클래스별 홀드아웃 지표를 작은 안전한 계약으로 정규화한다."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise HTTPException(400, "class_metrics는 클래스별 성능 객체여야 합니다")
    cleaned = {}
    for name, row in value.items():
        if not isinstance(name, str) or not 0 < len(name) <= 120 or not isinstance(row, dict):
            raise HTTPException(400, "class_metrics 형식이 올바르지 않습니다")
        instances = row.get("test_instances")
        if isinstance(instances, bool) or not isinstance(instances, int) or instances < 0:
            raise HTTPException(400, f"class_metrics.{name}.test_instances는 0 이상 정수여야 합니다")
        cleaned[name] = {
            "test_map50": _model_metric(row.get("test_map50"),
                                          f"class_metrics.{name}.test_map50"),
            "test_map50_95": _model_metric(row.get("test_map50_95"),
                                             f"class_metrics.{name}.test_map50_95"),
            "test_instances": instances,
        }
    return cleaned


def _model_quality(metrics: dict, split_counts: dict, class_metrics: dict | None = None,
                   classes: list[str] | None = None) -> tuple[str, str]:
    """학습 성공과 전문 검증을 분리한다. 낮은 점수는 절대 자동 적용하지 않는다."""
    val_map50 = _model_metric(metrics.get("val_map50"), "val_map50")
    test_map50 = _model_metric(metrics.get("test_map50"), "test_map50")
    test_images = int(split_counts.get("test", 0) or 0)
    if test_images >= train.MIN_TEST and test_map50 is not None:
        if test_map50 < MODEL_QUALITY_FLOOR:
            return "failed", f"홀드아웃 mAP50 {test_map50:.3f}가 품질 하한 {MODEL_QUALITY_FLOOR:.2f} 미만입니다"
        if class_metrics is not None and classes:
            weak = [(name, class_metrics[name]["test_map50"])
                    for name in classes if name in class_metrics
                    and class_metrics[name]["test_instances"] >= MIN_CLASS_TEST_INSTANCES
                    and class_metrics[name]["test_map50"] is not None
                    and class_metrics[name]["test_map50"] < MODEL_CLASS_QUALITY_FLOOR]
            if weak:
                detail = ", ".join(f"{name} {score:.3f}" for name, score in weak)
                return "failed", (f"취약 클래스 홀드아웃 mAP50이 {MODEL_CLASS_QUALITY_FLOOR:.2f} 미만입니다: "
                                  f"{detail}")
            unsupported = [name for name in classes if name not in class_metrics
                           or class_metrics[name]["test_instances"] < MIN_CLASS_TEST_INSTANCES
                           or class_metrics[name]["test_map50"] is None]
            if unsupported:
                return "unverified", ("클래스별 홀드아웃 근거가 부족합니다: "
                                      + ", ".join(unsupported))
        return "verified", f"독립 홀드아웃 {test_images}장에서 품질 하한을 통과했습니다"
    if val_map50 is None:
        return "unverified", "성능표가 없어 적용 전 검증이 필요합니다"
    if val_map50 < MODEL_QUALITY_FLOOR:
        return "failed", f"validation mAP50 {val_map50:.3f}가 품질 하한 {MODEL_QUALITY_FLOOR:.2f} 미만입니다"
    return "provisional", "validation은 통과했지만 충분한 독립 홀드아웃이 없어 실험 모델입니다"


def _project_model_classes(pid: int) -> tuple[list[str], str]:
    conn = get_db()
    row = conn.execute("SELECT name, ontology FROM projects WHERE id=?", (pid,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "프로젝트 없음")
    ontology = json.loads(row["ontology"] or "[]")
    names = [c.get("name") for c in ontology
             if isinstance(c, dict) and isinstance(c.get("name"), str) and c["name"]]
    return names, row["name"]


def _validate_model_classes(expected: list[str], actual) -> list[str]:
    if not isinstance(actual, list) or not actual or not all(
            isinstance(name, str) and 0 < len(name) <= 120 for name in actual):
        raise HTTPException(400, "모델 클래스명 목록이 올바르지 않습니다")
    if actual != expected:
        raise HTTPException(400, "모델 클래스 순서가 프로젝트와 다릅니다 — "
                            f"프로젝트 {expected}, 모델 {actual}")
    return actual


def _quality_from_model(model: dict) -> str:
    meta = model.get("meta") or {}
    if meta.get("quality_status"):
        return meta["quality_status"]
    score = model.get("test_map50")
    if score is not None:
        return "verified" if score >= MODEL_QUALITY_FLOOR else "failed"
    score = model.get("map50")
    if score is not None:
        return "provisional" if score >= MODEL_QUALITY_FLOOR else "failed"
    return "unverified"


def _safe_bundle_files(path: Path) -> tuple[dict, str]:
    """번들 메타와 모델 멤버를 검증한다. 디스크에 풀지 않아 zip-slip을 없앤다."""
    try:
        with zipfile.ZipFile(path) as z:
            infos = [info for info in z.infolist() if not info.is_dir()]
            if not infos or len(infos) > MAX_BUNDLE_MEMBERS:
                raise HTTPException(400, f"모델 번들은 파일 {MAX_BUNDLE_MEMBERS}개 이하여야 합니다")
            total = 0
            for info in infos:
                normalized = info.filename.replace("\\", "/")
                member = PurePosixPath(normalized)
                unix_mode = (info.external_attr >> 16) & 0o170000
                if (not normalized or "\x00" in normalized or member.is_absolute()
                        or ".." in member.parts or (member.parts and ":" in member.parts[0])
                        or unix_mode == 0o120000):
                    raise HTTPException(400, f"안전하지 않은 번들 경로입니다: {info.filename}")
                total += info.file_size
                if info.file_size > MAX_MODEL_BYTES or total > MAX_MODEL_BYTES + MAX_MANIFEST_BYTES:
                    raise HTTPException(400, "모델 번들의 압축 해제 크기가 제한을 넘습니다")
            manifests = [i for i in infos if PurePosixPath(i.filename).name == "autolabel-model.json"]
            models = [i for i in infos if PurePosixPath(i.filename).suffix.lower() == ".pt"]
            if len(manifests) != 1 or len(models) != 1:
                raise HTTPException(400, "번들에는 best.pt와 autolabel-model.json이 각각 하나 필요합니다")
            if manifests[0].file_size > MAX_MANIFEST_BYTES:
                raise HTTPException(400, "모델 성능표가 너무 큽니다")
            try:
                manifest = json.loads(z.read(manifests[0]).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise HTTPException(400, "autolabel-model.json 형식이 올바르지 않습니다")
    except zipfile.BadZipFile:
        raise HTTPException(400, "손상된 모델 번들입니다")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MODEL_BUNDLE_SCHEMA:
        raise HTTPException(400, f"지원하는 모델 번들 schema_version은 {MODEL_BUNDLE_SCHEMA}입니다")
    counts = manifest.get("split_counts")
    if not isinstance(counts, dict):
        raise HTTPException(400, "split_counts가 필요합니다")
    split_counts = {}
    for split in ("train", "val", "test"):
        value = counts.get(split)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HTTPException(400, f"split_counts.{split}은 0 이상 정수여야 합니다")
        split_counts[split] = value
    approved = manifest.get("approved_images")
    if isinstance(approved, bool) or not isinstance(approved, int) or approved <= 0:
        raise HTTPException(400, "approved_images는 1 이상 정수여야 합니다")
    if approved != sum(split_counts.values()):
        raise HTTPException(400, "approved_images와 train/val/test 합계가 다릅니다")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        raise HTTPException(400, "metrics가 필요합니다")
    clean_metrics = {key: _model_metric(metrics.get(key), key) for key in (
        "val_map50", "val_map50_95", "test_map50", "test_map50_95")}
    clean_class_metrics = _model_class_metrics(manifest.get("class_metrics"))
    architecture = manifest.get("architecture")
    if not isinstance(architecture, str) or not 0 < len(architecture) <= 80:
        raise HTTPException(400, "architecture가 올바르지 않습니다")
    epochs = manifest.get("epochs_requested")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 500:
        raise HTTPException(400, "epochs_requested는 1~500 사이 정수여야 합니다")
    clean = {
        "schema_version": MODEL_BUNDLE_SCHEMA,
        "classes": manifest.get("classes"),
        "architecture": architecture,
        "approved_images": approved,
        "split_counts": split_counts,
        "epochs_requested": epochs,
        "metrics": clean_metrics,
        "class_metrics": clean_class_metrics,
    }
    return clean, models[0].filename


def _store_model(pid: int, source, *, bundle: Path | None = None) -> Path:
    """사용자 다운로드 경로에 의존하지 않도록 모델을 앱 관리 디렉터리에 복사한다."""
    target_dir = MODELS_DIR / str(pid)
    target_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".model-", suffix=".part", dir=target_dir)
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(fd, "wb") as out:
            if bundle is None:
                stream_ctx = open(source, "rb")
            else:
                archive = zipfile.ZipFile(bundle)
                stream_ctx = archive.open(source)
            try:
                while True:
                    chunk = stream_ctx.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_MODEL_BYTES:
                        raise HTTPException(400, "모델 파일이 512MB 제한을 넘습니다")
                    digest.update(chunk)
                    out.write(chunk)
            finally:
                stream_ctx.close()
                if bundle is not None:
                    archive.close()
        if total == 0:
            raise HTTPException(400, "모델 파일이 비어 있습니다")
        destination = target_dir / f"{digest.hexdigest()[:20]}.pt"
        if destination.exists():
            os.unlink(tmp_name)
        else:
            os.replace(tmp_name, destination)
        return destination
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise

@app.post("/api/projects/{pid}/models/import")
def import_model(pid: int, body: dict):
    """Colab 번들 또는 외부 .pt를 검증·보관한다. 등록과 적용은 별도 결정이다."""
    raw_path = body.get("path") if isinstance(body, dict) else None
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise HTTPException(400, "모델 경로가 필요합니다")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise HTTPException(400, f"모델 파일 없음: {path}")
    if path.stat().st_size > MAX_MODEL_BYTES:
        raise HTTPException(400, "모델 파일이 512MB 제한을 넘습니다")
    source_name = Path(body.get("_source_name") or path.name).name
    expected_names, _project_name = _project_model_classes(pid)

    if path.suffix.lower() == ".zip":
        manifest, model_member = _safe_bundle_files(path)
        names = _validate_model_classes(expected_names, manifest["classes"])
        class_metrics = manifest["class_metrics"]
        if class_metrics is not None and not set(class_metrics).issubset(names):
            raise HTTPException(400, "class_metrics에 프로젝트에 없는 클래스가 있습니다")
        metrics = manifest["metrics"]
        split_counts = manifest["split_counts"]
        train_images = manifest["approved_images"]
        arch = manifest["architecture"]
        stored_path = _store_model(pid, model_member, bundle=path)
        bundle_meta = {"schema_version": manifest["schema_version"],
                       "epochs_requested": manifest["epochs_requested"],
                       "source_bundle": source_name}
    elif path.suffix.lower() == ".pt":
        names = body.get("names")
        if not names:
            try:
                from ultralytics import YOLO

                m = YOLO(str(path))
                names = [m.names[i] for i in sorted(m.names)]
            except Exception as e:
                raise HTTPException(400, f"클래스명 자동 추출 실패 — names를 넘겨주세요 ({e})")
        names = _validate_model_classes(expected_names, names)
        metrics = {
            "val_map50": _model_metric(body.get("map50"), "map50"),
            "val_map50_95": _model_metric(body.get("map50_95"), "map50_95"),
            "test_map50": _model_metric(body.get("test_map50"), "test_map50"),
            "test_map50_95": _model_metric(body.get("test_map50_95"), "test_map50_95"),
        }
        class_metrics = None
        try:
            train_images = int(body.get("train_images", 0))
        except (TypeError, ValueError):
            raise HTTPException(400, "train_images는 0 이상 정수여야 합니다")
        if train_images < 0:
            raise HTTPException(400, "train_images는 0 이상 정수여야 합니다")
        split_counts = {"train": train_images, "val": 0, "test": 0}
        arch = body.get("arch", "external")
        stored_path = _store_model(pid, path)
        bundle_meta = {"source_file": source_name}
    else:
        raise HTTPException(400, "지원 형식은 Colab 모델 번들(.zip) 또는 YOLO 모델(.pt)입니다")

    quality_status, quality_reason = _model_quality(
        metrics, split_counts, class_metrics, names)
    activate = bool(body.get("activate", False))
    if activate and quality_status in {"failed", "unverified"} and not body.get("force"):
        raise HTTPException(409, f"모델 품질 검증을 통과하지 못해 적용할 수 없습니다: {quality_reason}")
    meta = {
        "arch": arch,
        "names": names,
        "imported": True,
        "quality_status": quality_status,
        "quality_reason": quality_reason,
        "metrics": metrics,
        "class_metrics": class_metrics,
        "split_counts": split_counts,
        **bundle_meta,
    }
    conn = get_db()
    if activate:
        conn.execute("UPDATE models SET active=0 WHERE project_id=?", (pid,))
    cur = conn.execute(
        "INSERT INTO models (project_id, path, map50, test_map50, train_images, active, meta) "
        "VALUES (?,?,?,?,?,?,?)",
        (pid, str(stored_path), metrics["val_map50"], metrics["test_map50"], train_images,
         int(activate), json.dumps(meta, ensure_ascii=False)))
    mid = cur.lastrowid
    conn.commit()
    conn.close()
    return {"ok": True, "id": mid, "names": names, "active": activate,
            "quality_status": quality_status, "quality_reason": quality_reason,
            "metrics": metrics, "class_metrics": class_metrics,
            "split_counts": split_counts, "path": str(stored_path)}


@app.post("/api/projects/{pid}/models/import-upload")
async def import_model_upload(pid: int, file: UploadFile):
    """브라우저 파일 선택으로 받은 모델을 임시 보관한 뒤 기존 검증 경로로 넘긴다."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".zip", ".pt"}:
        raise HTTPException(400, "지원 형식은 Colab 모델 번들(.zip) 또는 YOLO 모델(.pt)입니다")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".model-upload-", suffix=suffix, dir=DATA_ROOT)
    total = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_MODEL_BYTES:
                    raise HTTPException(400, "모델 파일이 512MB 제한을 넘습니다")
                out.write(chunk)
        if total == 0:
            raise HTTPException(400, "모델 파일이 비어 있습니다")
        return import_model(pid, {"path": tmp_name, "_source_name": file.filename})
    finally:
        await file.close()
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


# ---------- 자동 승인 (TBAL) / 일괄 작업 ----------

@app.post("/api/projects/{pid}/auto-approve")
def auto_approve(pid: int, body: dict):
    """검증·캘리브레이션된 전용 모델의 고정밀 초안만 자동 승인.

    dry_run=true면 대상만 세어 돌려준다 (승인 전 미리보기).
    confidence는 모델마다 의미가 달라 범용 모델 점수만으로 승인하면 안 된다.
    홀드아웃 검증 모델 + QA val에서 95% 정밀도를 확인한 클래스만 통과한다.
    """
    try:
        min_conf = float(body.get("min_conf", 0.7))
    except (TypeError, ValueError):
        raise HTTPException(400, "min_conf는 0~1 사이 숫자여야 합니다")
    if not math.isfinite(min_conf) or not 0 < min_conf < 1:
        raise HTTPException(400, "min_conf는 0~1 사이 숫자여야 합니다")
    thresholds = body.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        raise HTTPException(400, "thresholds는 클래스별 숫자 객체여야 합니다")
    dry = bool(body.get("dry_run", False))

    conn = get_db()
    project = conn.execute("SELECT ontology FROM projects WHERE id=?", (pid,)).fetchone()
    if not project:
        conn.close()
        raise HTTPException(404, "프로젝트 없음")
    ontology = json.loads(project["ontology"])
    active_model = train.active_model(pid)
    verified_model = (active_model if active_model
                      and _quality_from_model(active_model) == "verified" else None)
    calibrated = {}
    for item in ontology:
        try:
            tau = float(item.get("approval_threshold"))
            precision = float(item.get("approval_precision"))
            support = int(item.get("approval_support"))
        except (TypeError, ValueError):
            continue
        if (verified_model and item.get("approval_source") == "qa_val"
                and item.get("approval_model_id") == verified_model["id"] and math.isfinite(tau)
                and 0 < tau < 1 and precision >= 0.95 and support >= 8):
            calibrated[item["name"]] = tau
    rows = conn.execute(
        "SELECT * FROM images WHERE project_id=? AND status='prelabeled'", (pid,)).fetchall()
    targets, skipped_lowconf, skipped_empty = [], 0, 0
    skipped_unsafe_model = skipped_uncalibrated = 0
    for im in rows:
        anns = [row_to_dict(a) for a in conn.execute(
            "SELECT * FROM annotations WHERE image_id=?", (im["id"],))]
        if not anns:
            skipped_empty += 1
            # '아무것도 못 찾음'은 정상 음성의 증거가 아니다. 음성 전용
            # 캘리브레이션이 없으므로 빈 초안은 항상 사람이 확인한다.
            continue
        ok = True
        skip_kind = None
        for a in anns:
            if a["source"] != "model":
                continue
            meta = a.get("meta") or {}
            if (not verified_model or meta.get("model_id") != verified_model["id"]):
                ok = False
                skip_kind = "unsafe"
                break
            if a["class_name"] not in calibrated:
                ok = False
                skip_kind = "uncalibrated"
                break
            try:
                requested = float(thresholds.get(a["class_name"], min_conf))
            except (TypeError, ValueError):
                conn.close()
                raise HTTPException(400, "thresholds 값은 0~1 사이 숫자여야 합니다")
            need = max(min_conf, calibrated[a["class_name"]], requested)
            if (a["confidence"] is None or a["confidence"] < need):
                ok = False
                skip_kind = "lowconf"
                break
        if ok:
            targets.append(im["id"])
        elif skip_kind == "unsafe":
            skipped_unsafe_model += 1
        elif skip_kind == "uncalibrated":
            skipped_uncalibrated += 1
        else:
            skipped_lowconf += 1

    if not dry and targets:
        conn.executemany("UPDATE images SET status='approved' WHERE id=?",
                         [(i,) for i in targets])
        conn.commit()
    total = len(rows)
    conn.close()
    result = {
        "pending": total, "approved": len(targets),
        "skipped_low_confidence": skipped_lowconf, "skipped_no_label": skipped_empty,
        "skipped_unsafe_model": skipped_unsafe_model,
        "skipped_uncalibrated": skipped_uncalibrated,
        "verified_model": bool(verified_model),
        "calibrated_classes": sorted(calibrated),
        "coverage": round(len(targets) / total, 3) if total else 0,
        "dry_run": dry,
    }
    if skipped_unsafe_model:
        result["blocked_reason"] = (
            "범용 모델 또는 홀드아웃 검증을 통과하지 않은 모델의 초안은 자동 승인하지 않습니다. "
            "직접 검수해 승인 라벨을 만든 뒤 전용 모델을 학습하세요.")
    elif skipped_uncalibrated:
        result["blocked_reason"] = (
            "클래스별 자동 승인 임계값이 아직 검증되지 않았습니다. "
            "QA 분석 후 정밀도 95% 권장 임계값을 적용하세요.")
    if not dry and targets:
        result["train"] = train.maybe_start_training(pid)
    return result


@app.post("/api/images/bulk-status")
def bulk_status(body: dict):
    """선택한 이미지들의 상태를 한 번에 변경."""
    ids = body.get("image_ids")
    status = _valid_status(body.get("status"))
    if not isinstance(ids, list) or not ids or not all(isinstance(i, int) for i in ids):
        raise HTTPException(400, "image_ids는 하나 이상의 정수 배열이어야 합니다")
    ids = list(dict.fromkeys(ids))
    conn = get_db()
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, project_id FROM images WHERE id IN ({marks})", ids).fetchall()
    if len(rows) != len(ids) or len({r["project_id"] for r in rows}) != 1:
        conn.close()
        raise HTTPException(400, "이미지는 모두 존재하며 같은 프로젝트 소속이어야 합니다")
    conn.executemany("UPDATE images SET status=? WHERE id=?",
                     [(status, i) for i in ids])
    conn.commit()
    pid = rows[0]["project_id"]
    conn.close()
    trained = None
    if status == "approved":
        trained = train.maybe_start_training(pid, debounce=True)
    return {"ok": True, "count": len(ids), "train": trained}


# ---------- 삭제 ----------

@app.delete("/api/images/{iid}")
def delete_image(iid: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404)
    linked = bool(row["src_path"]) if "src_path" in row.keys() else False
    conn.execute("DELETE FROM foundation_candidates WHERE image_id=?", (iid,))
    conn.execute("DELETE FROM foundation_audits WHERE image_id=?", (iid,))
    conn.execute("DELETE FROM annotations WHERE image_id=?", (iid,))
    conn.execute("DELETE FROM images WHERE id=?", (iid,))
    conn.commit()
    conn.close()
    # 연결 임포트 이미지의 실제 파일은 사용자의 원본 데이터셋이다 — 프로젝트에서만
    # 빼고 디스크는 절대 건드리지 않는다. 지우는 건 우리가 만든 업로드 복사본뿐.
    if not linked:
        (DATA_DIR / str(row["project_id"]) / f"{row['id']}_{row['file_name']}").unlink(
            missing_ok=True)
    return {"ok": True}


@app.delete("/api/projects/{pid}")
def delete_project(pid: int):
    import shutil

    conn = get_db()
    conn.execute("DELETE FROM foundation_candidates WHERE project_id=?", (pid,))
    conn.execute("DELETE FROM foundation_audits WHERE project_id=?", (pid,))
    conn.execute(
        "DELETE FROM annotations WHERE image_id IN (SELECT id FROM images WHERE project_id=?)",
        (pid,))
    conn.execute("DELETE FROM images WHERE project_id=?", (pid,))
    conn.execute("DELETE FROM models WHERE project_id=?", (pid,))
    conn.execute("DELETE FROM projects WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    shutil.rmtree(DATA_DIR / str(pid), ignore_errors=True)
    return {"ok": True}


# ---------- 모델 다운로드 ----------

@app.get("/api/projects/{pid}/model")
def download_model(pid: int):
    m = train.active_model(pid)
    if not m:
        raise HTTPException(404, "활성 전용 모델 없음")
    # 강제 적용한 외부 .pt는 성능표가 없을 수 있다. None을 소수점 형식으로
    # 만들다 다운로드가 500이 되지 않도록 상태를 정직하게 파일명에 남긴다.
    score = m.get("map50")
    metric = f"map{score:.3f}" if isinstance(score, (int, float)) else "unverified"
    return FileResponse(m["path"], filename=f"model_p{pid}_{metric}.pt")


# ---------- zip 익스포트 (바로 학습 가능한 구조) ----------

def _exportable_images(conn, pid: int, include_rejected: bool):
    """익스포트 대상 이미지.

    거부(rejected)는 "이 데이터는 쓰지 말라"는 표시다. 그런데 익스포트가
    상태를 보지 않아 거부한 이미지가 학습셋에 그대로 실려 나갔다.
    되살릴 필요가 있으면 include_rejected=1로 명시해야 한다.
    """
    if include_rejected:
        return conn.execute("SELECT * FROM images WHERE project_id=?", (pid,)).fetchall()
    return conn.execute(
        "SELECT * FROM images WHERE project_id=? AND status!='rejected'", (pid,)).fetchall()


@app.get("/api/projects/{pid}/export.zip")
def export_zip(pid: int, fmt: str = "yolo", include_rejected: bool = False):
    import os as _os
    import tempfile
    import zipfile

    from starlette.background import BackgroundTask

    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    ontology = json.loads(proj["ontology"])
    names = [c["name"] for c in ontology]
    cls_id = {n: i for i, n in enumerate(names)}
    images = _exportable_images(conn, pid, include_rejected)

    # zip을 RAM(BytesIO)에 통째로 만들면 메모리가 데이터셋 크기만큼 치솟는다 —
    # 이미지 항목은 DEFLATE로 거의 안 줄어들어 수만 장이면 수십 GB다 (OOM).
    # 디스크 임시파일에 쓰고 전송이 끝나면 지운다. 이미지는 이미 압축본이라
    # STORED로 넣어 CPU도 아낀다.
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    missing = 0  # 원본이 사라진 이미지 — 헤더로 알린다 (조용한 빈 zip 방지)
    skipped = 0  # 온톨로지에 없는 클래스의 라벨 — 오라벨 대신 제외하고 알린다
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        if fmt == "yolo":
            z.writestr("data.yaml",
                       f"path: .\ntrain: images\nval: images\nnames: {json.dumps(names)}\n")
            for im in images:
                fname = f"{im['id']}_{im['file_name']}"
                src = _row_image_path(im)
                if not src.exists():
                    missing += 1
                    continue
                z.write(src, f"images/{fname}", compress_type=zipfile.ZIP_STORED)
                lines = []
                for a in conn.execute(
                        "SELECT * FROM annotations WHERE image_id=?", (im["id"],)):
                    d = row_to_dict(a)
                    if d["class_name"] not in cls_id:
                        skipped += 1
                        continue
                    x, y, w, h = d["bbox"]
                    lines.append(
                        f"{cls_id[d['class_name']]} "
                        f"{(x + w / 2) / im['width']:.6f} {(y + h / 2) / im['height']:.6f} "
                        f"{w / im['width']:.6f} {h / im['height']:.6f}")
                z.writestr(f"labels/{Path(fname).stem}.txt", "\n".join(lines))
        else:  # coco
            coco = export(pid, fmt="coco", include_rejected=include_rejected)
            skipped = coco.pop("skipped_unknown_class", 0)  # 표준 COCO 키가 아니라 뺀다
            # 여러 폴더를 한 프로젝트로 임포트하면 동명 파일이 생긴다. zip 안에서
            # 덮어쓰이지 않게 id를 붙이고, json의 file_name도 같이 맞춘다.
            for entry in coco.get("images", []):
                entry["file_name"] = f"{entry['id']}_{entry['file_name']}"
            z.writestr("annotations.json", json.dumps(coco, ensure_ascii=False))
            for im in images:
                src = _row_image_path(im)
                if src.exists():
                    z.write(src, f"images/{im['id']}_{im['file_name']}",
                            compress_type=zipfile.ZIP_STORED)
                else:
                    missing += 1
    conn.close()
    tmp.close()
    return FileResponse(
        tmp.name, media_type="application/zip",
        headers={"Content-Disposition": _attachment(f"{proj['name']}_{fmt}.zip"),
                 "X-Images-Exported": str(len(images) - missing),
                 "X-Images-Missing": str(missing),
                 "X-Annotations-Skipped": str(skipped)},
        background=BackgroundTask(_os.unlink, tmp.name))


# ---------- 통계적 배치 검수 ----------

def _acceptance_token(pid: int, status: str, ids: list[int], target: float,
                      confidence: float, max_defects) -> str:
    raw = json.dumps([pid, status, ids, target, confidence, max_defects], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:20]

@app.post("/api/projects/{pid}/acceptance-plan")
def acceptance_plan(pid: int, body: dict | None = None):
    """리뷰 대기 배치를 몇 장 검사하면 되는지 계산하고 표본을 뽑아준다."""
    from server import sampling

    b = body or {}
    status = b.get("status", "prelabeled")
    _valid_status(status)
    target, confidence, max_defects = _acceptance_params(b)
    conn = get_db()
    if not conn.execute("SELECT id FROM projects WHERE id=?", (pid,)).fetchone():
        conn.close()
        raise HTTPException(404, "프로젝트 없음")
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM images WHERE project_id=? AND status=? ORDER BY id", (pid, status))]
    conn.close()
    p = sampling.plan(len(ids), target, confidence, max_defects)
    p["sample_image_ids"] = sampling.pick_sample(ids, p["sample_size"])
    p["status"] = status
    p["lot_token"] = _acceptance_token(
        pid, status, ids, target, confidence, p["max_defects"])
    return p


@app.post("/api/projects/{pid}/acceptance-result")
def acceptance_result(pid: int, body: dict):
    """검사 결과로 배치 승인/반려를 판정하고, 승인 시 일괄 승인까지 수행."""
    from server import sampling

    status = _valid_status(body.get("status", "prelabeled"))
    target, confidence, parsed_max = _acceptance_params(body)
    if parsed_max is None:
        raise HTTPException(400, "max_defects가 필요합니다")
    max_defects = parsed_max
    conn = get_db()
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM images WHERE project_id=? AND status=? ORDER BY id", (pid, status))]
    token = _acceptance_token(pid, status, ids, target, confidence, max_defects)
    if not body.get("lot_token") or body["lot_token"] != token:
        conn.close()
        raise HTTPException(409, "검수 계획 이후 배치가 변경됐습니다 — 새 계획을 만드세요")
    plan = sampling.plan(len(ids), target, confidence, max_defects)
    sample_size = int(body["sample_size"])
    defects = int(body["defects"])
    if sample_size != plan["sample_size"] or not 0 <= defects <= sample_size:
        conn.close()
        raise HTTPException(400, "표본 크기 또는 불량 수가 검수 계획과 맞지 않습니다")
    v = sampling.verdict(sample_size, defects, plan["max_defects"], target, confidence)
    if v["accepted"] and body.get("apply", True):
        # 검사한 로트의 정확한 id만 승인한다. 해시 확인 직후 새 항목이 들어와도
        # 그 항목은 다음 검수 대상으로 남아야 한다.
        conn.executemany("UPDATE images SET status='approved' WHERE id=?", [(i,) for i in ids])
        n = len(ids)
        conn.commit()
        v["approved_images"] = n
        v["train"] = train.maybe_start_training(pid, debounce=True)
    conn.close()
    return v


# ---------- 클라우드 학습 레인 ----------

@app.get("/api/projects/{pid}/training-dataset.zip")
def training_dataset_zip(pid: int):
    """Colab/외부 학습 전용 데이터셋.

    일반 익스포트는 전달·백업 용도라 리뷰 전 이미지도 포함할 수 있다. 학습
    레인에서는 승인 완료 이미지에 한정하고, 로컬 학습과 동일한 고정
    train/val/test 분할을 써야 사람 검수가 끝나지 않은 초안과 평가 누출이
    모델에 들어가지 않는다.
    """
    import shutil
    import tempfile
    import zipfile

    from starlette.background import BackgroundTask

    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if not proj:
        conn.close()
        raise HTTPException(404, "프로젝트 없음")
    approved = conn.execute(
        "SELECT COUNT(*) c FROM images WHERE project_id=? AND status='approved'",
        (pid,)).fetchone()["c"]
    ontology = json.loads(proj["ontology"])
    conn.close()
    if not ontology:
        raise HTTPException(400, "학습 전에 클래스를 하나 이상 정의하세요")
    if not approved:
        raise HTTPException(400, "승인된 학습 이미지가 없습니다")

    temp_root = Path(tempfile.mkdtemp(prefix=f"autolabel-p{pid}-train-"))
    dataset = temp_root / "dataset"
    archive = temp_root / "approved-training.zip"
    try:
        train._export_yolo_dataset(pid, dataset)
        # 워커에서는 절대경로가 편하지만 다운로드 패키지는 다른 머신에서
        # 풀리므로 이동 가능한 상대경로여야 한다. 분할 경로는 그대로 보존한다.
        yaml_path = dataset / "data.yaml"
        yaml_text = yaml_path.read_text()
        yaml_path.write_text("path: .\n" + "\n".join(yaml_text.splitlines()[1:]) + "\n")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(dataset.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(dataset))
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    split_counts = {
        split: len(list((dataset / "images" / split).glob("*")))
        for split in ("train", "val", "test")
    }
    return FileResponse(
        archive, media_type="application/zip",
        headers={
            "Content-Disposition": _attachment(f"{proj['name']}_approved_training.zip"),
            "X-Approved-Images": str(sum(split_counts.values())),
            "X-Train-Images": str(split_counts["train"]),
            "X-Val-Images": str(split_counts["val"]),
            "X-Test-Images": str(split_counts["test"]),
        },
        background=BackgroundTask(shutil.rmtree, temp_root, True))

@app.get("/api/projects/{pid}/colab-notebook")
def colab_notebook(pid: int, arch: str | None = None, epochs: int | None = None):
    """대규모 학습용 Colab 노트북 생성 — 로컬 MPS로 감당 안 될 때의 탈출구.

    노트북은 (1) 도구에서 받은 zip 업로드 → (2) GPU 학습 → (3) best.pt 다운로드
    → (4) 도구에 모델 임포트 안내까지 담는다.
    """
    from fastapi.responses import Response

    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if not proj:
        conn.close()
        raise HTTPException(404, "프로젝트 없음")
    n_approved = conn.execute(
        "SELECT COUNT(*) c FROM images WHERE project_id=? AND status='approved'",
        (pid,)).fetchone()["c"]
    conn.close()
    names = [c["name"] for c in json.loads(proj["ontology"])]
    from server.train_worker import pick_arch

    allowed_arch = {"yolo11n", "yolo11s", "yolo11m"}
    if arch is not None and arch not in allowed_arch:
        raise HTTPException(400, f"arch는 {', '.join(sorted(allowed_arch))} 중 하나여야 합니다")
    readiness = train.training_readiness(pid)
    selected_epochs = epochs if epochs is not None else readiness["expected_epochs"]
    if not 1 <= selected_epochs <= 500:
        raise HTTPException(400, "epochs는 1~500 사이여야 합니다")
    selected_arch = pick_arch(n_approved, arch)
    split_counts = readiness["split_counts"]
    readiness_note = ("전문 평가 가능" if readiness["professional_ready"]
                      else "실험 모델 — 독립 홀드아웃이 충분해질 때까지 자동 적용하지 않음")

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            f"# {proj['name']} — 클라우드 학습\n\n",
            f"승인 라벨 **{n_approved}장** · 클래스 {', '.join(names)}\n\n",
            f"예상 분할 **{split_counts['train']}/{split_counts['val']}/{split_counts['test']}** "
            f"(train/val/test) · **{readiness_note}**\n\n",
            "1. 런타임 → 런타임 유형 변경 → **T4 GPU**\n",
            "2. 아래 셀 순서대로 실행 (2번째 셀에서 도구의 `승인 학습 데이터.zip` 업로드)\n",
            "3. 마지막 셀에서 `autolabel-model.zip` 다운로드 → 도구의 **Colab 결과 가져오기**에 경로 입력\n\n",
            "이 패키지는 승인 완료 데이터만 포함하며 train/val/test를 분리합니다.\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": ["!nvidia-smi -L\n", "%pip install -q ultralytics"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": ["from google.colab import files\n",
                    "up = files.upload()   # 도구에서 받은 승인 학습 데이터.zip 선택\n",
                    "import zipfile, glob\n",
                    "zipfile.ZipFile(list(up)[0]).extractall('ds')\n",
                    "print(glob.glob('ds/*'))"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
            "# data.yaml의 경로를 Colab 기준으로 교정\n",
            "import yaml, pathlib\n",
            "p = pathlib.Path('ds/data.yaml')\n",
            "d = yaml.safe_load(p.read_text())\n",
            "d['path'] = str(pathlib.Path('ds').resolve())\n",
            "p.write_text(yaml.safe_dump(d))\n",
            "print(d)"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
            "from ultralytics import YOLO\n",
            f"model = YOLO('{selected_arch}.pt')\n",
            f"model.train(data='ds/data.yaml', epochs={selected_epochs}, imgsz=640, batch=16,\n",
            "            patience=20, device=0, project='out', name='train', workers=2)"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
            "import json, shutil\n",
            "best_path = pathlib.Path(model.trainer.best)  # Ultralytics 버전별 저장 루트에 안전\n",
            "m = YOLO(str(best_path))\n",
            "val_res = m.val(data='ds/data.yaml', split='val', device=0, workers=2)\n",
            "test_images = list(pathlib.Path('ds/images/test').glob('*'))\n",
            "test_res = m.val(data='ds/data.yaml', split='test', device=0, workers=2) if test_images else None\n",
            "metrics = {'val_map50': float(val_res.box.map50), "
            "'val_map50_95': float(val_res.box.map), "
            "'test_map50': float(test_res.box.map50) if test_res else None, "
            "'test_map50_95': float(test_res.box.map) if test_res else None}\n",
            "class_metrics = None\n",
            "if test_res:\n",
            "    class_metrics = {}\n",
            "    for idx, map50, map50_95, instances in zip(test_res.box.ap_class_index, "
            "test_res.box.ap50, test_res.box.ap, test_res.nt_per_class):\n",
            "        class_metrics[m.names[int(idx)]] = {'test_map50': float(map50), "
            "'test_map50_95': float(map50_95), 'test_instances': int(instances)}\n",
            f"manifest = {{'schema_version': {MODEL_BUNDLE_SCHEMA}, "
            f"'classes': {json.dumps(names, ensure_ascii=False)}, "
            f"'architecture': '{selected_arch}', 'approved_images': {n_approved}, "
            f"'split_counts': {json.dumps(split_counts)}, 'epochs_requested': {selected_epochs}, "
            "'metrics': metrics, 'class_metrics': class_metrics}\n",
            "bundle = pathlib.Path('autolabel-model')\n",
            "if bundle.exists(): shutil.rmtree(bundle)\n",
            "bundle.mkdir()\n",
            "shutil.copy2(best_path, bundle / 'best.pt')\n",
            "(bundle / 'autolabel-model.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))\n",
            "shutil.make_archive('autolabel-model', 'zip', root_dir=bundle)\n",
            "print('AUTOLABEL_RESULT', json.dumps(manifest, ensure_ascii=False))\n",
            "print('BUNDLE_READY', pathlib.Path('autolabel-model.zip').resolve())"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
            "from google.colab import files\n",
            "files.download('autolabel-model.zip')"]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## 도구로 되돌리기\n\n",
            "받은 `autolabel-model.zip` 경로를 오토라벨 도구의 **Colab 결과 가져오기**에 입력하세요. ",
            "도구가 클래스·분할·성능을 검증해 먼저 후보로 등록하고, 통과한 모델만 별도 적용합니다.\n"]},
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU",
          "colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    return Response(
        content=json.dumps(nb, ensure_ascii=False, indent=1),
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition":
                 _attachment(f"{proj['name']}_colab_train.ipynb")})


# ---------- QA ----------

@app.post("/api/projects/{pid}/qa")
def run_qa(pid: int, background: bool = False):
    from server import qa

    # 대규모(수천 장+)는 백그라운드 심판 잡으로
    if background:
        return qa.start_judge(pid)
    return qa.analyze(pid)


@app.get("/api/projects/{pid}/qa/status")
def qa_status(pid: int):
    from server import qa

    return qa.job_status(pid)  # jobs가 메모리→디스크 폴백을 이미 담당


@app.get("/api/images/{iid}/suggestions")
def suggestions(iid: int, min_conf: float = 0.4):
    """활성 모델 예측 중 기존 라벨과 겹치지 않는 것 = 누락 의심 제안."""
    from server import qa
    from server.qa import _match

    conn = get_db()
    im = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
    if not im:
        conn.close()
        raise HTTPException(404, "이미지 없음")
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (im["project_id"],)).fetchone()
    labels = [row_to_dict(a) for a in conn.execute(
        "SELECT * FROM annotations WHERE image_id=?", (iid,))]
    conn.close()

    student = train.active_model(im["project_id"])
    if not student:
        raise HTTPException(400, "활성 전용 모델 없음")
    ontology = [{**c, "threshold": min_conf} for c in json.loads(proj["ontology"])]
    preds = ml.detect_student(Image.open(_image_path(iid)).convert("RGB"), student, ontology)
    _matched, spurious, missing = _match(preds, labels)
    # 겹치는 건 새 객체가 아니라 같은 객체를 다르게 잡은 것 — 반영하면 중복 라벨이 된다
    new_objects = qa.filter_new_objects(spurious, labels)
    return {
        # 모델은 찾았는데 라벨에 없음 → 추가 제안
        "missing_labels": new_objects,
        # 기존 라벨과 겹쳐서 제외된 수 (박스가 어긋난 것이지 누락은 아님)
        "overlapping_skipped": len(spurious) - len(new_objects),
        # 라벨에 있는데 모델이 못 찾음 → 오라벨 의심(참고용)
        "model_missed": [{"class_name": a["class_name"], "bbox": a["bbox"]} for a in missing],
    }


@app.post("/api/images/{iid}/apply-suggestions")
def apply_suggestions(iid: int, body: dict):
    """제안된 박스들을 실제 라벨로 추가 (라벨 세탁 원클릭)."""
    conn = get_db()
    n = 0
    for s in body["boxes"]:
        conn.execute(
            "INSERT INTO annotations (image_id, class_name, bbox, confidence, source, meta) "
            "VALUES (?,?,?,?,?,?)",
            (iid, s["class_name"], json.dumps(s["bbox"]), s.get("confidence"),
             "model", json.dumps({"applied_from": "suggestion"})))
        n += 1
    conn.commit()
    conn.close()
    return {"ok": True, "added": n}


# ---------- 능동 샘플 선별 ----------

@app.get("/api/projects/{pid}/next-to-label")
def next_to_label(pid: int, n: int = 20):
    """다음에 라벨할 가치가 높은 이미지 추천.

    점수 = 불확실성(임계값 근처 예측 비율) + 검출 희소성 + 미라벨 우선.
    모델이 헷갈리는 이미지를 먼저 사람에게 보내 라벨 예산 효율을 높인다.
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT i.*, "
        "(SELECT COUNT(*) FROM annotations a WHERE a.image_id=i.id) AS ann_count, "
        "(SELECT AVG(a.confidence) FROM annotations a WHERE a.image_id=i.id) AS avg_conf, "
        "(SELECT MIN(a.confidence) FROM annotations a WHERE a.image_id=i.id) AS min_conf "
        "FROM images i WHERE i.project_id=? AND i.status IN ('unlabeled','prelabeled')",
        (pid,)).fetchall()
    conn.close()

    scored = []
    for r in rows:
        d = dict(r)
        score = 0.0
        if d["ann_count"] == 0:
            score += 1.0  # 라벨 없는 이미지는 정보량 미지 — 우선
        if d["min_conf"] is not None:
            # 0.5 근처가 가장 불확실 (모델이 갈팡질팡)
            score += 1.5 * (1 - abs(d["min_conf"] - 0.5) * 2)
        if d["avg_conf"] is not None and d["avg_conf"] < 0.6:
            score += 0.5
        if d["qa_score"]:
            score += min(d["qa_score"] / 5, 1.0)  # 심판 의심도 반영
        scored.append({"image_id": d["id"], "file_name": d["file_name"],
                       "score": round(score, 3), "ann_count": d["ann_count"],
                       "min_conf": d["min_conf"], "status": d["status"]})
    scored.sort(key=lambda s: -s["score"])
    return {"total_candidates": len(scored), "recommended": scored[:n]}


# ---------- 모델 히스토리 ----------

@app.get("/api/projects/{pid}/models")
def list_models(pid: int):
    conn = get_db()
    rows = [row_to_dict(r) for r in conn.execute(
        "SELECT * FROM models WHERE project_id=? ORDER BY id DESC", (pid,))]
    conn.close()
    return rows


@app.post("/api/projects/{pid}/models/{mid}/activate")
def activate_model(pid: int, mid: int, force: bool = False):
    """이전 모델로 롤백 (성능 회귀 시)."""
    conn = get_db()
    # 대상 확인이 먼저다. 없는 id로 들어오면 전부 비활성화만 되고 아무것도
    # 활성화되지 않아, ok를 받고도 전용 모델이 사라진다 (오토라벨이 조용히 강등).
    target = conn.execute(
        "SELECT * FROM models WHERE id=? AND project_id=?", (mid, pid)).fetchone()
    if not target:
        conn.close()
        raise HTTPException(404, f"모델 {mid}이 프로젝트 {pid}에 없음")
    model = row_to_dict(target)
    quality = _quality_from_model(model)
    if quality in {"failed", "unverified"} and not force:
        conn.close()
        reason = (model.get("meta") or {}).get("quality_reason") or "성능표가 없습니다"
        raise HTTPException(409, f"모델 품질 검증을 통과하지 못해 적용할 수 없습니다: {reason}")
    conn.execute("UPDATE models SET active=0 WHERE project_id=?", (pid,))
    conn.execute("UPDATE models SET active=1 WHERE id=?", (mid,))
    conn.commit()
    conn.close()
    return {"ok": True, "active_model_id": mid}


# ---------- 파인튜닝 ----------

@app.post("/api/projects/{pid}/train")
def trigger_train(pid: int):
    return train.maybe_start_training(pid, force=True)


@app.get("/api/projects/{pid}/train/status")
def train_status(pid: int):
    return {"job": train.job_status(pid), "active_model": train.active_model(pid)}


@app.get("/api/projects/{pid}/train/readiness")
def train_readiness(pid: int):
    result = train.training_readiness(pid)
    if result is None:
        raise HTTPException(404, f"프로젝트 {pid} 없음")
    return result


# ---------- 익스포트 ----------

@app.get("/api/projects/{pid}/export")
def export(pid: int, fmt: str = "coco", include_rejected: bool = False):
    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    ontology = json.loads(proj["ontology"])
    cat_id = {c["name"]: i + 1 for i, c in enumerate(ontology)}
    images = _exportable_images(conn, pid, include_rejected)

    # 온톨로지에 없는 클래스(이름 변경·삭제 뒤 남은 옛 라벨)는 건너뛴다.
    # 예전엔 yolo가 class 0으로, coco가 category_id 0으로 조용히 오라벨해
    # 이 파일로 학습한 외부 모델이 엉뚱한 클래스를 배웠다.
    if fmt == "coco":
        out = {
            "images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": c["name"]} for i, c in enumerate(ontology)],
            "skipped_unknown_class": 0,
        }
        aid = 0
        for im in images:
            out["images"].append({
                "id": im["id"], "file_name": im["file_name"],
                "width": im["width"], "height": im["height"]})
            for a in conn.execute(
                    "SELECT * FROM annotations WHERE image_id=?", (im["id"],)):
                d = row_to_dict(a)
                if d["class_name"] not in cat_id:
                    out["skipped_unknown_class"] += 1
                    continue
                aid += 1
                out["annotations"].append({
                    "id": aid, "image_id": im["id"],
                    "category_id": cat_id[d["class_name"]],
                    "bbox": d["bbox"], "area": d["bbox"][2] * d["bbox"][3],
                    "segmentation": d.get("segmentation"), "iscrowd": 0,
                    "score": d.get("confidence"), "source": d["source"]})
        conn.close()
        return out

    if fmt == "yolo":
        files = {}
        for im in images:
            lines = []
            for a in conn.execute(
                    "SELECT * FROM annotations WHERE image_id=?", (im["id"],)):
                d = row_to_dict(a)
                if d["class_name"] not in cat_id:
                    continue
                x, y, w, h = d["bbox"]
                lines.append(
                    f"{cat_id[d['class_name']] - 1} "
                    f"{(x + w / 2) / im['width']:.6f} {(y + h / 2) / im['height']:.6f} "
                    f"{w / im['width']:.6f} {h / im['height']:.6f}")
            files[Path(im["file_name"]).stem + ".txt"] = "\n".join(lines)
        conn.close()
        return files

    conn.close()
    raise HTTPException(400, "fmt는 coco 또는 yolo")


# ---------- 프론트 정적 서빙 (빌드 후) ----------

WEBAPP_DIST = Path(__file__).parent.parent / "webapp" / "dist"
if WEBAPP_DIST.exists():
    app.mount("/", StaticFiles(directory=WEBAPP_DIST, html=True), name="webapp")

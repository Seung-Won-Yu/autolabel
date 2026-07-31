"""데이터셋 연결 임포트 — 복사 없이 원본 폴더를 참조하고 기존 라벨을 읽어온다.

대용량 공공 데이터셋(수만~수십만 장) 대응:
- 이미지는 src_path로 원본 경로만 기록 (디스크 사용 0)
- 라벨은 YOLO txt / COCO json 자동 판별 후 DB로 임포트
- 서브셋 샘플링: 특정 클래스 포함 이미지 우선 등
"""
import json
import random
import threading
from pathlib import Path

from PIL import Image

from server import jobs
from server.db import get_db

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_jobs: dict[int, dict] = {}


def job_status(pid: int) -> dict:
    return _jobs.get(pid, {"status": "idle"})


def preview(images_dir: str, labels_dir: str | None = None,
            coco_json: str | None = None) -> dict:
    """임포트 전 미리보기 — 이미지 수, 라벨 형식, 클래스 목록 추정."""
    idir = Path(images_dir).expanduser()
    if not idir.is_dir():
        return {"error": f"이미지 폴더 없음: {idir}"}
    imgs = [p for p in idir.rglob("*") if p.suffix.lower() in IMG_EXT]
    out = {"images": len(imgs), "images_dir": str(idir), "sample": [p.name for p in imgs[:3]]}

    if coco_json:
        cpath = Path(coco_json).expanduser()
        if not cpath.exists():
            return {**out, "error": f"COCO json 없음: {cpath}"}
        d = json.loads(cpath.read_text())
        out.update(format="coco", classes=[c["name"] for c in d.get("categories", [])],
                   annotations=len(d.get("annotations", [])),
                   labeled_images=len(d.get("images", [])))
        return out

    ldir = Path(labels_dir).expanduser() if labels_dir else idir.parent / "labels"
    if ldir.is_dir():
        txts = list(ldir.rglob("*.txt"))
        ids = set()
        n_box = 0
        for t in txts[:2000]:  # 표본으로 클래스 id 수집
            for line in t.read_text().splitlines():
                parts = line.split()
                if parts:
                    ids.add(int(parts[0]))
                    n_box += 1
        out.update(format="yolo", labels_dir=str(ldir), label_files=len(txts),
                   class_ids=sorted(ids), sampled_boxes=n_box)
    else:
        out.update(format="none", note="라벨 폴더를 찾지 못함 — 이미지만 임포트됩니다")
    return out


def _run_import(pid: int, images_dir: str, labels_dir: str | None,
                coco_json: str | None, class_names: list[str],
                limit: int | None, require_class: str | None, seed: int):
    job = _jobs[pid]
    conn = get_db()
    try:
        idir = Path(images_dir).expanduser()
        by_name = {}
        for p in idir.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                by_name.setdefault(p.name, p)
                by_name.setdefault(p.stem, p)

        # 라벨 로드: {이미지 stem 또는 파일명: [(class_name, [x,y,w,h]), ...]}
        labels: dict[str, list] = {}
        sizes: dict[str, tuple] = {}
        if coco_json:
            d = json.loads(Path(coco_json).expanduser().read_text())
            cats = {c["id"]: c["name"] for c in d.get("categories", [])}
            imgs = {im["id"]: im for im in d.get("images", [])}
            for a in d.get("annotations", []):
                im = imgs.get(a["image_id"])
                if not im:
                    continue
                key = Path(im["file_name"]).name
                labels.setdefault(key, []).append(
                    (cats.get(a["category_id"], "unknown"), [float(v) for v in a["bbox"]]))
                sizes[key] = (im.get("width"), im.get("height"))
        else:
            ldir = Path(labels_dir).expanduser() if labels_dir else idir.parent / "labels"
            for t in ldir.rglob("*.txt"):
                rows = []
                for line in t.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    name = class_names[cid] if cid < len(class_names) else f"class{cid}"
                    rows.append((name, [cx, cy, bw, bh]))  # 정규화 좌표 — 나중에 픽셀 변환
                labels[t.stem] = rows

        # 대상 이미지 선정 (서브셋 샘플링)
        candidates = []
        for key, rows in labels.items():
            src = by_name.get(key) or by_name.get(Path(key).stem)
            if not src:
                continue
            if require_class and not any(r[0] == require_class for r in rows):
                continue
            candidates.append((key, src, rows))
        if not labels:  # 라벨 없이 이미지만
            # by_name은 파일명과 stem 두 키가 같은 파일을 가리킨다 (라벨 매칭용).
            # 그대로 values()를 돌면 모든 이미지가 두 번 임포트된다 (실측).
            uniq = dict.fromkeys(by_name.values())
            candidates = [(p.name, p, []) for p in uniq if p.is_file()]

        random.Random(seed).shuffle(candidates)
        if limit:
            candidates = candidates[:limit]
        job.update(total=len(candidates), done=0)

        is_coco = bool(coco_json)
        for n, (key, src, rows) in enumerate(candidates, 1):
            w, h = sizes.get(key, (None, None))
            if not w or not h:
                try:
                    w, h = Image.open(src).size
                except Exception:
                    continue
            cur = conn.execute(
                "INSERT INTO images (project_id, file_name, width, height, status, src_path) "
                "VALUES (?,?,?,?,?,?)",
                (pid, src.name, w, h, "prelabeled" if rows else "unlabeled", str(src)))
            iid = cur.lastrowid
            for name, box in rows:
                if is_coco:
                    bbox = [round(v, 1) for v in box]  # COCO는 이미 픽셀 xywh
                else:
                    cx, cy, bw, bh = box
                    bbox = [round((cx - bw / 2) * w, 1), round((cy - bh / 2) * h, 1),
                            round(bw * w, 1), round(bh * h, 1)]
                conn.execute(
                    "INSERT INTO annotations (image_id, class_name, bbox, source, meta) "
                    "VALUES (?,?,?,?,?)",
                    (iid, name, json.dumps(bbox), "import",
                     json.dumps({"origin": coco_json or labels_dir or "yolo"})))
            if n % 200 == 0:
                conn.commit()
                job.update(done=n)
                jobs.update("import", pid, done=n, total=len(candidates))
        conn.commit()
        job.update(status="completed", done=len(candidates))
        jobs.update("import", pid, status="completed", done=len(candidates))
    except Exception as e:
        job.update(status="failed", error=str(e))
        jobs.update("import", pid, status="failed", error=str(e))
    finally:
        conn.close()


def start_import(pid: int, body: dict) -> dict:
    if _jobs.get(pid, {}).get("status") == "running":
        return _jobs[pid]
    _jobs[pid] = {"status": "running", "done": 0, "total": 0}
    # 서버 재시작 시 기록이 사라져 "완료"로 오독되지 않게 디스크에도 남긴다
    jobs.start("import", pid, done=0, total=0)
    threading.Thread(
        target=_run_import,
        args=(pid, body["images_dir"], body.get("labels_dir"), body.get("coco_json"),
              body.get("class_names", []), body.get("limit"),
              body.get("require_class"), int(body.get("seed", 42))),
        daemon=True).start()
    return _jobs[pid]

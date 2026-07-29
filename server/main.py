"""오토라벨 도구 백엔드 — FastAPI.

실행: .venv/bin/python -m uvicorn server.main:app --port 8899 --reload
"""
import io
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from server import ml, train
from server.db import get_db, init_db, row_to_dict

DATA_DIR = Path(__file__).parent.parent / "data" / "uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="autolabel")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
init_db()

# 배치 잡 상태 (MVP: 인메모리 단일 워커 — Phase 2에서 큐로 교체)
_jobs: dict[int, dict] = {}


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


@app.get("/api/projects")
def list_projects():
    conn = get_db()
    rows = [row_to_dict(r) for r in conn.execute(
        "SELECT p.*, (SELECT COUNT(*) FROM images i WHERE i.project_id=p.id) AS image_count "
        "FROM projects p ORDER BY p.id DESC")]
    conn.close()
    return rows


@app.put("/api/projects/{pid}/ontology")
def update_ontology(pid: int, body: dict):
    conn = get_db()
    conn.execute("UPDATE projects SET ontology=? WHERE id=?",
                 (json.dumps(body["ontology"]), pid))
    conn.commit()
    conn.close()
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
    saved = []
    pdir = DATA_DIR / str(pid)
    pdir.mkdir(exist_ok=True)
    for f in files:
        data = await f.read()
        try:
            im = Image.open(io.BytesIO(data))
            w, h = im.size
        except Exception:
            continue
        cur = conn.execute(
            "INSERT INTO images (project_id, file_name, width, height) VALUES (?,?,?,?)",
            (pid, f.filename, w, h))
        iid = cur.lastrowid
        # 파일명 충돌 회피 — id 프리픽스 저장
        (pdir / f"{iid}_{f.filename}").write_bytes(data)
        saved.append(iid)
    conn.commit()
    conn.close()
    return {"saved": saved}


@app.get("/api/projects/{pid}/images")
def list_images(pid: int, status: str | None = None):
    conn = get_db()
    q = ("SELECT i.*, (SELECT COUNT(*) FROM annotations a WHERE a.image_id=i.id) AS ann_count, "
         "(SELECT MIN(a.confidence) FROM annotations a WHERE a.image_id=i.id) AS min_conf "
         "FROM images i WHERE i.project_id=?")
    args: list = [pid]
    if status:
        q += " AND i.status=?"
        args.append(status)
    rows = [row_to_dict(r) for r in conn.execute(q + " ORDER BY i.id", args)]
    conn.close()
    return rows


def _image_path(iid: int) -> Path:
    conn = get_db()
    row = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404)
    return DATA_DIR / str(row["project_id"]) / f"{iid}_{row['file_name']}"


@app.get("/api/images/{iid}/file")
def image_file(iid: int):
    return FileResponse(_image_path(iid))


@app.put("/api/images/{iid}/status")
def set_image_status(iid: int, body: dict):
    conn = get_db()
    conn.execute("UPDATE images SET status=? WHERE id=?", (body["status"], iid))
    conn.commit()
    row = conn.execute("SELECT project_id FROM images WHERE id=?", (iid,)).fetchone()
    conn.close()
    # 승인 누적 시 자동 파인튜닝 트리거 (조건 미달이면 no-op)
    trained = None
    if body["status"] == "approved" and row:
        trained = train.maybe_start_training(row["project_id"])
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
    """이미지의 어노테이션 전체 교체 (캔버스 저장). 사람 수정은 source='human'."""
    conn = get_db()
    conn.execute("DELETE FROM annotations WHERE image_id=?", (iid,))
    for a in body["annotations"]:
        conn.execute(
            "INSERT INTO annotations (image_id, class_name, bbox, segmentation, confidence, "
            "parent_annotation_id, source, meta) VALUES (?,?,?,?,?,?,?,?)",
            (iid, a["class_name"], json.dumps(a["bbox"]),
             json.dumps(a.get("segmentation")) if a.get("segmentation") else None,
             a.get("confidence"), a.get("parent_annotation_id"),
             a.get("source", "human"), json.dumps(a.get("meta", {}))))
    conn.commit()
    conn.close()
    return {"ok": True}


# ---------- SAM 임베딩 (브라우저 디코더용) ----------

@app.get("/api/images/{iid}/embed")
def embed(iid: int):
    return ml.embed_image(_image_path(iid).read_bytes())


# ---------- 오토라벨 ----------

def _detect_auto(pid: int, image: Image.Image, ontology: list, engine: str = "auto"):
    """엔진 라우팅: 활성 학생 모델 우선, 없으면 파운데이션(GDINO). 반환: (검출, 사용 엔진)."""
    student = train.active_model(pid) if engine in ("auto", "student") else None
    if student:
        return ml.detect_student(image, student, ontology), f"student(mAP50 {student['map50']})"
    return ml.detect(image, ontology), "foundation"


@app.post("/api/images/{iid}/autolabel")
def autolabel_one(iid: int, body: dict | None = None):
    """단일 이미지 오토라벨 (프리뷰용). 결과는 저장하지 않고 반환만."""
    conn = get_db()
    img = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (img["project_id"],)).fetchone()
    conn.close()
    ontology = (body or {}).get("ontology") or json.loads(proj["ontology"])
    if not ontology:
        raise HTTPException(400, "온톨로지가 비어 있음")
    image = Image.open(_image_path(iid)).convert("RGB")
    dets, engine = _detect_auto(img["project_id"], image, ontology,
                                (body or {}).get("engine", "auto"))
    if (body or {}).get("masks", True):
        rles = ml.boxes_to_masks(image, [d["bbox"] for d in dets])
        for d, r in zip(dets, rles):
            d["segmentation"] = r
    return {"detections": dets, "engine": engine}


def _run_batch(pid: int, image_ids: list[int], ontology: list[dict], masks: bool):
    job = _jobs[pid]
    conn = get_db()
    try:
        for n, iid in enumerate(image_ids, 1):
            image = Image.open(_image_path(iid)).convert("RGB")
            dets, _engine = _detect_auto(pid, image, ontology)
            rles = ml.boxes_to_masks(image, [d["bbox"] for d in dets]) if masks else []
            conn.execute(
                "DELETE FROM annotations WHERE image_id=? AND source='model'", (iid,))
            for i, d in enumerate(dets):
                conn.execute(
                    "INSERT INTO annotations (image_id, class_name, bbox, segmentation, "
                    "confidence, source, meta) VALUES (?,?,?,?,?,?,?)",
                    (iid, d["class_name"], json.dumps(d["bbox"]),
                     json.dumps(rles[i]) if i < len(rles) else None,
                     d["confidence"], "model",
                     json.dumps({"model": ml.DINO_MODEL, "ontology": ontology})))
            conn.execute(
                "UPDATE images SET status='prelabeled' WHERE id=? AND status='unlabeled'",
                (iid,))
            conn.commit()
            job.update(done=n, total=len(image_ids))
        job["status"] = "completed"
    except Exception as e:  # 잡 실패를 상태로 노출
        job.update(status="failed", error=str(e))
    finally:
        conn.close()


@app.post("/api/projects/{pid}/autolabel")
def autolabel_batch(pid: int, body: dict):
    """배치 오토라벨 — 백그라운드 스레드 (MVP)."""
    if _jobs.get(pid, {}).get("status") == "running":
        raise HTTPException(409, "이미 실행 중인 잡 있음")
    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    ontology = body.get("ontology") or json.loads(proj["ontology"])
    ids = body.get("image_ids") or [
        r["id"] for r in conn.execute(
            "SELECT id FROM images WHERE project_id=?", (pid,))]
    conn.close()
    _jobs[pid] = {"status": "running", "done": 0, "total": len(ids)}
    threading.Thread(
        target=_run_batch, args=(pid, ids, ontology, body.get("masks", True)),
        daemon=True).start()
    return _jobs[pid]


@app.get("/api/projects/{pid}/autolabel/status")
def autolabel_status(pid: int):
    return _jobs.get(pid, {"status": "idle"})


# ---------- 시각 예시 검출 ----------

@app.post("/api/images/{iid}/exemplar")
def exemplar(iid: int, body: dict):
    """예시 박스 1개 → 같은 이미지에서 유사 객체 전부 검출."""
    import numpy as np

    image = np.array(Image.open(_image_path(iid)).convert("RGB"))
    dets = ml.exemplar_detect(
        image, body["bbox"],
        topk=int(body.get("topk", 20)),
        sim_thr=float(body.get("sim_thr", 0.6)))
    cls = body.get("class_name", "")
    for d in dets:
        d["class_name"] = cls
    return {"detections": dets}


# ---------- QA ----------

@app.post("/api/projects/{pid}/qa")
def run_qa(pid: int):
    from server import qa

    return qa.analyze(pid)


# ---------- 파인튜닝 ----------

@app.post("/api/projects/{pid}/train")
def trigger_train(pid: int):
    return train.maybe_start_training(pid, force=True)


@app.get("/api/projects/{pid}/train/status")
def train_status(pid: int):
    return {"job": train.job_status(pid), "active_model": train.active_model(pid)}


# ---------- 익스포트 ----------

@app.get("/api/projects/{pid}/export")
def export(pid: int, fmt: str = "coco"):
    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    ontology = json.loads(proj["ontology"])
    cat_id = {c["name"]: i + 1 for i, c in enumerate(ontology)}
    images = conn.execute("SELECT * FROM images WHERE project_id=?", (pid,)).fetchall()

    if fmt == "coco":
        out = {
            "images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": c["name"]} for i, c in enumerate(ontology)],
        }
        aid = 0
        for im in images:
            out["images"].append({
                "id": im["id"], "file_name": im["file_name"],
                "width": im["width"], "height": im["height"]})
            for a in conn.execute(
                    "SELECT * FROM annotations WHERE image_id=?", (im["id"],)):
                d = row_to_dict(a)
                aid += 1
                out["annotations"].append({
                    "id": aid, "image_id": im["id"],
                    "category_id": cat_id.get(d["class_name"], 0),
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
                x, y, w, h = d["bbox"]
                cid = cat_id.get(d["class_name"], 1) - 1
                lines.append(
                    f"{cid} {(x + w / 2) / im['width']:.6f} {(y + h / 2) / im['height']:.6f} "
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

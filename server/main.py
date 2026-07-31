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

from server import importer, ml, train
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
    # 연결 임포트된 이미지는 원본 경로 참조 (복사본 없음)
    src = row["src_path"] if "src_path" in row.keys() else None
    if src:
        return Path(src)
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
    """엔진 라우팅: 활성 학생 모델 우선, 없으면 파운데이션(GDINO).

    온톨로지에 '부모.자식' 표기가 있으면 part 캐스케이드까지 수행한다.
    반환: (검출, 사용 엔진)
    """
    from server import parts

    parent_onto, parts_by_parent = parts.parse_ontology(ontology)
    student = train.active_model(pid) if engine in ("auto", "student") else None
    if student:
        dets = ml.detect_student(image, student, ontology)
        used = f"student(mAP50 {student['map50']})"
        # 학생 모델이 part까지 학습했으면 캐스케이드 불필요
        if parts_by_parent and not any("." in d["class_name"] for d in dets):
            dets = dets + parts.detect_with_parts(image, ontology, dets)
            used += "+parts"
        return dets, used

    dets = ml.detect(image, parent_onto or ontology)
    used = "foundation"
    if parts_by_parent:
        dets = dets + parts.detect_with_parts(image, ontology, dets)
        used += "+parts"
    return dets, used


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
                     json.dumps({"model": ml.DINO_MODEL, "ontology": ontology})))
                if "_parent_index" not in d:
                    parent_ids[i] = cur.lastrowid
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


# ---------- 데이터셋 연결 임포트 ----------

@app.post("/api/import/preview")
def import_preview(body: dict):
    """폴더 경로만 주면 이미지 수·라벨 형식·클래스를 미리 알려준다."""
    return importer.preview(body["images_dir"], body.get("labels_dir"), body.get("coco_json"))


@app.post("/api/projects/{pid}/import")
def import_dataset(pid: int, body: dict):
    """복사 없이 폴더를 연결하고 기존 라벨을 가져온다 (대용량 데이터셋용)."""
    return importer.start_import(pid, body)


@app.get("/api/projects/{pid}/import/status")
def import_status(pid: int):
    return importer.job_status(pid)


# ---------- 외부 모델 임포트 ----------

@app.post("/api/projects/{pid}/models/import")
def import_model(pid: int, body: dict):
    """도구 밖에서 학습한 YOLO .pt를 전용 모델로 등록."""
    path = Path(body["path"]).expanduser()
    if not path.exists():
        raise HTTPException(400, f"모델 파일 없음: {path}")
    names = body.get("names")
    if not names:
        try:
            from ultralytics import YOLO

            m = YOLO(str(path))
            names = [m.names[i] for i in sorted(m.names)]
        except Exception as e:
            raise HTTPException(400, f"클래스명 자동 추출 실패 — names를 넘겨주세요 ({e})")
    conn = get_db()
    if body.get("activate", True):
        conn.execute("UPDATE models SET active=0 WHERE project_id=?", (pid,))
    conn.execute(
        "INSERT INTO models (project_id, path, map50, train_images, active, meta) "
        "VALUES (?,?,?,?,?,?)",
        (pid, str(path), body.get("map50"), body.get("train_images", 0),
         int(body.get("activate", True)),
         json.dumps({"arch": body.get("arch", "external"), "names": names,
                     "imported": True})))
    conn.commit()
    conn.close()
    return {"ok": True, "names": names}


# ---------- 자동 승인 (TBAL) / 일괄 작업 ----------

@app.post("/api/projects/{pid}/auto-approve")
def auto_approve(pid: int, body: dict):
    """모든 박스가 임계값 이상인 리뷰 대기 이미지를 자동 승인.

    dry_run=true면 대상만 세어 돌려준다 (승인 전 미리보기).
    thresholds: {클래스: 최소 confidence} — 없으면 min_conf 일괄 적용.
    """
    min_conf = float(body.get("min_conf", 0.7))
    thresholds = body.get("thresholds") or {}
    dry = bool(body.get("dry_run", False))
    require_labeled = bool(body.get("require_labeled", True))

    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM images WHERE project_id=? AND status='prelabeled'", (pid,)).fetchall()
    targets, skipped_lowconf, skipped_empty = [], 0, 0
    for im in rows:
        anns = [row_to_dict(a) for a in conn.execute(
            "SELECT * FROM annotations WHERE image_id=?", (im["id"],))]
        if not anns:
            skipped_empty += 1
            if require_labeled:
                continue
        ok = True
        for a in anns:
            need = float(thresholds.get(a["class_name"], min_conf))
            # 사람이 만든 라벨(confidence None)은 통과, 모델 라벨만 임계값 검사
            if a["confidence"] is not None and a["confidence"] < need:
                ok = False
                break
        if ok:
            targets.append(im["id"])
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
        "coverage": round(len(targets) / total, 3) if total else 0,
        "dry_run": dry,
    }
    if not dry and targets:
        result["train"] = train.maybe_start_training(pid)
    return result


@app.post("/api/images/bulk-status")
def bulk_status(body: dict):
    """선택한 이미지들의 상태를 한 번에 변경."""
    ids = body["image_ids"]
    conn = get_db()
    conn.executemany("UPDATE images SET status=? WHERE id=?",
                     [(body["status"], i) for i in ids])
    conn.commit()
    pid = conn.execute("SELECT project_id FROM images WHERE id=?", (ids[0],)).fetchone()
    conn.close()
    trained = None
    if body["status"] == "approved" and pid:
        trained = train.maybe_start_training(pid["project_id"])
    return {"ok": True, "count": len(ids), "train": trained}


# ---------- 삭제 ----------

@app.delete("/api/images/{iid}")
def delete_image(iid: int):
    path = _image_path(iid)
    conn = get_db()
    conn.execute("DELETE FROM annotations WHERE image_id=?", (iid,))
    conn.execute("DELETE FROM images WHERE id=?", (iid,))
    conn.commit()
    conn.close()
    path.unlink(missing_ok=True)
    return {"ok": True}


@app.delete("/api/projects/{pid}")
def delete_project(pid: int):
    import shutil

    conn = get_db()
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
    return FileResponse(m["path"], filename=f"model_p{pid}_map{m['map50']:.3f}.pt")


# ---------- zip 익스포트 (바로 학습 가능한 구조) ----------

@app.get("/api/projects/{pid}/export.zip")
def export_zip(pid: int, fmt: str = "yolo"):
    import io as _io
    import zipfile

    from fastapi.responses import StreamingResponse

    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    ontology = json.loads(proj["ontology"])
    names = [c["name"] for c in ontology]
    cls_id = {n: i for i, n in enumerate(names)}
    images = conn.execute("SELECT * FROM images WHERE project_id=?", (pid,)).fetchall()

    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if fmt == "yolo":
            z.writestr("data.yaml",
                       f"path: .\ntrain: images\nval: images\nnames: {json.dumps(names)}\n")
            for im in images:
                fname = f"{im['id']}_{im['file_name']}"
                src = DATA_DIR / str(pid) / fname
                if not src.exists():
                    continue
                z.write(src, f"images/{fname}")
                lines = []
                for a in conn.execute(
                        "SELECT * FROM annotations WHERE image_id=?", (im["id"],)):
                    d = row_to_dict(a)
                    if d["class_name"] not in cls_id:
                        continue
                    x, y, w, h = d["bbox"]
                    lines.append(
                        f"{cls_id[d['class_name']]} "
                        f"{(x + w / 2) / im['width']:.6f} {(y + h / 2) / im['height']:.6f} "
                        f"{w / im['width']:.6f} {h / im['height']:.6f}")
                z.writestr(f"labels/{Path(fname).stem}.txt", "\n".join(lines))
        else:  # coco
            coco = export(pid, fmt="coco")
            z.writestr("annotations.json", json.dumps(coco, ensure_ascii=False))
            for im in images:
                src = DATA_DIR / str(pid) / f"{im['id']}_{im['file_name']}"
                if src.exists():
                    z.write(src, f"images/{im['file_name']}")
    conn.close()
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition":
                 f"attachment; filename={proj['name']}_{fmt}.zip"})


# ---------- 클라우드 학습 레인 ----------

@app.get("/api/projects/{pid}/colab-notebook")
def colab_notebook(pid: int, arch: str = "yolo11m", epochs: int = 100):
    """대규모 학습용 Colab 노트북 생성 — 로컬 MPS로 감당 안 될 때의 탈출구.

    노트북은 (1) 도구에서 받은 zip 업로드 → (2) GPU 학습 → (3) best.pt 다운로드
    → (4) 도구에 모델 임포트 안내까지 담는다.
    """
    from fastapi.responses import Response

    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    n_approved = conn.execute(
        "SELECT COUNT(*) c FROM images WHERE project_id=? AND status='approved'",
        (pid,)).fetchone()["c"]
    conn.close()
    names = [c["name"] for c in json.loads(proj["ontology"])]

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            f"# {proj['name']} — 클라우드 학습\n\n",
            f"승인 라벨 **{n_approved}장** · 클래스 {', '.join(names)}\n\n",
            "1. 런타임 → 런타임 유형 변경 → **T4 GPU**\n",
            "2. 아래 셀 순서대로 실행 (2번째 셀에서 도구의 `YOLO.zip` 업로드)\n",
            "3. 마지막 셀에서 `best.pt` 다운로드 → 도구의 **외부 모델 등록**에 경로 입력\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": ["!nvidia-smi -L\n", "%pip install -q ultralytics"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": ["from google.colab import files\n",
                    "up = files.upload()   # 도구에서 받은 YOLO.zip 선택\n",
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
            "d.setdefault('train', 'images'); d.setdefault('val', 'images')\n",
            "p.write_text(yaml.safe_dump(d))\n",
            "print(d)"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
            "from ultralytics import YOLO\n",
            f"model = YOLO('{arch}.pt')\n",
            f"model.train(data='ds/data.yaml', epochs={epochs}, imgsz=640, batch=16,\n",
            "            patience=20, device=0, project='out', name='train')"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
            "m = YOLO('out/train/weights/best.pt')\n",
            "res = m.val(data='ds/data.yaml', device=0)\n",
            "print('mAP50:', res.box.map50, '| mAP50-95:', res.box.map)\n",
            "from google.colab import files\n",
            "files.download('out/train/weights/best.pt')"]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## 도구로 되돌리기\n\n",
            "받은 `best.pt`를 로컬에 두고, 오토라벨 도구의 **기존 데이터셋 연결 →",
            " 외부 학습 모델 등록**에 그 경로를 입력하면 즉시 오토라벨 엔진으로 쓰입니다.\n"]},
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU",
          "colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    return Response(
        content=json.dumps(nb, ensure_ascii=False, indent=1),
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition":
                 f"attachment; filename={proj['name']}_colab_train.ipynb"})


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

    return qa.job_status(pid)


@app.get("/api/images/{iid}/suggestions")
def suggestions(iid: int, min_conf: float = 0.4):
    """활성 모델 예측 중 기존 라벨과 겹치지 않는 것 = 누락 의심 제안."""
    from server.qa import _match

    conn = get_db()
    im = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
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
    return {
        # 모델은 찾았는데 라벨에 없음 → 추가 제안
        "missing_labels": spurious,
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
def activate_model(pid: int, mid: int):
    """이전 모델로 롤백 (성능 회귀 시)."""
    conn = get_db()
    conn.execute("UPDATE models SET active=0 WHERE project_id=?", (pid,))
    conn.execute("UPDATE models SET active=1 WHERE id=? AND project_id=?", (mid, pid))
    conn.commit()
    conn.close()
    return {"ok": True}


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

"""API 종단 회귀 테스트 — 실제로 깨졌던 경로를 재현해 막는다."""
import io
import json
import zipfile
from pathlib import Path

import pytest


def _project(client, name="t", ontology=None):
    r = client.post("/api/projects", json={
        "name": name,
        "ontology": ontology or [{"name": "person", "prompt": "person", "threshold": 0.35}]})
    assert r.status_code == 200
    return r.json()["id"]


def _model_bundle(path: Path, *, classes=None, val_map50=0.5,
                  test_map50=None, split_counts=None, class_metrics=None,
                  member="best.pt"):
    """Colab이 내려줄 모델+성능표 번들의 최소 테스트 픽스처."""
    classes = classes or ["person"]
    split_counts = split_counts or {"train": 60, "val": 30, "test": 0}
    manifest = {
        "schema_version": 1,
        "classes": classes,
        "architecture": "yolo11n",
        "approved_images": sum(split_counts.values()),
        "split_counts": split_counts,
        "epochs_requested": 60,
        "metrics": {
            "val_map50": val_map50,
            "val_map50_95": 0.25 if val_map50 is not None else None,
            "test_map50": test_map50,
            "test_map50_95": 0.2 if test_map50 is not None else None,
        },
    }
    if class_metrics is not None:
        manifest["class_metrics"] = class_metrics
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(member, b"fake-yolo-weights")
        z.writestr("autolabel-model.json", json.dumps(manifest))
    return manifest


def test_cors_allows_local_app_and_blocks_external_sites(client):
    local = client.options("/api/projects", headers={
        "Origin": "http://127.0.0.1:5173",
        "Access-Control-Request-Method": "GET",
    })
    assert local.status_code == 200
    assert local.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"

    external = client.options("/api/projects", headers={
        "Origin": "https://malicious.example",
        "Access-Control-Request-Method": "GET",
    })
    assert external.status_code == 400
    assert "access-control-allow-origin" not in external.headers


def test_project_crud_and_ontology(client):
    pid = _project(client)
    assert any(p["id"] == pid for p in client.get("/api/projects").json())

    onto = [{"name": "cat", "prompt": "cat", "threshold": 0.4}]
    assert client.put(f"/api/projects/{pid}/ontology", json={"ontology": onto}).json()["ok"]
    assert client.get(f"/api/projects/{pid}").json()["ontology"] == onto

    assert client.delete(f"/api/projects/{pid}").json()["ok"]
    assert not any(p["id"] == pid for p in client.get("/api/projects").json())


def test_foundation_profile_endpoint_starts_as_learning(client):
    pid = _project(client, "foundation-profile")
    r = client.get(f"/api/projects/{pid}/foundation-profile")
    assert r.status_code == 200
    profile = r.json()
    assert profile["status"] == "learning"
    assert profile["reviewed_images"] == 0
    assert profile["remaining_images"] == 3
    assert profile["classes"][0]["selection"] == "comparing"


def test_upload_annotate_export_roundtrip(client, make_image, tmp_path):
    pid = _project(client)
    img = make_image(tmp_path, "a.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("a.jpg", f, "image/jpeg"))]).json()["saved"][0]

    anns = [{"class_name": "person", "bbox": [10, 20, 100, 200], "source": "human"}]
    assert client.put(f"/api/images/{iid}/annotations", json={"annotations": anns}).json()["ok"]
    got = client.get(f"/api/images/{iid}/annotations").json()
    assert len(got) == 1 and got[0]["bbox"] == [10, 20, 100, 200]

    coco = client.get(f"/api/projects/{pid}/export?fmt=coco").json()
    assert len(coco["annotations"]) == 1
    assert coco["categories"][0]["name"] == "person"

    zip_res = client.get(f"/api/projects/{pid}/export.zip?fmt=yolo")
    assert zip_res.status_code == 200 and zip_res.content[:2] == b"PK"
    names = zipfile.ZipFile(io.BytesIO(zip_res.content)).namelist()
    assert "data.yaml" in names
    assert sum(n.startswith("images/") for n in names) == 1, names
    assert sum(n.startswith("labels/") for n in names) == 1, names

    # COCO zip은 자체 정합해야 한다 — json의 file_name이 zip 안 실제 경로와 일치
    cz = zipfile.ZipFile(io.BytesIO(
        client.get(f"/api/projects/{pid}/export.zip?fmt=coco").content))
    meta = json.loads(cz.read("annotations.json"))
    inside = set(cz.namelist())
    for entry in meta["images"]:
        assert f"images/{entry['file_name']}" in inside, (entry, inside)


def test_linked_import_reads_yolo_labels_without_copying(client, make_image, tmp_path):
    """연결 임포트: 원본 경로 참조 + 정규화 좌표 → 픽셀 변환."""
    imgs, labels = tmp_path / "images", tmp_path / "labels"
    make_image(imgs, "x.jpg", size=(400, 200))
    labels.mkdir(parents=True)
    (labels / "x.txt").write_text("0 0.5 0.5 0.25 0.5\n")

    pid = _project(client, "linked")
    prev = client.post("/api/import/preview", json={
        "images_dir": str(imgs), "labels_dir": str(labels)}).json()
    assert prev["format"] == "yolo" and prev["images"] == 1

    client.post(f"/api/projects/{pid}/import", json={
        "images_dir": str(imgs), "labels_dir": str(labels), "class_names": ["person"]})
    for _ in range(50):
        if client.get(f"/api/projects/{pid}/import/status").json()["status"] != "running":
            break
        import time
        time.sleep(0.1)

    rows = client.get(f"/api/projects/{pid}/images").json()
    assert len(rows) == 1 and rows[0]["ann_count"] == 1
    a = client.get(f"/api/images/{rows[0]['id']}/annotations").json()[0]
    # cx=0.5,cy=0.5,w=0.25,h=0.5 on 400x200 → x=150,y=50,w=100,h=100
    assert a["bbox"] == [150.0, 50.0, 100.0, 100.0]
    # 원본을 그대로 서빙해야 한다 (복사본 없음)
    assert client.get(f"/api/images/{rows[0]['id']}/file").status_code == 200

    # 익스포트도 src_path를 따라가야 한다 — 업로드 경로만 보면 이미지가 통째로
    # 빠진 채 data.yaml만 든 zip이 나간다 (실제로 발생했던 사고)
    z = client.get(f"/api/projects/{pid}/export.zip?fmt=yolo")
    assert z.headers["x-images-missing"] == "0"
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert any(n.startswith("images/") for n in names), names
    assert any(n.startswith("labels/") for n in names), names

    # 프로젝트 삭제는 uploads만 정리한다. 연결 임포트 원본까지 지우면 사용자
    # 데이터셋이 통째로 날아간다 — 파괴적 경로라 회귀를 테스트로 못박는다.
    assert client.delete(f"/api/projects/{pid}").json()["ok"]
    assert (imgs / "x.jpg").exists(), "프로젝트 삭제가 원본 이미지를 지웠다"
    assert (labels / "x.txt").exists(), "프로젝트 삭제가 원본 라벨을 지웠다"


def test_auto_approve_respects_threshold(client, make_image, tmp_path):
    """검증·캘리브레이션된 전용 모델에서도 고신뢰만 승인한다."""
    pid = _project(client, "approve")
    from server.db import get_db
    conn = get_db()
    mid = conn.execute(
        "INSERT INTO models (project_id,path,map50,test_map50,train_images,active,meta) "
        "VALUES (?,?,?,?,?,?,?)",
        (pid, "/tmp/verified.pt", 0.8, 0.78, 100, 1,
         json.dumps({"quality_status": "verified"}))).lastrowid
    conn.commit()
    conn.close()
    ontology = [{"name": "person", "prompt": "person", "threshold": 0.35,
                 "approval_threshold": 0.65, "approval_precision": 0.95,
                 "approval_support": 12, "approval_source": "qa_val",
                 "approval_model_id": mid}]
    client.put(f"/api/projects/{pid}/ontology", json={"ontology": ontology})
    ids = []
    for i, conf in enumerate([0.9, 0.3]):
        img = make_image(tmp_path / f"ap{i}", f"{i}.jpg")
        with open(img, "rb") as f:
            iid = client.post(f"/api/projects/{pid}/images",
                              files=[("files", (f"{i}.jpg", f, "image/jpeg"))]).json()["saved"][0]
        client.put(f"/api/images/{iid}/annotations", json={"annotations": [
            {"class_name": "person", "bbox": [1, 1, 10, 10],
             "confidence": conf, "source": "model",
             "meta": {"engine": "student", "model_id": mid}}]})
        client.put(f"/api/images/{iid}/status", json={"status": "prelabeled"})
        ids.append(iid)

    dry = client.post(f"/api/projects/{pid}/auto-approve",
                      json={"min_conf": 0.7, "dry_run": True}).json()
    assert dry["approved"] == 1 and dry["skipped_low_confidence"] == 1

    client.post(f"/api/projects/{pid}/auto-approve", json={"min_conf": 0.7})
    states = {r["id"]: r["status"] for r in client.get(f"/api/projects/{pid}/images").json()}
    assert states[ids[0]] == "approved" and states[ids[1]] == "prelabeled"


def test_auto_approve_blocks_foundation_and_uncalibrated_models(client, make_image, tmp_path):
    """SAM/GDINO의 높은 confidence와 미캘리브레이션 학생 모델은 자동승인 금지."""
    pid = _project(client, "approve-safe")
    img = make_image(tmp_path / "unsafe", "unsafe.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("unsafe.jpg", f, "image/jpeg"))]).json()["saved"][0]
    client.put(f"/api/images/{iid}/annotations", json={"annotations": [{
        "class_name": "person", "bbox": [1, 1, 10, 10], "confidence": 0.99,
        "source": "model", "meta": {"engine": "sam3"}}]})
    client.put(f"/api/images/{iid}/status", json={"status": "prelabeled"})

    unsafe = client.post(f"/api/projects/{pid}/auto-approve",
                         json={"dry_run": True}).json()
    assert unsafe["approved"] == 0 and unsafe["skipped_unsafe_model"] == 1
    assert "자동 승인하지 않습니다" in unsafe["blocked_reason"]

    from server.db import get_db
    conn = get_db()
    mid = conn.execute(
        "INSERT INTO models (project_id,path,map50,test_map50,train_images,active,meta) "
        "VALUES (?,?,?,?,?,?,?)",
        (pid, "/tmp/student.pt", 0.8, 0.78, 100, 1,
         json.dumps({"quality_status": "verified"}))).lastrowid
    conn.execute("UPDATE annotations SET meta=? WHERE image_id=?",
                 (json.dumps({"engine": "student", "model_id": mid}), iid))
    conn.commit()
    conn.close()
    client.put(f"/api/projects/{pid}/ontology", json={"ontology": [{
        "name": "person", "prompt": "person", "threshold": 0.35,
        "approval_threshold": 0.7, "approval_precision": 0.95,
        "approval_support": 20, "approval_source": "qa_val",
        "approval_model_id": mid + 1,
    }]})

    uncalibrated = client.post(f"/api/projects/{pid}/auto-approve",
                               json={"dry_run": True}).json()
    assert uncalibrated["approved"] == 0
    assert uncalibrated["skipped_uncalibrated"] == 1
    assert "QA 분석" in uncalibrated["blocked_reason"]


def test_auto_approve_never_approves_empty_without_negative_calibration(client, make_image, tmp_path):
    pid = _project(client, "approve-empty")
    img = make_image(tmp_path / "empty", "empty.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("empty.jpg", f, "image/jpeg"))]).json()["saved"][0]
    client.put(f"/api/images/{iid}/status", json={"status": "prelabeled"})

    dry = client.post(f"/api/projects/{pid}/auto-approve",
                      json={"dry_run": True, "require_labeled": False}).json()
    assert dry["approved"] == 0 and dry["skipped_no_label"] == 1


def test_bulk_status_and_next_to_label(client, make_image, tmp_path):
    pid = _project(client, "bulk")
    ids = []
    for i in range(3):
        img = make_image(tmp_path / f"b{i}", f"{i}.jpg")
        with open(img, "rb") as f:
            ids.append(client.post(f"/api/projects/{pid}/images",
                       files=[("files", (f"{i}.jpg", f, "image/jpeg"))]).json()["saved"][0])
    r = client.post("/api/images/bulk-status",
                    json={"image_ids": ids, "status": "approved"}).json()
    assert r["count"] == 3
    assert all(x["status"] == "approved"
               for x in client.get(f"/api/projects/{pid}/images").json())

    rec = client.get(f"/api/projects/{pid}/next-to-label?n=2").json()
    assert "recommended" in rec  # 승인된 것은 후보에서 빠진다
    assert rec["total_candidates"] == 0


def test_approval_below_training_minimum_does_not_schedule_timer(
        client, make_image, tmp_path):
    from server import train

    pid = _project(client, "no-premature-train")
    img = make_image(tmp_path / "no-premature", "a.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("a.jpg", f, "image/jpeg"))]).json()["saved"][0]

    response = client.put(f"/api/images/{iid}/status", json={"status": "approved"}).json()
    assert response["train"] == {"status": "skipped", "approved": 1, "need": 8}
    assert pid not in train._timers


def test_acceptance_plan_and_result(client, make_image, tmp_path):
    pid = _project(client, "accept")
    for i in range(40):
        img = make_image(tmp_path / "acc", f"{i}.jpg")
        with open(img, "rb") as f:
            iid = client.post(f"/api/projects/{pid}/images",
                              files=[("files", (f"{i}.jpg", f, "image/jpeg"))]).json()["saved"][0]
        client.put(f"/api/images/{iid}/status", json={"status": "prelabeled"})

    plan = client.post(f"/api/projects/{pid}/acceptance-plan", json={}).json()
    assert plan["lot_size"] == 40 and plan["sample_size"] <= 40
    assert len(plan["sample_image_ids"]) == plan["sample_size"]

    res = client.post(f"/api/projects/{pid}/acceptance-result", json={
        "sample_size": plan["sample_size"], "defects": 0,
        "max_defects": plan["max_defects"], "apply": True,
        "status": plan["status"], "lot_token": plan["lot_token"]}).json()
    assert res["accepted"] and res["approved_images"] == 40


def test_acceptance_rejects_changed_lot(client, make_image, tmp_path):
    """검수 계획 뒤 들어온 이미지를 검사 없이 함께 승인하면 안 된다."""
    pid = _project(client, "accept-snapshot")

    def add(name):
        img = make_image(tmp_path / "acc-snap", name)
        with open(img, "rb") as f:
            iid = client.post(f"/api/projects/{pid}/images",
                              files=[("files", (name, f, "image/jpeg"))]).json()["saved"][0]
        client.put(f"/api/images/{iid}/status", json={"status": "prelabeled"})
        return iid

    add("before.jpg")
    plan = client.post(f"/api/projects/{pid}/acceptance-plan", json={}).json()
    late = add("late.jpg")
    r = client.post(f"/api/projects/{pid}/acceptance-result", json={
        "sample_size": plan["sample_size"], "defects": 0,
        "max_defects": plan["max_defects"], "status": plan["status"],
        "lot_token": plan["lot_token"], "apply": True,
    })
    assert r.status_code == 409
    states = {im["id"]: im["status"] for im in client.get(
        f"/api/projects/{pid}/images").json()}
    assert states[late] == "prelabeled" and set(states.values()) == {"prelabeled"}


def test_api_rejects_invalid_status_and_cross_project_ids(client, make_image, tmp_path):
    """오타 상태와 프로젝트 경계를 넘는 일괄 작업은 데이터 상태를 오염시킨다."""
    pids, ids = [], []
    for i in range(2):
        pid = _project(client, f"scope-{i}")
        pids.append(pid)
        img = make_image(tmp_path / f"scope-{i}", "a.jpg")
        with open(img, "rb") as f:
            ids.append(client.post(f"/api/projects/{pid}/images",
                       files=[("files", ("a.jpg", f, "image/jpeg"))]).json()["saved"][0])

    assert client.put(f"/api/images/{ids[0]}/status", json={"status": "typo"}).status_code == 400
    assert client.put(f"/api/images/{ids[0]}/annotations", json={"annotations": [{
        "class_name": "person", "bbox": [1, 1, -2, 3]}]}).status_code == 400
    assert client.post("/api/images/bulk-status", json={
        "image_ids": ids, "status": "approved"}).status_code == 400
    assert client.post(f"/api/projects/{pids[0]}/autolabel", json={
        "image_ids": [ids[1]], "masks": False}).status_code == 400
    assert client.put("/api/projects/999999/ontology", json={"ontology": []}).status_code == 404
    assert client.post(f"/api/projects/{pids[0]}/acceptance-plan", json={
        "target_error_rate": 0}).status_code == 400

    img = make_image(tmp_path / "missing-project", "a.jpg")
    with open(img, "rb") as f:
        assert client.post("/api/projects/999999/images",
                           files=[("files", ("a.jpg", f, "image/jpeg"))]).status_code == 404


def test_rejected_images_are_excluded_from_export(client, make_image, tmp_path):
    """거부 = "이 데이터는 쓰지 말라". 그런데 익스포트가 상태를 안 봤다.

    학습은 status='approved'만 쓰므로 안전했지만, 익스포트한 zip에는 거부한
    이미지와 라벨이 그대로 실려 나갔다.
    """
    pid = _project(client, "reject")
    ids = []
    for i in range(2):
        img = make_image(tmp_path / "rej", f"{i}.jpg")
        with open(img, "rb") as f:
            iid = client.post(f"/api/projects/{pid}/images",
                              files=[("files", (f"{i}.jpg", f, "image/jpeg"))]).json()["saved"][0]
        client.put(f"/api/images/{iid}/annotations", json={"annotations": [
            {"class_name": "person", "bbox": [1, 2, 3, 4], "source": "human"}]})
        ids.append(iid)

    client.post("/api/images/bulk-status", json={"image_ids": [ids[0]], "status": "rejected"})

    coco = client.get(f"/api/projects/{pid}/export?fmt=coco").json()
    assert [i["id"] for i in coco["images"]] == [ids[1]], coco["images"]
    assert len(coco["annotations"]) == 1

    z = zipfile.ZipFile(io.BytesIO(
        client.get(f"/api/projects/{pid}/export.zip?fmt=yolo").content))
    names = z.namelist()
    assert not any(f"{ids[0]}_" in n for n in names), names
    assert any(f"{ids[1]}_" in n for n in names), names

    # 되살릴 필요가 있으면 명시해서 포함할 수 있다
    all_coco = client.get(
        f"/api/projects/{pid}/export?fmt=coco&include_rejected=true").json()
    assert len(all_coco["images"]) == 2


def test_thumb_is_small_and_cached(client, make_image, tmp_path):
    """목록 썸네일은 원본보다 훨씬 작아야 한다.

    예전엔 목록이 원본을 그대로 받아 44x44로 줄여 그렸다 — signature 143장은
    스크롤 한 번에 9.3MB, 2MB 사진 1만 장이면 20GB를 썸네일에만 쓴다.
    """
    pid = _project(client, "thumb")
    img = make_image(tmp_path / "th", "big.jpg", size=(1600, 1200))
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("big.jpg", f, "image/jpeg"))]).json()["saved"][0]

    r = client.get(f"/api/images/{iid}/thumb")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    full = client.get(f"/api/images/{iid}/file")
    assert len(r.content) < len(full.content) / 4, (len(r.content), len(full.content))
    assert "max-age" in r.headers.get("cache-control", "")

    # 실제로 요청한 크기 안에 들어와야 한다
    from PIL import Image as PILImage
    w, h = PILImage.open(io.BytesIO(r.content)).size
    assert max(w, h) <= 96, (w, h)
    # 상한을 넘겨 요청해도 THUMB_MAX로 잘린다 (원본을 되돌려주면 의미가 없다)
    big = client.get(f"/api/images/{iid}/thumb?size=9999")
    w2, h2 = PILImage.open(io.BytesIO(big.content)).size
    assert max(w2, h2) <= 128, (w2, h2)

    # 두 번째 요청은 캐시 파일에서 나와야 한다 (내용 동일)
    assert client.get(f"/api/images/{iid}/thumb").content == r.content


def test_prompt_lab_rejects_empty_input(client, make_image, tmp_path):
    """프롬프트 실험은 비교 대상이 있어야 한다."""
    pid = _project(client, "lab")
    assert client.post(f"/api/projects/{pid}/prompt-lab",
                       json={"prompts": []}).status_code == 400
    assert client.post(f"/api/projects/{pid}/prompt-lab",
                       json={"prompts": ["  ", ""]}).status_code == 400
    # 프롬프트는 있지만 이미지가 없으면 그것도 알려야 한다
    assert client.post(f"/api/projects/{pid}/prompt-lab",
                       json={"prompts": ["person"]}).status_code == 400


def test_prompt_lab_dedupes_candidates():
    """같은 프롬프트를 두 번 돌리면 추론 시간만 쓰고 결과표에 같은 줄이 두 개 뜬다."""
    from server.main import MAX_LAB_PROMPTS, _unique_prompts

    assert _unique_prompts([" person ", "person", "dog", "", "person", "dog"]) \
        == ["person", "dog"]
    assert _unique_prompts([None, 3, "  ", "cat"]) == ["cat"]  # 잘못된 입력도 흘리지 않는다
    assert len(_unique_prompts([f"p{i}" for i in range(50)])) == MAX_LAB_PROMPTS


def test_export_works_with_non_ascii_project_name(client, make_image, tmp_path):
    """한글 프로젝트명으로도 익스포트가 되어야 한다.

    HTTP 헤더는 latin-1만 담는다. Content-Disposition에 프로젝트명을 그대로
    넣던 탓에 한글 이름이면 UnicodeEncodeError로 500이 났다 — 한국어 사용자는
    익스포트가 통째로 막혀 있었고, QA 프로젝트가 전부 ASCII 이름이라 놓쳤다.
    """
    pid = _project(client, "한글 프로젝트 테스트")
    img = make_image(tmp_path / "ko", "a.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("a.jpg", f, "image/jpeg"))]).json()["saved"][0]
    client.put(f"/api/images/{iid}/annotations", json={"annotations": [
        {"class_name": "person", "bbox": [1, 2, 3, 4], "source": "human"}]})

    for fmt in ("yolo", "coco"):
        r = client.get(f"/api/projects/{pid}/export.zip?fmt={fmt}")
        assert r.status_code == 200, (fmt, r.status_code)
        cd = r.headers["content-disposition"]
        cd.encode("latin-1")  # 헤더가 실제로 전송 가능해야 한다
        assert "filename*=UTF-8''" in cd, cd

    nb = client.get(f"/api/projects/{pid}/colab-notebook")
    assert nb.status_code == 200
    nb.headers["content-disposition"].encode("latin-1")
    cells = nb.json()["cells"]
    code = "\n".join("".join(cell.get("source", [])) for cell in cells
                     if cell["cell_type"] == "code")
    assert "pathlib.Path(model.trainer.best)" in code
    assert "test_res.nt_per_class" in code and "'class_metrics': class_metrics" in code
    assert "shutil.rmtree(bundle)" in code
    assert "YOLO('out/train/weights/best.pt')" not in code
    assert cells[-2]["source"] == ["from google.colab import files\n",
                                    "files.download('autolabel-model.zip')"]


def test_model_import_and_rollback(client, tmp_path):
    """외부 .pt는 검증 후 비활성 등록되고, 명시 적용·롤백만 활성화한다."""
    pid = _project(client, "models")
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"fake-weights-a")
    b.write_bytes(b"fake-weights-b")

    r1 = client.post(f"/api/projects/{pid}/models/import", json={
        "path": str(a), "names": ["person"], "map50": 0.5}).json()
    assert r1["quality_status"] == "provisional" and not r1["active"]
    assert client.get(f"/api/projects/{pid}/train/status").json()["active_model"] is None
    assert client.post(f"/api/projects/{pid}/models/{r1['id']}/activate").json()["ok"]

    # 두 번째 등록도 현재 챔피언을 조용히 교체하면 안 된다.
    r2 = client.post(f"/api/projects/{pid}/models/import", json={
        "path": str(b), "names": ["person"], "map50": 0.7}).json()
    models = client.get(f"/api/projects/{pid}/models").json()
    assert [m["id"] for m in models if m["active"]] == [r1["id"]], models
    assert client.post(f"/api/projects/{pid}/models/{r2['id']}/activate").json()["ok"]

    # 롤백
    assert client.post(f"/api/projects/{pid}/models/{r1['id']}/activate").json()["ok"]
    assert client.get(f"/api/projects/{pid}/train/status").json()["active_model"]["id"] == r1["id"]


def test_unverified_active_model_can_still_be_downloaded(client, tmp_path):
    """사용자가 강제 적용한 raw .pt도 다운로드 경로 자체는 깨지면 안 된다."""
    from server.db import get_db

    pid = _project(client, "download-unverified")
    weights = tmp_path / "raw.pt"
    weights.write_bytes(b"raw-weights")
    conn = get_db()
    conn.execute(
        "INSERT INTO models (project_id,path,map50,train_images,active,meta) "
        "VALUES (?,?,?,?,?,?)",
        (pid, str(weights), None, 0, 1, json.dumps({"quality_status": "unverified"})))
    conn.commit()
    conn.close()

    response = client.get(f"/api/projects/{pid}/model")
    assert response.status_code == 200 and response.content == b"raw-weights"
    disposition = response.headers["content-disposition"]
    assert "model_p" in disposition and "unverified.pt" in disposition


def test_model_bundle_is_validated_copied_and_not_auto_activated(client, tmp_path):
    """Colab 번들은 클래스·성능표를 읽고 관리 경로에 복사한 뒤 대기시킨다."""
    pid = _project(client, "bundle")
    bundle = tmp_path / "autolabel-model.zip"
    _model_bundle(bundle, val_map50=0.62,
                  split_counts={"train": 60, "val": 30, "test": 0})

    r = client.post(f"/api/projects/{pid}/models/import", json={"path": str(bundle)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["quality_status"] == "provisional"
    assert not body["active"] and body["metrics"]["val_map50"] == 0.62
    assert client.get(f"/api/projects/{pid}/train/status").json()["active_model"] is None

    model = client.get(f"/api/projects/{pid}/models").json()[0]
    stored = Path(model["path"])
    assert stored.exists() and stored != bundle and stored.read_bytes() == b"fake-yolo-weights"
    assert model["train_images"] == 90
    assert model["meta"]["quality_status"] == "provisional"


def test_model_bundle_blocks_bad_quality_class_mismatch_and_zip_escape(client, tmp_path):
    pid = _project(client, "bundle-guards")

    mismatch = tmp_path / "mismatch.zip"
    _model_bundle(mismatch, classes=["cat"])
    r = client.post(f"/api/projects/{pid}/models/import", json={"path": str(mismatch)})
    assert r.status_code == 400 and "클래스" in r.text

    failed = tmp_path / "failed.zip"
    _model_bundle(failed, val_map50=0.01)
    body = client.post(f"/api/projects/{pid}/models/import", json={"path": str(failed)}).json()
    assert body["quality_status"] == "failed" and not body["active"]
    blocked = client.post(f"/api/projects/{pid}/models/{body['id']}/activate")
    assert blocked.status_code == 409 and "품질" in blocked.text

    escaped = tmp_path / "escaped.zip"
    _model_bundle(escaped, member="../best.pt")
    r = client.post(f"/api/projects/{pid}/models/import", json={"path": str(escaped)})
    assert r.status_code == 400 and "안전하지" in r.text


def test_model_bundle_blocks_weak_class_hidden_by_good_average(client, tmp_path):
    """전체 평균이 좋아도 실사용 클래스 하나가 무너지면 전문 모델로 적용하지 않는다."""
    pid = _project(client, "class-gate", ontology=[
        {"name": "crazing", "prompt": "crazing", "threshold": 0.35},
        {"name": "scratches", "prompt": "scratches", "threshold": 0.35},
    ])
    bundle = tmp_path / "weak-class.zip"
    class_metrics = {
        "crazing": {"test_map50": 0.064, "test_map50_95": 0.018,
                    "test_instances": 8},
        "scratches": {"test_map50": 0.619, "test_map50_95": 0.274,
                      "test_instances": 11},
    }
    _model_bundle(bundle, classes=["crazing", "scratches"], val_map50=0.59,
                  test_map50=0.52, split_counts={"train": 70, "val": 30, "test": 20},
                  class_metrics=class_metrics)

    body = client.post(f"/api/projects/{pid}/models/import",
                       json={"path": str(bundle)}).json()
    assert body["quality_status"] == "failed" and not body["active"]
    assert "crazing 0.064" in body["quality_reason"]
    assert body["class_metrics"] == class_metrics
    assert client.post(f"/api/projects/{pid}/models/{body['id']}/activate").status_code == 409


def test_model_bundle_can_be_selected_in_browser_without_pasting_a_path(client, tmp_path):
    pid = _project(client, "browser-upload")
    bundle = tmp_path / "autolabel-model.zip"
    _model_bundle(bundle, val_map50=0.62,
                  split_counts={"train": 60, "val": 30, "test": 0})

    response = client.post(f"/api/projects/{pid}/models/import-upload", files={
        "file": (bundle.name, bundle.read_bytes(), "application/zip")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["quality_status"] == "provisional" and not body["active"]
    assert Path(body["path"]).read_bytes() == b"fake-yolo-weights"
    model = client.get(f"/api/projects/{pid}/models").json()[0]
    assert model["meta"]["source_bundle"] == "autolabel-model.zip"
    assert not list(Path(tmp_path).parent.glob(".model-upload-*"))


def test_training_worker_starts_from_code_root(client, tmp_path, monkeypatch):
    """데이터 저장 루트를 바꿔도 워커의 cwd까지 바뀌면 server 모듈을 못 찾는다."""
    import os

    from server import train

    pid = _project(client, "worker-cwd")
    monkeypatch.setattr(train, "RUNS", tmp_path / "runs")
    train._procs.clear()
    captured = {}

    class FakeProc:
        pid = os.getpid()

        def poll(self):
            return None

    def fake_popen(args, **kwargs):
        captured.update(args=args, cwd=kwargs["cwd"])
        return FakeProc()

    monkeypatch.setattr(train.subprocess, "Popen", fake_popen)
    result = client.post(f"/api/projects/{pid}/train").json()

    assert result["status"] == "running"
    assert captured["cwd"] == train.CODE_ROOT
    assert (captured["cwd"] / "server" / "train_worker.py").exists()


def test_training_readiness_previews_split_and_trigger(client, make_image, tmp_path):
    """학습을 시작하기 전에 UI가 실제 분할과 다음 자동 조건을 알 수 있어야 한다."""
    pid = _project(client, "readiness")
    files = []
    for i in range(8):
        path = make_image(tmp_path, f"ready-{i}.jpg")
        files.append(("files", (path.name, path.read_bytes(), "image/jpeg")))
    saved = client.post(f"/api/projects/{pid}/images", files=files).json()["saved"]
    # 상태 API는 자동 학습 디바운스를 예약하므로, 이 읽기 전용 준비도 테스트는
    # DB 상태만 바꿔 백그라운드 워커가 우연히 뜨지 않게 격리한다.
    from server.db import get_db
    conn = get_db()
    conn.executemany("UPDATE images SET status='approved' WHERE id=?",
                     [(iid,) for iid in saved])
    conn.commit()
    conn.close()

    r = client.get(f"/api/projects/{pid}/train/readiness")
    assert r.status_code == 200
    ready = r.json()
    assert ready["approved"] == 8
    assert sum(ready["split_counts"].values()) == 8
    assert ready["split_counts"]["train"] >= 4
    assert ready["ready_manual"] and ready["ready_auto"]
    assert ready["stage"] == "experiment" and not ready["professional_ready"]
    assert ready["split_counts"]["test"] == 0
    assert any("홀드아웃" in warning for warning in ready["warnings"])
    assert ready["remaining_auto"] == 0 and ready["next_auto_at"] == 8
    assert ready["recommended_arch"] == "yolo11n" and ready["expected_epochs"] == 60

    assert client.get("/api/projects/999999/train/readiness").status_code == 404


def test_rollback_to_unknown_model_is_rejected(client, tmp_path):
    """없는 모델로 롤백하면 거부해야 한다.

    예전에는 전부 비활성화만 하고 아무것도 활성화하지 않은 채 ok를 반환해,
    클릭 한 번으로 전용 모델이 사라지고 오토라벨이 조용히 파운데이션으로
    강등됐다.
    """
    pid = _project(client, "rollback")
    w = tmp_path / "w.pt"
    w.write_bytes(b"fake-weights")
    client.post(f"/api/projects/{pid}/models/import", json={
        "path": str(w), "names": ["person"], "map50": 0.5})
    model = client.get(f"/api/projects/{pid}/models").json()[0]
    assert client.post(f"/api/projects/{pid}/models/{model['id']}/activate").status_code == 200
    before = client.get(f"/api/projects/{pid}/train/status").json()["active_model"]

    assert client.post(f"/api/projects/{pid}/models/999999/activate").status_code == 404
    after = client.get(f"/api/projects/{pid}/train/status").json()["active_model"]
    assert after and after["id"] == before["id"], "실패한 롤백이 활성 모델을 날렸다"

    # 남의 프로젝트 모델도 마찬가지로 거부
    other = _project(client, "rollback-other")
    assert client.post(f"/api/projects/{other}/models/{before['id']}/activate").status_code == 404


def test_delete_linked_image_keeps_original_file(client, make_image, tmp_path):
    """연결 임포트 이미지 삭제는 프로젝트에서만 뺀다 — 파일은 사용자 원본이다.

    실측 결함(감사): delete_image가 이미지 경로를 검사 없이 unlink해, 복사
    없이 연결한 외부 데이터셋의 원본 파일이 디스크에서 영구 삭제됐다.
    """
    import time

    imgs = tmp_path / "orig"
    make_image(imgs, "keep.jpg")
    pid = _project(client, "del-linked")
    client.post(f"/api/projects/{pid}/import", json={
        "images_dir": str(imgs), "class_names": ["person"]})
    for _ in range(50):
        if client.get(f"/api/projects/{pid}/import/status").json()["status"] != "running":
            break
        time.sleep(0.1)

    rows = client.get(f"/api/projects/{pid}/images").json()
    assert len(rows) == 1
    assert client.delete(f"/api/images/{rows[0]['id']}").json()["ok"]
    assert (imgs / "keep.jpg").exists(), "이미지 삭제가 연결 원본 파일을 지웠다"
    assert client.get(f"/api/projects/{pid}/images").json() == []

    # 업로드 이미지의 복사본은 우리가 만든 파일이라 지우는 게 맞다
    up = make_image(tmp_path / "up", "copy.jpg")
    with open(up, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("copy.jpg", f, "image/jpeg"))]).json()["saved"][0]
    assert client.get(f"/api/images/{iid}/file").status_code == 200
    client.delete(f"/api/images/{iid}")
    assert client.get(f"/api/images/{iid}/file").status_code == 404
    assert up.exists(), "업로드 원본(사용자 파일)은 애초에 서버가 만진 적 없어야 한다"


def test_batch_autolabel_default_skips_reviewed_images(client, make_image, tmp_path,
                                                       monkeypatch):
    """기본 배치 대상은 리뷰 전 이미지만.

    실측 결함(감사): 대상 쿼리에 상태 필터가 없어 승인 이미지의 검토된 라벨을
    무검토 검출로 교체했다. status는 approved로 남아 오염된 라벨이 승인
    데이터로 둔갑해 학습셋까지 흘러간다.
    """
    import time

    from server import main as m

    pid = _project(client, "batch-target")
    ids = {}
    for st in ("approved", "prelabeled", "unlabeled"):
        img = make_image(tmp_path / "bt", f"{st}.jpg")
        with open(img, "rb") as f:
            iid = client.post(f"/api/projects/{pid}/images",
                              files=[("files", (f"{st}.jpg", f, "image/jpeg"))]).json()["saved"][0]
        if st != "unlabeled":
            client.put(f"/api/images/{iid}/status", json={"status": st})
        ids[st] = iid

    seen = {}

    def fake_run(pid_, image_ids, ontology, masks, profile="balanced", candidate_conf=0.10):
        seen["ids"] = image_ids
        seen["profile"] = profile
        from server import jobs as _jobs_mod
        _jobs_mod.update("autolabel", pid_, status="completed")

    monkeypatch.setattr(m, "_run_batch", fake_run)
    r = client.post(f"/api/projects/{pid}/autolabel", json={}).json()
    assert r["total"] == 2, r
    for _ in range(40):
        if "ids" in seen:
            break
        time.sleep(0.05)
    assert set(seen["ids"]) == {ids["prelabeled"], ids["unlabeled"]}, seen
    assert seen["profile"] == "balanced"

    # 명시적으로 지목하면 승인 이미지도 재실행할 수 있어야 한다 (의도된 탈출구)
    r = client.post(f"/api/projects/{pid}/autolabel",
                    json={"image_ids": [ids["approved"]]}).json()
    assert r["total"] == 1

    # 전부 리뷰가 끝났으면 대상 0장 — 돌리는 척하지 말고 즉시 알린다
    client.post("/api/images/bulk-status",
                json={"image_ids": list(ids.values()), "status": "approved"})
    r = client.post(f"/api/projects/{pid}/autolabel", json={}).json()
    assert r["status"] == "completed" and r["total"] == 0
    assert "덮어쓰지 않습니다" in r["advice"]


def test_batch_autolabel_preserves_human_boxes_and_records_engine(
        client, make_image, tmp_path, monkeypatch):
    """재실행은 사람 라벨을 중복시키지 않고 실제 엔진·프로필을 기록한다."""
    from server import main

    ontology = [
        {"name": "defect", "threshold": 0.35},
        {"name": "defect.part", "threshold": 0.35},
    ]
    pid = _project(client, "batch-merge", ontology)
    img = make_image(tmp_path / "batch-merge", "a.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("a.jpg", f, "image/jpeg"))]).json()["saved"][0]
    client.put(f"/api/images/{iid}/annotations", json={"annotations": [
        {"class_name": "defect", "bbox": [0, 0, 100, 100], "source": "human"},
        {"class_name": "defect", "bbox": [300, 300, 20, 20], "source": "model"},
    ]})

    monkeypatch.setattr(main, "_detect_auto", lambda *_args, **_kw: ([
        {"class_name": "defect", "bbox": [10, 10, 20, 20], "confidence": 0.9},
        {"class_name": "defect", "bbox": [200, 200, 20, 20], "confidence": 0.8},
        {"class_name": "defect.part", "bbox": [10, 10, 20, 20], "confidence": 0.7},
    ], "student(fake)+recall"))

    main._run_batch(pid, [iid], ontology, False, "recall", 0.1)
    anns = client.get(f"/api/images/{iid}/annotations").json()

    assert len(anns) == 3, anns
    assert sum(a["source"] == "human" for a in anns) == 1
    assert not any(a["source"] == "model" and a["bbox"] == [10, 10, 20, 20]
                   and a["class_name"] == "defect" for a in anns)
    model_anns = [a for a in anns if a["source"] == "model"]
    assert all(a["meta"]["engine"] == "student(fake)+recall" for a in model_anns)
    assert all(a["meta"]["profile"] == "recall" for a in model_anns)

    one = client.post(f"/api/images/{iid}/autolabel", json={
        "profile": "recall", "candidate_conf": 0.1, "masks": False}).json()
    assert one["profile"] == "recall"
    assert all(d["meta"]["engine"] == "student(fake)+recall" for d in one["detections"])
    assert all(d["meta"]["profile"] == "recall" for d in one["detections"])

    bad = client.post(f"/api/images/{iid}/autolabel", json={"profile": "magic"})
    assert bad.status_code == 400


def test_routed_batch_keeps_collecting_both_engine_samples(
        client, make_image, tmp_path, monkeypatch):
    """경로가 정해진 뒤에도 재탐색 주기마다 양쪽을 돌려 근거가 계속 자란다.

    이게 없으면 sam3_ran=1 AND gdino_ran=1 행이 더 안 생겨 build_profile이
    초기 표본에 갇힌다 — 틀린 라우팅을 뒤집을 데이터가 영원히 안 쌓인다.
    """
    from server import ensemble, foundation, main
    from server.db import get_db
    from server.ensemble import fuse_foundation_detections

    ontology = [{"name": "defect", "prompt": "surface defect"}]
    pid = _project(client, "route-explore", ontology)

    # SEED_IMAGES만큼 승인된 양쪽 감사 표본을 만들어 seed 단계를 끝낸다.
    conn = get_db()
    for i in range(ensemble.SEED_IMAGES):
        seed_id = conn.execute(
            "INSERT INTO images (project_id,file_name,width,height,status) "
            "VALUES (?,?,?,?,?)", (pid, f"seed{i}.jpg", 200, 200, "approved")).lastrowid
        conn.execute(
            "INSERT INTO annotations (image_id,class_name,bbox,source) VALUES (?,?,?,?)",
            (seed_id, "defect", "[10,10,20,20]", "human"))
        # SAM3는 정답과 일치, GDINO는 빗나감 -> 검수 비용이 낮은 sam3로 라우팅된다
        foundation.replace_audit(conn, pid, seed_id, fuse_foundation_detections(
            [{"class_name": "defect", "bbox": [10, 10, 20, 20], "confidence": 0.8}],
            [{"class_name": "defect", "bbox": [100, 100, 20, 20], "confidence": 0.7}],
        ), "ensemble(sam3+gdino)")
    conn.commit()
    assert foundation.audited_both(conn, pid) == ensemble.SEED_IMAGES
    conn.close()

    img = make_image(tmp_path / "route-explore", "b.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("b.jpg", f, "image/jpeg"))]).json()["saved"][0]

    seen = []

    def fake_detect(_pid, _image, _onto, engine="auto", *_a, **_kw):
        seen.append(engine)
        used = ("ensemble(sam3+gdino)" if engine == "ensemble" else
                "routed(sam3)" if engine == "routed" else "sam3(단독)")
        return [{"class_name": "defect", "bbox": [10, 10, 20, 20], "confidence": 0.9}], used

    monkeypatch.setattr(main, "_detect_auto", fake_detect)
    batch = [iid] * (ensemble.EXPLORE_EVERY + 2)
    main._run_batch(pid, batch, ontology, False)

    # seed는 끝났으므로 EXPLORE_EVERY 주기의 장만 양쪽을 돌린다.
    assert seen[0] == "ensemble"
    assert seen[1:ensemble.EXPLORE_EVERY] == ["routed"] * (ensemble.EXPLORE_EVERY - 1)
    assert seen[ensemble.EXPLORE_EVERY] == "ensemble"

    plan = client.get(f"/api/projects/{pid}/autolabel/status").json()["engine_plan"]
    assert plan["mode"] == "routed"
    assert plan["both_engine_images"] == 2
    assert plan["seeded_before"] == ensemble.SEED_IMAGES


def test_export_skips_labels_of_removed_classes(client, make_image, tmp_path):
    """온톨로지에서 빠진 클래스의 라벨은 제외해야 한다.

    실측 결함(감사): 클래스 이름을 바꾸면 옛 이름의 라벨이 yolo에서 class 0,
    coco에서 category_id 0으로 조용히 오라벨돼 — 이 파일로 학습한 외부 모델이
    엉뚱한 클래스를 배운다.
    """
    pid = _project(client, "rename-cls", ontology=[
        {"name": "cat", "prompt": "cat", "threshold": 0.35},
        {"name": "dog", "prompt": "dog", "threshold": 0.35}])
    img = make_image(tmp_path / "rn", "a.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("a.jpg", f, "image/jpeg"))]).json()["saved"][0]
    client.put(f"/api/images/{iid}/annotations", json={"annotations": [
        {"class_name": "cat", "bbox": [1, 2, 30, 40], "source": "human"},
        {"class_name": "dog", "bbox": [50, 60, 30, 40], "source": "human"}]})
    # 클래스 이름 변경 — 어노테이션의 dog는 옛 이름으로 남는다
    client.put(f"/api/projects/{pid}/ontology", json={"ontology": [
        {"name": "cat", "prompt": "cat", "threshold": 0.35},
        {"name": "puppy", "prompt": "puppy", "threshold": 0.35}]})

    coco = client.get(f"/api/projects/{pid}/export?fmt=coco").json()
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["category_id"] == 1  # cat — 0은 존재하면 안 된다
    assert coco["skipped_unknown_class"] == 1

    yolo = client.get(f"/api/projects/{pid}/export?fmt=yolo").json()
    lines = [ln for ln in yolo["a.txt"].splitlines() if ln]
    assert len(lines) == 1 and lines[0].startswith("0 "), lines  # cat만, dog 라벨 제외

    # zip 경로도 제외 건수를 헤더로 알린다 (조용한 누락 방지)
    z = client.get(f"/api/projects/{pid}/export.zip?fmt=yolo")
    assert z.headers["x-annotations-skipped"] == "1"
    cz = client.get(f"/api/projects/{pid}/export.zip?fmt=coco")
    assert cz.headers["x-annotations-skipped"] == "1"
    meta = json.loads(zipfile.ZipFile(io.BytesIO(cz.content)).read("annotations.json"))
    assert "skipped_unknown_class" not in meta  # 표준 COCO 키가 아니다


def test_upload_reports_failures_and_sanitizes_filenames(client, make_image, tmp_path):
    """업로드는 실패를 세서 알리고, 경로 구분자 파일명은 basename으로 정규화한다.

    실측 결함(감사): 디코드 실패 파일을 조용히 건너뛰어 프론트가 전량 성공으로
    보고했고, 파일명에 '/'가 섞이면 없는 디렉터리를 가리켜 배치 전체가 500으로
    죽고 이미 쓴 파일은 고아로 남았다.
    """
    pid = _project(client, "upload-hard")
    ok = make_image(tmp_path / "uh", "good.jpg")
    with open(ok, "rb") as f:
        r = client.post(f"/api/projects/{pid}/images", files=[
            ("files", ("good.jpg", f, "image/jpeg")),
            ("files", ("broken.jpg", io.BytesIO(b"not an image"), "image/jpeg")),
        ]).json()
    assert len(r["saved"]) == 1
    assert r["failed"] == ["broken.jpg"]

    with open(ok, "rb") as f:
        r = client.post(f"/api/projects/{pid}/images", files=[
            ("files", ("sub/dir/evil.jpg", f, "image/jpeg"))])
    assert r.status_code == 200
    r = r.json()
    assert len(r["saved"]) == 1 and not r["failed"], r
    rows = client.get(f"/api/projects/{pid}/images").json()
    assert any(x["file_name"] == "evil.jpg" for x in rows), rows
    assert client.get(f"/api/images/{r['saved'][0]}/file").status_code == 200


def test_missing_ids_return_404_not_500(client):
    """없는 id는 500이 아니라 404여야 한다.

    실측 결함(감사): fetchone() 결과를 존재 확인 없이 인덱싱해 None['...']
    TypeError 500 — 다른 탭에서 삭제된 이미지의 낡은 목록으로도 도달한다.
    """
    assert client.post("/api/images/999999/autolabel", json={}).status_code == 404
    assert client.get("/api/images/999999/suggestions").status_code == 404
    assert client.post("/api/projects/999999/autolabel", json={}).status_code == 404


def test_vlm_judge_stores_verdicts_and_reuses_cache(client, make_image, tmp_path,
                                                    monkeypatch):
    """문맥 심판: rubric 저장 → 박스별 판정이 meta.vlm에 남는다.

    같은 기준 재실행은 캐시를 써서 판정 함수를 다시 부르지 않아야 한다 —
    VLM 호출은 건당 비용이라 재실행이 공짜여야 안심하고 다시 돌린다.
    """
    import time

    from server import vlm

    pid = _project(client, "vlm")
    img = make_image(tmp_path / "vlm", "a.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("a.jpg", f, "image/jpeg"))]).json()["saved"][0]
    client.put(f"/api/images/{iid}/annotations", json={"annotations": [
        {"class_name": "person", "bbox": [10, 10, 50, 50], "source": "model"},
        {"class_name": "person", "bbox": [100, 100, 50, 50], "source": "model"}]})
    client.put(f"/api/images/{iid}/status", json={"status": "prelabeled"})

    # 기준 없이 실행하면 400 — 기준 없는 판정은 의미가 없다
    assert client.post(f"/api/projects/{pid}/vlm-judge", json={}).status_code == 400
    assert client.put(f"/api/projects/{pid}/rubric",
                      json={"rubric": "정면을 보는 사람만 person"}).json()["ok"]
    assert client.get(f"/api/projects/{pid}").json()["rubric"].startswith("정면")

    calls = []
    behavior = {"fail_transiently": True}

    def fake_judge(image, bbox, class_name, rubric, prov):
        calls.append(list(bbox))
        if behavior["fail_transiently"] and bbox[0] == 10:
            # 일시 장애 흉내 (429 등) — 실제 judge_box와 같은 형태로 반환
            return {"verdict": "unsure", "reason": "판정 실패: 429", "error": True}
        return {"verdict": "fail" if bbox[0] == 10 else "pass", "reason": "테스트 근거"}

    monkeypatch.setattr(vlm, "judge_box", fake_judge)
    monkeypatch.setattr(vlm, "provider", lambda: "anthropic")

    def run_and_wait():
        r = client.post(f"/api/projects/{pid}/vlm-judge", json={})
        assert r.status_code == 200, r.text
        for _ in range(50):
            s = client.get(f"/api/projects/{pid}/vlm-judge/status").json()
            if s["status"] != "running":
                return s
            time.sleep(0.1)
        raise AssertionError("판정이 끝나지 않음")

    s = run_and_wait()
    assert s["status"] == "completed", s
    assert s["pass"] == 1 and s["unsure"] == 1 and len(calls) == 2
    # 박스 단위 진행률 — 박스 많은 이미지에서 이미지 단위 진행만 보이면
    # 수십 분째 멈춘 것처럼 보인다 (실측)
    assert s["total_boxes"] == 2 and s["done_boxes"] == 2, s

    anns = client.get(f"/api/images/{iid}/annotations").json()
    assert all(a["meta"]["vlm"]["rubric_sha"] for a in anns)

    # 일시 장애로 실패한 판정은 캐시되면 안 된다 — 재실행 시 그 박스만 재판정.
    # 이게 없으면 429 한 번에 영구 unsure가 되고, 기준을 바꿔 전량 재과금하는
    # 것 말고는 탈출구가 없다.
    behavior["fail_transiently"] = False
    s = run_and_wait()
    assert s["status"] == "completed", s
    assert len(calls) == 3, "오류 판정이 캐시로 굳었다"
    assert s["fail"] == 1 and s["cached"] == 1

    # 정상 판정은 전량 캐시 — 재실행 비용 0
    s = run_and_wait()
    assert s["cached"] == 2 and len(calls) == 3, "캐시가 있는데 VLM을 다시 불렀다"

    # 박스를 수정하면 그 박스만 재판정 — 낡은 판정이 유효한 척하면 안 된다
    anns = client.get(f"/api/images/{iid}/annotations").json()
    for a in anns:
        if a["bbox"][0] == 10:
            a["bbox"] = [12, 10, 50, 50]
    client.put(f"/api/images/{iid}/annotations", json={"annotations": anns})
    s = run_and_wait()
    assert len(calls) == 4 and s["cached"] == 1, (calls, s)

    # 저장(전체 교체)이 판정을 지우면 안 된다 — 클라이언트 사본에 vlm이 없어도
    # id 기준으로 보존 병합된다 (유료 판정·캐시 보호)
    anns = client.get(f"/api/images/{iid}/annotations").json()
    stripped = [{"id": a["id"], "class_name": a["class_name"], "bbox": a["bbox"],
                 "source": a["source"]} for a in anns]
    client.put(f"/api/images/{iid}/annotations", json={"annotations": stripped})
    anns = client.get(f"/api/images/{iid}/annotations").json()
    assert all((a["meta"] or {}).get("vlm") for a in anns), "저장이 판정을 지웠다"
    s = run_and_wait()
    assert len(calls) == 4 and s["cached"] == 2, "보존된 판정이 캐시로 안 잡힌다"

    # 기준이 바뀌면 전량 다시 판정한다
    client.put(f"/api/projects/{pid}/rubric", json={"rubric": "완전히 다른 기준"})
    run_and_wait()
    assert len(calls) == 6


def test_annotation_save_keeps_ids_and_background_vlm(client, make_image, tmp_path):
    """여러 이미지가 있어도 반복 저장이 행 id와 백그라운드 VLM 판정을 지킨다.

    DELETE+INSERT 방식은 다른 이미지가 더 높은 rowid를 가진 순간 저장한 행의
    id가 바뀐다. 화면은 이전 id를 계속 보내므로 첫 저장에서 병합된 VLM 판정이
    두 번째 저장에서 사라졌다.
    """
    pid = _project(client, "stable-ann-id")
    ids = []
    for i in range(2):
        img = make_image(tmp_path / "stable", f"{i}.jpg")
        with open(img, "rb") as f:
            ids.append(client.post(
                f"/api/projects/{pid}/images",
                files=[("files", (f"{i}.jpg", f, "image/jpeg"))]).json()["saved"][0])

    first = client.put(f"/api/images/{ids[0]}/annotations", json={"annotations": [{
        "class_name": "person", "bbox": [10, 10, 30, 30], "source": "model",
        "meta": {"vlm": {"verdict": "pass", "reason": "ok", "rubric_sha": "abc",
                         "box": [[10, 10, 30, 30], "person"]}},
    }]}).json()["annotations"][0]
    # 다른 이미지의 행이 더 높은 rowid를 소유하도록 만든다.
    client.put(f"/api/images/{ids[1]}/annotations", json={"annotations": [{
        "class_name": "person", "bbox": [1, 1, 10, 10], "source": "model"}]})

    stale_client_copy = {
        "id": first["id"], "class_name": "person",
        "bbox": [10, 10, 30, 30], "source": "model",
    }
    for _ in range(2):
        saved = client.put(f"/api/images/{ids[0]}/annotations",
                           json={"annotations": [stale_client_copy]}).json()["annotations"][0]
        assert saved["id"] == first["id"]
        assert saved["meta"]["vlm"]["verdict"] == "pass"

    # 박스를 고치면 이전 박스를 판정한 VLM 결과는 유효하지 않다.
    stale_client_copy["bbox"] = [12, 10, 30, 30]
    changed = client.put(f"/api/images/{ids[0]}/annotations",
                         json={"annotations": [stale_client_copy]}).json()["annotations"][0]
    assert "vlm" not in changed["meta"]


def test_vlm_claude_code_adapter_parses_headless_output(monkeypatch, tmp_path):
    """Claude Code 헤드리스 어댑터 — 구독으로 호출하는 경로.

    `claude -p --output-format json`의 result 텍스트에서 verdict JSON을 뽑는다.
    모델이 JSON 앞뒤에 말을 붙여도 견뎌야 한다.
    """
    import glob
    import subprocess
    import tempfile

    from server import vlm

    # 실행 전 스냅샷 — 같은 머신에서 실제 심판이 돌고 있으면 그 임시 파일이
    # 보일 수 있다. 이 테스트가 만든 파일만 검사해야 한다
    before = set(glob.glob(f"{tempfile.gettempdir()}/vlm_judge_*.png"))

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd

        class R:
            returncode = 0
            stderr = ""
            stdout = json.dumps({"type": "result", "result":
                                 '판정 결과입니다: {"verdict": "fail", "reason": "기준 위반"} 이상.'})
        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = vlm._judge_claude_code(b"\x89PNG fake", "판정하라")
    assert out == {"verdict": "fail", "reason": "기준 위반"}
    assert captured["cmd"][0] == "claude" and "--output-format" in captured["cmd"]
    # 임시 crop 파일은 정리되어야 한다 (이 테스트가 새로 만든 것 기준)
    after = set(glob.glob(f"{tempfile.gettempdir()}/vlm_judge_*.png"))
    assert not (after - before)


def test_vlm_provider_prefers_subscription_over_local(monkeypatch):
    """제공자 자동 감지 순서: API 키 > Claude Code(구독, 무료) > Ollama."""
    from server import vlm

    monkeypatch.delenv("AUTOLABEL_VLM", raising=False)
    monkeypatch.setattr(vlm, "_anthropic_ready", lambda: False)
    monkeypatch.setattr(vlm, "_claude_code_ready", lambda: True)
    monkeypatch.setattr(vlm, "_ollama_ready", lambda: True)
    assert vlm.provider() == "claude-code"

    monkeypatch.setattr(vlm, "_anthropic_ready", lambda: True)
    assert vlm.provider() == "anthropic"

    monkeypatch.setenv("AUTOLABEL_VLM", "ollama")
    assert vlm.provider() == "ollama"
    monkeypatch.setenv("AUTOLABEL_VLM", "off")
    assert vlm.provider() is None


def test_vlm_judge_without_provider_returns_clear_error(client, make_image, tmp_path,
                                                        monkeypatch):
    """제공자가 없으면 조용히 실패하지 말고 설정 방법을 알려야 한다."""
    from server import vlm

    monkeypatch.setattr(vlm, "provider", lambda: None)
    pid = _project(client, "vlm-off")
    client.put(f"/api/projects/{pid}/rubric", json={"rubric": "기준"})
    img = make_image(tmp_path / "voff", "a.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("a.jpg", f, "image/jpeg"))]).json()["saved"][0]
    client.put(f"/api/images/{iid}/status", json={"status": "prelabeled"})

    r = client.post(f"/api/projects/{pid}/vlm-judge", json={})
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]

    # capabilities가 제공자 유무를 알린다 (UI 안내용)
    cap = client.get("/api/capabilities").json()
    assert "vlm" in cap and "vlm_hint" in cap


def test_capabilities_reports_model_availability(client):
    cap = client.get("/api/capabilities").json()
    assert set(cap) >= {"sam3", "sam_encoder", "device"}
    assert isinstance(cap["sam3"], bool)


def test_colab_notebook_is_valid_json(client):
    import ast

    pid = _project(client, "colab")
    r = client.get(f"/api/projects/{pid}/colab-notebook")
    nb = json.loads(r.content)
    assert nb["nbformat"] == 4 and nb["metadata"]["accelerator"] == "GPU"
    assert any("ultralytics" in "".join(c["source"]) for c in nb["cells"])
    source = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "승인 학습 데이터.zip" in source
    assert "autolabel-model.json" in source and "autolabel-model.zip" in source
    assert "val_map50" in source and "test_map50" in source
    assert "epochs=60" in source  # 작은 데이터 UI 추천값과 노트북 기본값이 같아야 한다
    assert "d.setdefault('train', 'images')" not in source
    bundle_cell = next(c for c in nb["cells"] if "autolabel-model.json" in "".join(c["source"]))
    ast.parse("".join(bundle_cell["source"]))

    assert client.get(f"/api/projects/{pid}/colab-notebook?arch=bad;rm&epochs=100").status_code == 400
    assert client.get(f"/api/projects/{pid}/colab-notebook?epochs=0").status_code == 400


def test_training_dataset_uses_only_approved_fixed_splits(client, make_image, tmp_path):
    """Colab 패키지에는 검수 전 초안이 들어가면 안 되고 train/val은 달라야 한다."""
    from server.db import get_db

    pid = _project(client, "approved-training")
    approved_ids = []
    for i in range(12):
        img = make_image(tmp_path / "approved-training", f"approved-{i}.jpg")
        with open(img, "rb") as f:
            iid = client.post(f"/api/projects/{pid}/images",
                              files=[("files", (img.name, f, "image/jpeg"))]).json()["saved"][0]
        approved_ids.append(iid)
        client.put(f"/api/images/{iid}/annotations", json={"annotations": [{
            "class_name": "person", "bbox": [10, 20, 100, 80], "source": "human",
        }]})
    draft = make_image(tmp_path / "approved-training", "review-pending.jpg")
    with open(draft, "rb") as f:
        draft_id = client.post(f"/api/projects/{pid}/images",
                               files=[("files", (draft.name, f, "image/jpeg"))]).json()["saved"][0]
    client.put(f"/api/images/{draft_id}/annotations", json={"annotations": [{
        "class_name": "person", "bbox": [1, 2, 3, 4], "source": "model",
    }]})
    conn = get_db()
    conn.executemany("UPDATE images SET status='approved' WHERE id=?",
                     [(iid,) for iid in approved_ids])
    conn.execute("UPDATE images SET status='prelabeled' WHERE id=?", (draft_id,))
    conn.commit()
    conn.close()

    r = client.get(f"/api/projects/{pid}/training-dataset.zip")
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))
    members = set(z.namelist())
    yaml = z.read("data.yaml").decode()
    assert "path: ." in yaml
    assert "train: images/train" in yaml
    assert "val: images/val" in yaml
    assert "test: images/test" in yaml
    image_members = {n for n in members if n.startswith("images/")}
    assert len(image_members) == len(approved_ids)
    assert not any("review-pending" in n for n in members)
    train_names = {Path(n).name for n in image_members if n.startswith("images/train/")}
    val_names = {Path(n).name for n in image_members if n.startswith("images/val/")}
    assert train_names and val_names and train_names.isdisjoint(val_names)

    from server import train
    exported_count, _ = train._export_yolo_dataset(pid, tmp_path / "training-export-direct")
    assert exported_count == len(approved_ids)  # test 홀드아웃도 재학습 기준 장수에 포함


def test_images_list_reports_vlm_flags_and_project_progress(client, make_image, tmp_path):
    """심판 위반 필터의 근거 — 이미지 목록에 위반·불확실 박스 수(vlm_flags)가
    나와야 위반이 어느 이미지에 있는지 찾는다 (실측: 뱃지·필터가 없어 15장을
    일일이 뒤졌다). 프로젝트 목록의 승인 진행도(approved_count)도 같은 이유."""
    pid = _project(client, "flags")
    img = make_image(tmp_path / "flags", "a.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("a.jpg", f, "image/jpeg"))]).json()["saved"][0]
    client.put(f"/api/images/{iid}/annotations", json={"annotations": [
        {"class_name": "person", "bbox": [10, 10, 50, 50], "source": "model",
         "meta": {"vlm": {"verdict": "fail", "reason": "x"}}},
        {"class_name": "person", "bbox": [70, 10, 50, 50], "source": "model",
         "meta": {"vlm": {"verdict": "pass", "reason": "y"}}},
        {"class_name": "person", "bbox": [130, 10, 50, 50], "source": "model",
         "meta": {"vlm": {"verdict": "unsure", "reason": "z"}}}]})
    me = next(i for i in client.get(f"/api/projects/{pid}/images").json() if i["id"] == iid)
    assert me["vlm_flags"] == 2  # fail + unsure — pass는 세지 않는다

    client.put(f"/api/images/{iid}/status", json={"status": "approved"})
    p = next(p for p in client.get("/api/projects").json() if p["id"] == pid)
    assert p["image_count"] == 1 and p["approved_count"] == 1


def test_batch_verdict_partial_coverage_is_not_condemned():
    """중간 검출률을 '제로샷 약함'으로 단정하지 않는다 — 대상 없는 이미지가
    섞인 데이터셋에선 낮은 커버리지가 정답이다 (실측: 개 9장+비개 6장에서
    9/15 검출 = 만점인데 'weak' 판정으로 프롬프트 실험·수동 라벨을 권했다)."""
    from server.main import _batch_verdict

    v = _batch_verdict({"hit": 9, "found": 11}, 15)
    assert v["verdict"] == "partial"
    assert "대상이 없다면 정상" in v["advice"]
    assert _batch_verdict({"hit": 15, "found": 20}, 15)["verdict"] == "good"
    assert _batch_verdict({"hit": 0, "found": 0}, 15)["verdict"] == "empty"


def test_batch_verdict_flags_class_holes_and_full_frame_boxes():
    """높은 검출률이 곧 정확도라는 오판을 막는다."""
    from server.main import _batch_verdict

    v = _batch_verdict({
        "hit": 22, "found": 55, "large_boxes": 10,
        "class_counts": {"pitted_surface": 28, "inclusion": 13,
                         "patches": 10, "scratches": 4},
    }, 30, ["crazing", "inclusion", "patches", "pitted_surface",
            "rolled-in_scale", "scratches"])

    assert v["verdict"] == "partial"
    assert v["missing_classes"] == ["crazing", "rolled-in_scale"]
    assert v["large_boxes"] == 10


def test_batch_verdict_prioritizes_low_agreement_candidates():
    from server.main import _batch_verdict

    v = _batch_verdict({
        "hit": 10, "found": 12, "class_counts": {"defect": 12},
        "agreement_counts": {"consensus": 2, "sam3_only": 6, "gdino_only": 4},
        "engine_plan": {"mode": "sam3", "both_engine_images": 4,
                        "seeded_before": 0, "seed_target": 30, "explore_every": 10},
    }, 10, ["defect"])
    assert v["verdict"] == "partial"
    assert "합의가 2/12개로 낮음" in v["advice"]
    # 왜 10장 중 4장만 두 엔진을 돌렸는지 숨기지 않고 예산으로 설명한다
    assert "10장 중 4장을 두 엔진으로 교차 검출" in v["advice"]
    assert v["agreement_counts"]["gdino_only"] == 4
    assert "검출률만으로" in v["advice"]


def test_import_without_images_dir_is_400_not_500(client):
    pid = _project(client, "imp400")
    r = client.post(f"/api/projects/{pid}/import", json={"image_dir": "/tmp/oops"})
    assert r.status_code == 400
    assert "images_dir" in r.json()["detail"]


def _vlm_setup(client, make_image, tmp_path, name, boxes):
    pid = _project(client, name)
    img = make_image(tmp_path / name, "a.jpg")
    with open(img, "rb") as f:
        iid = client.post(f"/api/projects/{pid}/images",
                          files=[("files", ("a.jpg", f, "image/jpeg"))]).json()["saved"][0]
    client.put(f"/api/images/{iid}/annotations", json={"annotations": [
        {"class_name": "person", "bbox": b, "source": "model"} for b in boxes]})
    client.put(f"/api/images/{iid}/status", json={"status": "prelabeled"})
    client.put(f"/api/projects/{pid}/rubric", json={"rubric": "사람만 부합"})
    return pid, iid


def _wait_job(client, status_url, tries=100):
    """백그라운드 잡 폴링 — running이 아닌 첫 상태를 반환."""
    import time
    for _ in range(tries):
        s = client.get(status_url).json()
        if s["status"] != "running":
            return s
        time.sleep(0.1)
    raise AssertionError(f"잡이 끝나지 않음: {status_url}")


def _vlm_run(client, pid):
    assert client.post(f"/api/projects/{pid}/vlm-judge", json={}).status_code == 200
    return _wait_job(client, f"/api/projects/{pid}/vlm-judge/status")


def test_vlm_judge_skips_tiny_boxes_without_calling_vlm(client, make_image, tmp_path,
                                                        monkeypatch):
    """한 변 24px 미만 박스는 VLM 호출 없이 불확실로 확정 — VLM도 저해상도
    crop엔 '식별 불가'만 답한다 (실측: kitchen 125박스 중 unsure 49건 대부분이
    초소형 컵). 스킵 판정도 캐시에 남아 재실행 때 다시 안 묻는다."""
    from server import vlm

    pid, iid = _vlm_setup(client, make_image, tmp_path, "tiny",
                          [[10, 10, 8, 8], [30, 30, 50, 50]])
    calls = []

    def fake(image, bbox, class_name, rubric, prov):
        calls.append(list(bbox))
        return {"verdict": "pass", "reason": "ok"}

    monkeypatch.setattr(vlm, "judge_box", fake)
    monkeypatch.setattr(vlm, "provider", lambda: "anthropic")

    s = _vlm_run(client, pid)
    assert s["status"] == "completed", s
    assert calls == [[30, 30, 50, 50]], "큰 박스만 VLM에 물어야 한다"
    tiny = next(a for a in client.get(f"/api/images/{iid}/annotations").json()
                if a["bbox"] == [10, 10, 8, 8])
    assert tiny["meta"]["vlm"]["verdict"] == "unsure"
    assert tiny["meta"]["vlm"]["skipped"] == "tiny"

    s = _vlm_run(client, pid)  # 재실행 — 스킵 판정도 캐시여야 한다
    assert s["cached"] == 2 and len(calls) == 1, s


def test_vlm_judge_runs_boxes_in_parallel(client, make_image, tmp_path, monkeypatch):
    """박스 판정은 병렬 — claude CLI 기동이 박스당 수 초~수십 초라 순차로는
    수백 박스에 수 시간이 든다. DB 쓰기는 메인 스레드 단일 레인 유지."""
    import threading
    import time

    from server import vlm

    monkeypatch.setenv("AUTOLABEL_VLM_WORKERS", "3")
    pid, iid = _vlm_setup(client, make_image, tmp_path, "parallel",
                          [[i * 60, 30, 50, 50] for i in range(4)])
    lock = threading.Lock()
    active = {"now": 0, "max": 0}

    def fake(image, bbox, class_name, rubric, prov):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.15)
        with lock:
            active["now"] -= 1
        return {"verdict": "pass", "reason": "ok"}

    monkeypatch.setattr(vlm, "judge_box", fake)
    monkeypatch.setattr(vlm, "provider", lambda: "anthropic")

    s = _vlm_run(client, pid)
    assert s["status"] == "completed" and s["pass"] == 4, s
    assert active["max"] >= 2, f"병렬 실행이 안 됨 (동시 최대 {active['max']})"


def test_video_upload_extracts_frames_and_reports_honestly(client, tmp_path,
                                                           monkeypatch):
    """비디오 레인: 프레임이 stride 간격으로 일반 이미지로 등록된다.

    모델 없는 모드에서는 트래킹을 건너뛰되 그 사실을 정직하게 알려야 한다 —
    조용히 0박스면 사용자는 트래킹이 실패한 줄 모른다.
    """
    import time

    import cv2
    import numpy as np

    # 개발 머신엔 models/sam3.pt가 있어 실제 트래킹이 돌아버린다 — 테스트는
    # 추출·등록·정직 보고만 검증한다
    monkeypatch.setenv("AUTOLABEL_NO_MODELS", "1")

    vid = tmp_path / "v.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert wr.isOpened(), "mp4v 인코더 사용 불가"
    for i in range(20):
        f = np.zeros((48, 64, 3), np.uint8)
        f[10:30, i * 2:i * 2 + 16] = 255
        wr.write(f)
    wr.release()

    pid = _project(client, "video")
    with open(vid, "rb") as f:
        r = client.post(f"/api/projects/{pid}/video?stride=5&max_frames=3",
                        files=[("file", ("v.mp4", f, "video/mp4"))])
    assert r.status_code == 200, r.text

    s = _wait_job(client, f"/api/projects/{pid}/video/status")
    assert s["status"] == "completed", s
    assert "트래킹 생략" in s["advice"]

    imgs = client.get(f"/api/projects/{pid}/images").json()
    assert len(imgs) == 3  # 프레임 0, 5, 10 (max_frames=3)
    assert [i["file_name"] for i in imgs] == ["v_f000000.jpg", "v_f000005.jpg", "v_f000010.jpg"]
    assert all(i["width"] == 64 and i["height"] == 48 for i in imgs)
    # 프레임 파일이 실제로 서빙된다
    assert client.get(f"/api/images/{imgs[0]['id']}/file").status_code == 200

    # 클래스 없는 프로젝트는 400 — 트래킹 프롬프트가 없다
    empty = client.post("/api/projects", json={"name": "novid", "ontology": []}).json()["id"]
    with open(vid, "rb") as f:
        r = client.post(f"/api/projects/{empty}/video",
                        files=[("file", ("v.mp4", f, "video/mp4"))])
    assert r.status_code == 400

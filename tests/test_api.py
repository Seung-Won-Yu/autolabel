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


def test_project_crud_and_ontology(client):
    pid = _project(client)
    assert any(p["id"] == pid for p in client.get("/api/projects").json())

    onto = [{"name": "cat", "prompt": "cat", "threshold": 0.4}]
    assert client.put(f"/api/projects/{pid}/ontology", json={"ontology": onto}).json()["ok"]
    assert client.get(f"/api/projects/{pid}").json()["ontology"] == onto

    assert client.delete(f"/api/projects/{pid}").json()["ok"]
    assert not any(p["id"] == pid for p in client.get("/api/projects").json())


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
    """고신뢰만 승인하고 저신뢰는 남겨야 한다."""
    pid = _project(client, "approve")
    ids = []
    for i, conf in enumerate([0.9, 0.3]):
        img = make_image(tmp_path / f"ap{i}", f"{i}.jpg")
        with open(img, "rb") as f:
            iid = client.post(f"/api/projects/{pid}/images",
                              files=[("files", (f"{i}.jpg", f, "image/jpeg"))]).json()["saved"][0]
        client.put(f"/api/images/{iid}/annotations", json={"annotations": [
            {"class_name": "person", "bbox": [1, 1, 10, 10],
             "confidence": conf, "source": "model"}]})
        client.put(f"/api/images/{iid}/status", json={"status": "prelabeled"})
        ids.append(iid)

    dry = client.post(f"/api/projects/{pid}/auto-approve",
                      json={"min_conf": 0.7, "dry_run": True}).json()
    assert dry["approved"] == 1 and dry["skipped_low_confidence"] == 1

    client.post(f"/api/projects/{pid}/auto-approve", json={"min_conf": 0.7})
    states = {r["id"]: r["status"] for r in client.get(f"/api/projects/{pid}/images").json()}
    assert states[ids[0]] == "approved" and states[ids[1]] == "prelabeled"


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
        "max_defects": plan["max_defects"], "apply": True}).json()
    assert res["accepted"] and res["approved_images"] == 40


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


def test_model_import_and_rollback(client, tmp_path):
    """외부 .pt 등록 → 두 번째 등록으로 교체 → 첫 모델로 롤백."""
    pid = _project(client, "models")
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"fake-weights-a")
    b.write_bytes(b"fake-weights-b")

    r1 = client.post(f"/api/projects/{pid}/models/import", json={
        "path": str(a), "names": ["person"], "map50": 0.5}).json()
    assert client.get(f"/api/projects/{pid}/train/status").json()["active_model"]["id"] == r1["id"]

    # 두 번째 등록이 챔피언을 교체해야 한다 (활성은 항상 하나)
    r2 = client.post(f"/api/projects/{pid}/models/import", json={
        "path": str(b), "names": ["person"], "map50": 0.7}).json()
    models = client.get(f"/api/projects/{pid}/models").json()
    assert [m["id"] for m in models if m["active"]] == [r2["id"]], models

    # 롤백
    assert client.post(f"/api/projects/{pid}/models/{r1['id']}/activate").json()["ok"]
    assert client.get(f"/api/projects/{pid}/train/status").json()["active_model"]["id"] == r1["id"]


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
    before = client.get(f"/api/projects/{pid}/train/status").json()["active_model"]

    assert client.post(f"/api/projects/{pid}/models/999999/activate").status_code == 404
    after = client.get(f"/api/projects/{pid}/train/status").json()["active_model"]
    assert after and after["id"] == before["id"], "실패한 롤백이 활성 모델을 날렸다"

    # 남의 프로젝트 모델도 마찬가지로 거부
    other = _project(client, "rollback-other")
    assert client.post(f"/api/projects/{other}/models/{before['id']}/activate").status_code == 404


def test_capabilities_reports_model_availability(client):
    cap = client.get("/api/capabilities").json()
    assert set(cap) >= {"sam3", "sam_encoder", "device"}
    assert isinstance(cap["sam3"], bool)


def test_colab_notebook_is_valid_json(client):
    pid = _project(client, "colab")
    r = client.get(f"/api/projects/{pid}/colab-notebook")
    nb = json.loads(r.content)
    assert nb["nbformat"] == 4 and nb["metadata"]["accelerator"] == "GPU"
    assert any("ultralytics" in "".join(c["source"]) for c in nb["cells"])

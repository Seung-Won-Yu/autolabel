"""API 종단 회귀 테스트 — 실제로 깨졌던 경로를 재현해 막는다."""
import json
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

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

    def fake_run(pid_, image_ids, ontology, masks):
        seen["ids"] = image_ids
        m._jobs[pid_].update(status="completed")

    monkeypatch.setattr(m, "_run_batch", fake_run)
    r = client.post(f"/api/projects/{pid}/autolabel", json={}).json()
    assert r["total"] == 2, r
    for _ in range(40):
        if "ids" in seen:
            break
        time.sleep(0.05)
    assert set(seen["ids"]) == {ids["prelabeled"], ids["unlabeled"]}, seen

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


def test_vlm_claude_code_adapter_parses_headless_output(monkeypatch, tmp_path):
    """Claude Code 헤드리스 어댑터 — 구독으로 호출하는 경로.

    `claude -p --output-format json`의 result 텍스트에서 verdict JSON을 뽑는다.
    모델이 JSON 앞뒤에 말을 붙여도 견뎌야 한다.
    """
    import subprocess

    from server import vlm

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
    # 임시 crop 파일은 정리되어야 한다
    import glob
    import tempfile
    assert not glob.glob(f"{tempfile.gettempdir()}/vlm_judge_*.png")


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
    pid = _project(client, "colab")
    r = client.get(f"/api/projects/{pid}/colab-notebook")
    nb = json.loads(r.content)
    assert nb["nbformat"] == 4 and nb["metadata"]["accelerator"] == "GPU"
    assert any("ultralytics" in "".join(c["source"]) for c in nb["cells"])


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

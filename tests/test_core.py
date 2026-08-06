"""핵심 로직 회귀 테스트 — 모델 추론 없이 도는 빠른 스위트.

여기 있는 케이스들은 전부 실제로 한 번씩 깨졌던 것들이다.
"""
import json
import os

import pytest

from server import sampling
from server.ensemble import agreement_counts, batch_engine, fuse_foundation_detections
from server.parts import parse_ontology
from server.tiling import merge_nms, suppress_trusted_overlaps, tile_boxes
from server.train_worker import epoch_progress, pick_arch, pick_imgsz


# ---------- acceptance sampling ----------

def test_sampling_plan_saves_inspection():
    """큰 배치일수록 검수 절감이 커야 한다."""
    small = sampling.plan(50)
    big = sampling.plan(1000)
    assert big["sample_size"] < big["lot_size"]
    assert big["saving"] > small["saving"]
    assert big["max_defects"] >= 0


def test_sampling_plan_small_lot_is_honest():
    """표본이 로트를 넘을 수 없다 — 작은 배치는 전수 검사라고 정직하게 말해야."""
    p = sampling.plan(10)
    assert p["sample_size"] <= 10
    assert p["saving"] == 0.0


def test_sampling_verdict_boundary():
    """허용치 경계에서 승인/반려가 뒤집혀야 한다."""
    ok = sampling.verdict(59, 0, 0, 0.05, 0.95)
    ng = sampling.verdict(59, 1, 0, 0.05, 0.95)
    assert ok["accepted"] and not ng["accepted"]


def test_sampling_pick_is_deterministic_and_sized():
    ids = list(range(100))
    a = sampling.pick_sample(ids, 10, seed=1)
    b = sampling.pick_sample(ids, 10, seed=1)
    assert a == b and len(a) == 10 and len(set(a)) == 10


# ---------- 학습 설정 자동 결정 ----------

def test_pick_arch_is_conservative_for_small_data():
    """실측상 소량에서 큰 모델은 손해 — 보수적으로 유지되어야 한다."""
    assert pick_arch(50) == "yolo11n"
    assert pick_arch(300) == "yolo11n"
    assert pick_arch(5000) in ("yolo11s", "yolo11m")
    assert pick_arch(50, override="yolo11m") == "yolo11m"


def test_epoch_progress_uses_real_elapsed_time_for_eta():
    """첫 epoch 실측 속도로 ETA를 만들고 마지막에는 정확히 0이 된다."""
    first = epoch_progress(0, 10, started_at=100, now=110)
    assert first == {
        "epoch": 1, "epochs": 10, "progress": 0.1,
        "elapsed_sec": 10, "eta_sec": 90,
    }
    middle = epoch_progress(4, 10, started_at=100, now=150)
    assert middle["progress"] == 0.5 and middle["eta_sec"] == 50
    assert epoch_progress(9, 10, started_at=100, now=220)["eta_sec"] == 0


def test_pick_imgsz_keeps_default_for_small_data(tmp_path):
    """해상도 상향은 실측에서 소량 데이터에 일관되게 손해였다."""
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    assert pick_imgsz(tmp_path, n_images=100) == 640
    assert pick_imgsz(tmp_path, n_images=5000) == 1280


def test_champion_gate_requires_same_val_comparison():
    from server.train_worker import should_promote

    assert should_promote(0.6, None, champion_required=False, champion_eval_ok=True)
    assert not should_promote(0.2, None, champion_required=False, champion_eval_ok=True)
    assert should_promote(0.61, 0.60, champion_required=True, champion_eval_ok=True)
    assert not should_promote(0.59, 0.60, champion_required=True, champion_eval_ok=True)
    # champion 평가가 실패했는데 과거 저장 점수로 대신 비교하면 시험지가 달라진다.
    assert not should_promote(0.9, None, champion_required=True, champion_eval_ok=False)


def test_operational_calibration_and_checkpoint_choice():
    from types import SimpleNamespace

    from server.train_worker import choose_deploy_checkpoint, operational_calibration

    metrics = SimpleNamespace(box=SimpleNamespace(
        f1_curve=[[0.1, 0.8, 0.5], [0.2, 0.3, 0.9]],
        px=[0.01, 0.2, 0.6], ap_class_index=[0, 1]))
    thresholds, f1 = operational_calibration(metrics, ["a", "b"])
    assert thresholds == {"a": 0.2, "b": 0.6}
    assert f1 == pytest.approx(0.85)
    assert choose_deploy_checkpoint(0.4, 0.8) == "last"
    assert choose_deploy_checkpoint(0.8, 0.79) == "best"


def test_sam_embedding_cache_is_bounded_lru(monkeypatch):
    """이미지를 넘겨 볼수록 SAM 임베딩이 무한히 RAM에 쌓이면 안 된다."""
    from server import ml

    monkeypatch.setattr(ml, "EMBED_CACHE_MAX", 2)
    ml._embed_cache.clear()
    ml._embed_cache_put("a", {"key": "a"})
    ml._embed_cache_put("b", {"key": "b"})
    assert ml._embed_cache_get("a")["key"] == "a"  # a가 최근 사용으로 승급
    ml._embed_cache_put("c", {"key": "c"})
    assert list(ml._embed_cache) == ["a", "c"]  # 가장 오래된 b 퇴출
    ml._embed_cache.clear()


# ---------- 계층 온톨로지 ----------

def test_parse_ontology_splits_parent_and_parts():
    onto = [
        {"name": "person"}, {"name": "person.head"}, {"name": "person.left_arm"},
        {"name": "car"},
    ]
    parents, parts = parse_ontology(onto)
    assert {p["name"] for p in parents} == {"person", "car"}
    assert [p["child"] for p in parts["person"]] == ["head", "left_arm"]
    assert "car" not in parts


# ---------- 타일링 ----------

def test_tile_boxes_cover_image_with_overlap():
    boxes = tile_boxes(2000, 1000, tile=800, overlap=0.2)
    assert boxes, "타일이 생성되어야 한다"
    # 우/하단 끝까지 커버되는지
    assert max(x + w for x, y, w, h in boxes) >= 2000
    assert max(y + h for x, y, w, h in boxes) >= 1000


def test_merge_nms_dedupes_same_class_overlap():
    dets = [
        {"class_name": "a", "bbox": [0, 0, 100, 100], "confidence": 0.9},
        {"class_name": "a", "bbox": [5, 5, 100, 100], "confidence": 0.7},   # 중복
        {"class_name": "b", "bbox": [0, 0, 100, 100], "confidence": 0.8},   # 다른 클래스
    ]
    out = merge_nms(dets)
    assert len(out) == 2
    assert out[0]["confidence"] == 0.9  # 높은 신뢰도가 살아남음


def test_human_overlap_suppression_keeps_other_classes():
    trusted = [{"class_name": "object", "bbox": [0, 0, 100, 100]}]
    dets = [
        {"class_name": "object", "bbox": [10, 10, 20, 20]},  # 포함된 중복
        {"class_name": "object", "bbox": [200, 200, 20, 20]},
        {"class_name": "object.part", "bbox": [10, 10, 20, 20]},
    ]
    kept, suppressed = suppress_trusted_overlaps(dets, trusted)
    assert suppressed == 1
    assert [d["class_name"] for d in kept] == ["object", "object.part"]


def test_foundation_ensemble_marks_consensus_and_keeps_disagreements():
    sam3 = [
        {"class_name": "crack", "bbox": [10, 10, 20, 20], "confidence": 0.8},
        {"class_name": "dent", "bbox": [60, 60, 10, 10], "confidence": 0.7},
    ]
    gdino = [
        {"class_name": "crack", "bbox": [11, 11, 20, 20], "confidence": 0.6},
        # 위치가 겹쳐도 클래스가 다르면 같은 객체로 합치지 않는다.
        {"class_name": "scratch", "bbox": [60, 60, 10, 10], "confidence": 0.9},
    ]
    result = fuse_foundation_detections(sam3, gdino)

    assert len(result) == 3
    assert agreement_counts(result) == {"consensus": 1, "sam3_only": 1, "gdino_only": 1}
    consensus = next(d for d in result
                     if d["meta"]["ensemble"]["agreement"] == "consensus")
    assert consensus["bbox"] == sam3[0]["bbox"]
    assert consensus["confidence"] == 0.6  # 서로 다른 척도라 보수적인 낮은 값
    assert consensus["meta"]["ensemble"]["match_iou"] > 0.8
    assert consensus["meta"]["ensemble"]["sam3_bbox"] == sam3[0]["bbox"]
    assert consensus["meta"]["ensemble"]["gdino_bbox"] == gdino[0]["bbox"]
    assert result[-1] == consensus  # 불일치 후보가 검수 목록의 앞쪽


def test_foundation_ensemble_matches_each_candidate_only_once():
    sam3 = [
        {"class_name": "crack", "bbox": [10, 10, 20, 20], "confidence": 0.9},
        {"class_name": "crack", "bbox": [12, 12, 20, 20], "confidence": 0.8},
    ]
    gdino = [{"class_name": "crack", "bbox": [11, 11, 20, 20], "confidence": 0.7}]
    result = fuse_foundation_detections(sam3, gdino)
    assert agreement_counts(result) == {"consensus": 1, "sam3_only": 1, "gdino_only": 0}


def test_seed_budget_runs_both_engines_until_sample_is_built():
    # 표본이 아직 없다 — 예산만큼은 무조건 양쪽을 돌린다.
    assert [batch_engine(i, "sam3", 0, seed_images=5) for i in range(5)] == ["ensemble"] * 5
    # 이전 배치에서 3장을 채웠으면 이번 배치는 2장만 더 채우면 된다.
    assert [batch_engine(i, "sam3", 3, seed_images=5, explore_every=10)
            for i in range(4)] == ["ensemble", "ensemble", "sam3", "sam3"]


def test_exploration_keeps_both_engine_samples_growing_after_routing():
    """경로가 정해진 뒤에도 주기적으로 양쪽을 돌려야 근거가 자란다.

    이게 없으면 sam3_ran=1 AND gdino_ran=1 행이 더 안 생겨서 build_profile이
    초기 표본에 영구히 갇힌다 — 틀린 판정을 뒤집을 데이터가 사라진다.
    """
    modes = [batch_engine(i, "routed", 99, seed_images=5, explore_every=4)
             for i in range(8)]
    assert modes == ["ensemble", "routed", "routed", "routed",
                     "ensemble", "routed", "routed", "routed"]
    assert modes.count("ensemble") == 2  # 8장 중 2장 = 1/4 주기 그대로


def test_recall_profile_uses_low_candidate_threshold_and_tta(monkeypatch):
    """누락 최소화는 온톨로지 승인 기준을 바꾸지 않고 후보만 관대하게 뽑는다."""
    from PIL import Image

    from server import main
    from server import tiling

    captured = {}
    monkeypatch.setattr(main.train, "active_model", lambda _pid: {"map50": 0.5})
    monkeypatch.setattr(tiling, "should_tile", lambda _image: False)

    def fake_detect(_image, _student, ontology, augment=False):
        captured.update(ontology=ontology, augment=augment)
        return []

    monkeypatch.setattr(main.ml, "detect_student", fake_detect)
    ontology = [{"name": "defect", "threshold": 0.35}]
    _dets, engine = main._detect_auto(
        1, Image.new("RGB", (640, 640)), ontology,
        profile="recall", candidate_conf=0.10)

    assert captured["ontology"][0]["threshold"] == 0.10
    assert ontology[0]["threshold"] == 0.35  # 저장 설정은 건드리지 않는다
    assert captured["augment"] is True
    assert "recall(conf 0.1,TTA)" in engine


def test_cold_start_runs_sam3_and_gdino_as_ensemble(monkeypatch):
    from PIL import Image

    from server import main, tiling

    monkeypatch.setattr(main.train, "active_model", lambda _pid: None)
    monkeypatch.setattr(main.ml, "sam3_available", lambda: True)
    monkeypatch.setattr(tiling, "should_tile", lambda _image: False)
    monkeypatch.setattr(main.ml, "detect_sam3", lambda *_args: [
        {"class_name": "defect", "bbox": [10, 10, 20, 20], "confidence": 0.8}])
    monkeypatch.setattr(main.ml, "detect", lambda *_args: [
        {"class_name": "defect", "bbox": [11, 11, 20, 20], "confidence": 0.6}])

    dets, engine = main._detect_auto(
        1, Image.new("RGB", (640, 640)), [{"name": "defect", "threshold": 0.35}])

    assert engine == "ensemble(sam3+gdino)"
    assert agreement_counts(dets)["consensus"] == 1


def test_cold_start_keeps_gdino_when_sam3_fails(monkeypatch):
    from PIL import Image

    from server import main, tiling

    monkeypatch.setattr(main.train, "active_model", lambda _pid: None)
    monkeypatch.setattr(main.ml, "sam3_available", lambda: True)
    monkeypatch.setattr(tiling, "should_tile", lambda _image: False)
    monkeypatch.setattr(main.ml, "detect_sam3", lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(main.ml, "detect", lambda *_args: [
        {"class_name": "defect", "bbox": [10, 10, 20, 20], "confidence": 0.7}])

    dets, engine = main._detect_auto(
        1, Image.new("RGB", (640, 640)), [{"name": "defect", "threshold": 0.35}])

    assert engine == "foundation(sam3 실패)"
    assert agreement_counts(dets)["gdino_only"] == 1


def test_settled_sam3_route_skips_gdino(monkeypatch):
    from PIL import Image

    from server import main, tiling

    monkeypatch.setattr(main.ml, "sam3_available", lambda: True)
    monkeypatch.setattr(tiling, "should_tile", lambda _image: False)
    monkeypatch.setattr(main.ml, "detect_sam3", lambda *_args: [
        {"class_name": "defect", "bbox": [10, 10, 20, 20], "confidence": 0.8}])
    monkeypatch.setattr(main.ml, "detect", lambda *_args: pytest.fail("GDINO should be skipped"))

    dets, engine = main._detect_auto(
        1, Image.new("RGB", (640, 640)), [{"name": "defect"}], engine="sam3")
    assert len(dets) == 1
    assert engine == "sam3(단독)"


def test_settled_sam3_failure_falls_back_to_gdino(monkeypatch):
    from PIL import Image

    from server import foundation, tiling

    monkeypatch.setattr(foundation.ml, "sam3_available", lambda: True)
    monkeypatch.setattr(tiling, "should_tile", lambda _image: False)
    monkeypatch.setattr(
        foundation.ml, "detect_sam3",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("transient failure")))
    monkeypatch.setattr(foundation.ml, "detect", lambda *_args: [
        {"class_name": "defect", "bbox": [10, 10, 20, 20], "confidence": 0.6}])

    dets, engine = foundation.detect(
        Image.new("RGB", (640, 640)), [{"name": "defect"}], engine="sam3")

    assert len(dets) == 1
    assert dets[0]["meta"]["foundation_engine"] == "gdino"
    assert engine == "foundation(sam3 실패 보충)"


def test_reviewed_foundation_profile_selects_lower_work_engine(client):
    """삭제(FP)와 새로 그리기(FN)를 승인 정답과 비교해 클래스 경로를 고른다."""
    from server import foundation
    from server.db import get_db

    project = client.post("/api/projects", json={
        "name": "foundation-calibration",
        "ontology": [{"name": "defect", "prompt": "surface defect"}],
    }).json()
    conn = get_db()
    for i in range(3):
        iid = conn.execute(
            "INSERT INTO images (project_id,file_name,width,height,status) "
            "VALUES (?,?,?,?,?)",
            (project["id"], f"{i}.jpg", 200, 200, "approved")).lastrowid
        conn.execute(
            "INSERT INTO annotations (image_id,class_name,bbox,source) VALUES (?,?,?,?)",
            (iid, "defect", "[10,10,20,20]", "human"))
        fused = fuse_foundation_detections(
            [{"class_name": "defect", "bbox": [10, 10, 20, 20], "confidence": 0.8}],
            [{"class_name": "defect", "bbox": [100, 100, 20, 20], "confidence": 0.7}],
        )
        assert foundation.replace_audit(
            conn, project["id"], iid, fused, "ensemble(sam3+gdino)")
    conn.commit()
    profile = foundation.build_profile(conn, project["id"], project["ontology"])
    conn.close()

    assert profile["status"] == "ready"
    assert profile["reviewed_images"] == 3
    cls = profile["classes"][0]
    assert cls["selection"] == "sam3"
    assert cls["sam3"]["review_cost"] == 0
    assert cls["gdino"]["review_cost"] == 12  # FP 3 + 누락 3*3


def test_class_routes_limit_prompts_to_selected_engine(monkeypatch):
    from PIL import Image

    from server import foundation, tiling

    calls = {}
    monkeypatch.setattr(foundation.ml, "sam3_available", lambda: True)
    monkeypatch.setattr(tiling, "should_tile", lambda _image: False)

    def sam(_image, ontology):
        calls["sam3"] = [c["name"] for c in ontology]
        return []

    def gdino(_image, ontology):
        calls["gdino"] = [c["name"] for c in ontology]
        return []

    monkeypatch.setattr(foundation.ml, "detect_sam3", sam)
    monkeypatch.setattr(foundation.ml, "detect", gdino)
    _dets, used = foundation.detect(
        Image.new("RGB", (100, 100)),
        [{"name": "crack"}, {"name": "dent"}],
        engine="routed", class_routes={"crack": "sam3", "dent": "gdino"})

    assert calls == {"sam3": ["crack"], "gdino": ["dent"]}
    assert used == "routed(sam3+gdino)"


def test_routed_engine_failure_falls_back_without_dropping_selected_class(monkeypatch):
    from PIL import Image

    from server import foundation, tiling

    gdino_calls = []
    monkeypatch.setattr(foundation.ml, "sam3_available", lambda: True)
    monkeypatch.setattr(tiling, "should_tile", lambda _image: False)
    monkeypatch.setattr(
        foundation.ml, "detect_sam3",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("sam3 unavailable")))

    def gdino(_image, ontology):
        names = [item["name"] for item in ontology]
        gdino_calls.append(names)
        return [{"class_name": name, "bbox": [0, 0, 10, 10], "confidence": 0.5}
                for name in names]

    monkeypatch.setattr(foundation.ml, "detect", gdino)
    dets, used = foundation.detect(
        Image.new("RGB", (100, 100)),
        [{"name": "crack"}, {"name": "dent"}],
        engine="routed", class_routes={"crack": "sam3", "dent": "gdino"})

    assert gdino_calls == [["dent"], ["crack"]]
    assert {det["class_name"] for det in dets} == {"crack", "dent"}
    assert used == "routed(gdino)"


def test_student_detector_passes_low_threshold_into_yolo():
    """후단 필터가 0.10이어도 YOLO 기본 0.25에서 후보를 먼저 버리면 소용없다."""
    from PIL import Image

    from server import ml

    captured = {}

    class FakeModel:
        def predict(self, _image, **kwargs):
            captured.update(kwargs)
            return [type("Result", (), {"boxes": []})()]

    ml._students["fake-low-conf.pt"] = FakeModel()
    try:
        ml.detect_student(
            Image.new("RGB", (32, 32)),
            {"path": "fake-low-conf.pt", "meta": {"names": ["defect"]}},
            [{"name": "defect", "threshold": 0.10}], augment=True)
    finally:
        ml._students.pop("fake-low-conf.pt", None)

    assert captured["conf"] == 0.10
    assert captured["augment"] is True


def test_student_detector_uses_model_calibrated_threshold():
    """AP가 높아도 confidence 척도가 낮은 모델은 기본 0.3에서 0건이 될 수 있다."""
    from PIL import Image

    from server import ml

    captured = {}

    class FakeModel:
        def predict(self, _image, **kwargs):
            captured.update(kwargs)
            return [type("Result", (), {"boxes": []})()]

    ml._students["fake-calibrated.pt"] = FakeModel()
    try:
        ml.detect_student(
            Image.new("RGB", (32, 32)),
            {"path": "fake-calibrated.pt", "meta": {
                "names": ["defect"], "calibrated_thresholds": {"defect": 0.04}}},
            [{"name": "defect", "threshold": 0.30}])
    finally:
        ml._students.pop("fake-calibrated.pt", None)

    assert captured["conf"] == 0.04


def test_train_status_detects_dead_worker_after_server_restart(tmp_path, monkeypatch):
    """서버가 재시작하면 Popen 핸들이 없다 — OS pid로 워커 생사를 물어야 한다.

    예전엔 핸들이 없으면 판단을 못 해서 상태가 영원히 running으로 멈췄다.
    """
    import json as _json

    from server import train

    monkeypatch.setattr(train, "RUNS", tmp_path)
    train._procs.clear()   # 서버 재시작 흉내 — 핸들 소실

    # 존재할 수 없는 pid를 남긴 채 running
    (tmp_path / "train_status_7.json").write_text(
        _json.dumps({"status": "running", "phase": "training", "pid_os": 2 ** 22}))
    st = train.job_status(7)
    assert st["status"] == "failed"
    assert "워커" in st["error"]

    # 살아 있는 프로세스(자기 자신)라면 running을 유지해야 한다 — 워커는 별도
    # 프로세스라 서버 재시작에도 생존한다
    (tmp_path / "train_status_8.json").write_text(
        _json.dumps({"status": "running", "phase": "training", "pid_os": os.getpid()}))
    assert train.job_status(8)["status"] == "running"


def test_job_state_survives_restart_and_marks_interrupted():
    """서버가 재시작하면 인프로세스 잡은 죽는다 — 그걸 완료로 읽어선 안 된다.

    실측 사고: 상태가 메모리에만 있어서 재시작 후 프론트가 기록 없음을 완료로
    해석해 "배치 오토라벨 완료: undefined/undefined장"을 띄웠다. 절반만 라벨된
    데이터를 두고 사용자는 끝난 줄 안다.
    """
    import time as _time

    from server import jobs

    jobs.start("autolabel", 4242, done=0, total=10)
    # 진행 카운터 디스크 쓰기는 스로틀된다(WRITE_INTERVAL) — 창을 넘겨서
    # 기록되게 한다. 실사용에서 진행 갱신은 초 단위로 흩어져 있고, 크래시로
    # 잃는 것은 마지막 창 하나치 숫자뿐이다.
    _time.sleep(jobs.WRITE_INTERVAL + 0.05)
    jobs.update("autolabel", 4242, done=3)
    assert jobs.get("autolabel", 4242)["done"] == 3

    # 재시작 흉내 — 메모리 캐시를 비우면 디스크 기록만 남는다
    jobs._cache.clear()
    assert jobs.get("autolabel", 4242)["status"] == "running"

    assert jobs.sweep_stale() >= 1
    jobs._cache.clear()
    after = jobs.get("autolabel", 4242)
    assert after["status"] == "interrupted"
    assert after["done"] == 3, "어디까지 처리했는지 알려줘야 한다"
    assert "다시 실행" in after["error"]

    # 완료된 잡은 정리 대상이 아니다
    jobs.update("autolabel", 4242, status="completed")
    jobs.sweep_stale()
    jobs._cache.clear()
    assert jobs.get("autolabel", 4242)["status"] == "completed"

    # 한 번도 실행하지 않은 잡은 idle
    assert jobs.get("autolabel", 999999)["status"] == "idle"


def test_plan_splits_never_starves_train():
    """val·test 하한이 train을 고갈시키면 안 된다.

    실측 사고: 승인 8장에서 need_val=max(30,·)이 pool 전체를 val로 소진해
    train 0장으로 학습이 실패했고, 배정이 DB에 고착돼 승인을 아무리 늘려도
    초기 이미지들이 학습에서 영영 배제됐다.
    """
    from server.train import plan_splits

    plan = plan_splits(dict.fromkeys(range(8)))
    counts = {s: list(plan.values()).count(s) for s in ("train", "val", "test")}
    assert counts["train"] >= 4, counts

    # 데이터가 충분하면 기존 하한·비율 그대로 (동작 변화 없음)
    plan = plan_splits(dict.fromkeys(range(1000)))
    counts = {s: list(plan.values()).count(s) for s in ("train", "val", "test")}
    assert counts == {"train": 650, "val": 200, "test": 150}, counts

    # 한 번 정해진 소속은 유지 — 라운드 간 비교 기준
    fixed = {0: "val", 1: "test", 2: "train"}
    plan = plan_splits({**fixed, **dict.fromkeys(range(3, 100))})
    assert all(plan[i] == s for i, s in fixed.items())

    # 과거 버그로 전량이 val/test에 고착된 DB는 재배정으로 복구되어야 한다
    plan = plan_splits({i: "val" for i in range(20)})
    assert list(plan.values()).count("train") >= 10, plan


def test_plan_splits_keeps_source_groups_together():
    """같은 영상의 인접 프레임이 학습과 평가 양쪽에 섞이면 점수가 부풀려진다."""
    from server.train import plan_splits

    assigned = dict.fromkeys(range(100))
    groups = {i: f"video:{i // 5}" for i in assigned}  # 영상 20개, 프레임 5장씩
    plan = plan_splits(assigned, groups)
    for group in set(groups.values()):
        splits = {plan[i] for i, g in groups.items() if g == group}
        assert len(splits) == 1, (group, splits)
    assert set(plan.values()) == {"train", "val", "test"}

    # migration 전 이미 갈라져 있던 그룹도 한 split으로 복구한다.
    assigned = {0: "train", 1: "val", 2: "test", 3: None}
    groups = dict.fromkeys(assigned, "same-video")
    fixed = plan_splits(assigned, groups)
    assert len(set(fixed.values())) == 1


def test_worker_status_update_never_clears_pid_os(tmp_path, monkeypatch):
    """워커의 상태 병합이 런처가 남긴 pid_os를 지우면 안 된다.

    실측 사고: 워커 첫 기록이 pid_os=None을 병합해 생사 판정이 무력화됐다 —
    서버 재시작 후 살아있는 학습을 failed로 오보하고, 재시도 가드가 failed를
    통과시켜 같은 프로젝트에 워커 2개가 겹쳤다.
    """
    import json as _json

    from server import train, train_worker

    monkeypatch.setattr(train, "RUNS", tmp_path)
    monkeypatch.setattr(train_worker, "RUNS", tmp_path)
    train._procs.clear()

    (tmp_path / "train_status_9.json").write_text(_json.dumps(
        {"status": "running", "phase": "starting", "pid_os": os.getpid()}))
    # 워커 기동 시의 병합과 같은 형태 — pid_os=None이 기존 값을 지우면 안 된다
    train_worker.write_status(9, status="running", phase="export", pid_os=None)
    st = train.job_status(9)
    assert st["status"] == "running", st
    assert st["pid_os"] == os.getpid()


def test_sam_inference_is_serialized_across_threads(monkeypatch):
    """set_image→predict 구간은 원자적이어야 한다.

    배치 스레드와 /embed(FastAPI 스레드풀)가 전역 SAM predictor를 공유한다.
    끼어들면 A의 박스가 B의 특징맵으로 디코딩돼 엉뚱한 마스크가 저장되고,
    잘못된 임베딩이 이미지 해시 키로 캐시에 눌러앉는다. torch 연산이 GIL을
    놓기 때문에 잠금 없이는 실제로 인터리빙된다.
    """
    import io as _io
    import threading
    import time as _time

    import numpy as np
    from PIL import Image as PILImage

    from server import ml

    events = []

    class SlowPredictor:
        def set_image(self, arr):
            events.append(("set", threading.get_ident()))
            _time.sleep(0.02)  # GIL을 놓는 무거운 인코딩 흉내

        def get_image_embedding(self):
            events.append(("emb", threading.get_ident()))

            class T:
                def cpu(self):
                    return self

                def numpy(self):
                    return np.zeros((1, 2, 2, 2), dtype=np.float32)
            return T()

    monkeypatch.setattr(ml, "get_sam", lambda: SlowPredictor())
    ml._embed_cache.clear()

    def png(color):
        buf = _io.BytesIO()
        PILImage.new("RGB", (4, 4), color).save(buf, "PNG")
        return buf.getvalue()

    threads = [threading.Thread(target=ml.embed_image, args=(png((i, 0, 0)),))
               for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(events) == 8
    for i in range(0, len(events), 2):
        assert events[i][0] == "set" and events[i + 1][0] == "emb", events
        assert events[i][1] == events[i + 1][1], \
            f"set_image와 predict 사이에 다른 스레드가 끼어들었다: {events}"


def test_batch_verdict_routes_by_detection_rate():
    """제로샷이 약한 건 정상이다. 문제는 그때 아무 안내가 없던 것."""
    from server.main import _batch_verdict

    assert _batch_verdict({"hit": 9, "found": 20}, 10)["verdict"] == "good"
    # 중간 커버리지는 'weak' 단정이 아니라 'partial' — 대상 없는 이미지가
    # 정상인 데이터셋일 수 있다 (test_api의 회귀 테스트 참조)
    assert _batch_verdict({"hit": 2, "found": 3}, 10)["verdict"] == "partial"
    empty = _batch_verdict({"hit": 0, "found": 0}, 10)
    assert empty["verdict"] == "empty"
    assert "프롬프트" in empty["advice"]  # 다음 수를 반드시 제시해야 한다


def test_batch_large_box_uses_xywh_not_xyxy():
    """bbox는 [x,y,w,h]다. x/y를 너비/높이에서 빼면 우측 박스를 놓친다."""
    from PIL import Image
    from server.main import _box_area_ratio

    image = Image.new("RGB", (200, 100))
    assert _box_area_ratio([100, 10, 180, 90], image) == 0.81


def test_drop_frame_filling_removes_degenerate_boxes():
    """프레임을 통째로 덮는 퇴화 검출은 버려야 한다.

    실측 사고: GDINO는 프롬프트에 맞는 것을 못 찾으면 입력 전체를 박스로
    뱉는다. 타일마다 하나씩 나오니 서명 12장 제로샷에서 박스 78개 중 69개가
    타일 격자 모양 쓰레기였고, 첫 오토라벨 화면이 격자로 덮였다.
    """
    from server.tiling import drop_frame_filling

    dets = [
        {"class_name": "a", "bbox": [0, 0, 800, 800]},        # 타일 전체
        {"class_name": "a", "bbox": [0, 80, 800, 720]},       # 면적비 0.90 — 면적 기준을 빠져나갔던 실측 케이스
        {"class_name": "a", "bbox": [10, 10, 80, 80]},        # 진짜 검출
        {"class_name": "a", "bbox": [0, 300, 800, 100]},      # 가로로 길고 납작 — 한 축만 커서 살아야 한다
    ]
    kept = drop_frame_filling(dets, 800, 800)
    assert [d["bbox"] for d in kept] == [[10, 10, 80, 80], [0, 300, 800, 100]], kept


# ---------- QA 매칭 ----------

def test_qa_match_finds_missing_and_spurious():
    from server.qa import _match

    preds = [
        {"class_name": "a", "bbox": [0, 0, 50, 50], "confidence": 0.9},    # 라벨과 일치
        {"class_name": "a", "bbox": [300, 300, 50, 50], "confidence": 0.8},  # 라벨 없음
    ]
    labels = [
        {"class_name": "a", "bbox": [2, 2, 50, 50]},
        {"class_name": "a", "bbox": [500, 500, 50, 50]},   # 모델이 못 찾음
    ]
    matched, spurious, missing = _match(preds, labels)
    assert len(matched) == 1
    assert len(spurious) == 1 and spurious[0]["bbox"][0] == 300
    assert len(missing) == 1 and missing[0]["bbox"][0] == 500


def test_missing_label_suggestion_rejects_overlapping_boxes():
    """'누락'은 새 객체여야 한다 — 기존 라벨과 겹친 박스를 반영하면 중복이 된다.

    실측 사고: 서명 데이터셋에서 IoU 0.406짜리 예측이 _match의 mAP 임계값
    0.5를 못 넘어 '누락'으로 나갔고, 원클릭 반영이 같은 서명에 박스를 하나 더
    박았다. 정탐 판정(IoU 0.5)과 신규 객체 판정은 다른 질문이다.
    """
    from server.qa import filter_new_objects

    labels = [{"class_name": "signature", "bbox": [653, 398, 198, 362]}]
    preds = [
        # 실제로 중복을 만들었던 박스 (IoU 0.406)
        {"class_name": "signature", "bbox": [641.9, 400.6, 156.1, 207.1], "confidence": 0.92},
        # 큰 라벨 안에 완전히 들어간 작은 박스 — IoU는 낮지만 새 객체가 아니다
        {"class_name": "signature", "bbox": [700, 450, 60, 60], "confidence": 0.8},
        # 진짜 새 객체 (겹침 없음)
        {"class_name": "signature", "bbox": [1200, 800, 150, 120], "confidence": 0.7},
    ]
    kept = filter_new_objects(preds, labels)
    assert [p["bbox"][0] for p in kept] == [1200], kept

    # 라벨이 없으면 전부 신규 — 걸러내면 안 된다
    assert len(filter_new_objects(preds, [])) == 3


def test_db_write_waits_for_concurrent_writer_instead_of_erroring():
    """동시 쓰기에서 즉사하지 않는다 — 심판이 판정을 기록하는 동안 프로젝트
    생성이 "database is locked" 500으로 죽었다 (실측). busy_timeout이 있으면
    잠깐 기다렸다가 성공해야 한다."""
    import tempfile
    from pathlib import Path as _Path

    from server import db

    # 전용 임시 DB — 실서버 DB나 다른 테스트와 얽히지 않게
    tmp = _Path(tempfile.mkdtemp()) / "lock.db"
    orig = db.DB_PATH
    db.DB_PATH = tmp
    try:
        db.init_db()
        _run_lock_scenario(db)
    finally:
        db.DB_PATH = orig


def _run_lock_scenario(db):
    import threading
    import time

    c1 = db.get_db()
    c1.execute("BEGIN IMMEDIATE")
    c1.execute("INSERT INTO projects (name) VALUES ('locker')")

    errors = []

    def writer():  # sqlite 커넥션은 만든 스레드에서만 쓸 수 있다
        try:
            c2 = db.get_db()
            # busy_timeout 없으면 여기서 sqlite3.OperationalError: database is locked
            c2.execute("INSERT INTO projects (name) VALUES ('waiter')")
            c2.commit()
            c2.close()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=writer)
    t.start()
    time.sleep(0.3)   # writer가 잠금에 걸려 기다리는 동안
    c1.commit()       # 잠금 해제 — writer가 이어서 성공해야 한다
    t.join(timeout=15)
    assert not errors, errors
    names = {r[0] for r in c1.execute(
        "SELECT name FROM projects WHERE name IN ('locker','waiter')")}
    assert names == {"locker", "waiter"}
    c1.close()

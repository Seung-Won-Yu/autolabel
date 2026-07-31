"""핵심 로직 회귀 테스트 — 모델 추론 없이 도는 빠른 스위트.

여기 있는 케이스들은 전부 실제로 한 번씩 깨졌던 것들이다.
"""
import json
import os

import pytest

from server import sampling
from server.parts import parse_ontology
from server.tiling import merge_nms, tile_boxes
from server.train_worker import pick_arch, pick_imgsz


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


def test_pick_imgsz_keeps_default_for_small_data(tmp_path):
    """해상도 상향은 실측에서 소량 데이터에 일관되게 손해였다."""
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    assert pick_imgsz(tmp_path, n_images=100) == 640
    assert pick_imgsz(tmp_path, n_images=5000) == 1280


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
    from server import jobs

    jobs.start("autolabel", 4242, done=0, total=10)
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
    assert _batch_verdict({"hit": 2, "found": 3}, 10)["verdict"] == "weak"
    empty = _batch_verdict({"hit": 0, "found": 0}, 10)
    assert empty["verdict"] == "empty"
    assert "프롬프트" in empty["advice"]  # 다음 수를 반드시 제시해야 한다


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

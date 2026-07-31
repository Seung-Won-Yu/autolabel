"""핵심 로직 회귀 테스트 — 모델 추론 없이 도는 빠른 스위트.

여기 있는 케이스들은 전부 실제로 한 번씩 깨졌던 것들이다.
"""
import json

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

"""part-level 캐스케이드 — 객체 안의 객체(사람의 팔·눈·코 등) 검출.

조사 결과 확정된 원칙:
- 전체 이미지에 "person's arm" 같은 프롬프트를 던지면 실패한다 (grounding 모델은
  명사구만 접지하고 소유격·공간 관계를 이해하지 못함).
- 부모를 먼저 찾고 그 crop 안에서 part를 찾으면 성공률이 크게 오른다 (OV-PARTS의
  Oracle-Obj vs Pred-Obj 격차). crop 확대는 작은 객체 검출을 끌어올린다 (SAHI).
- 사람 부위는 범용 grounding보다 인체 전문 모델(포즈 키포인트)이 정확하다.

온톨로지 표기: 클래스 이름에 "부모.자식" (예: "person.head")을 쓰면 part로 취급한다.
"""
import json

import numpy as np
import torch
from PIL import Image

from server import ml

# 인체 부위는 키포인트에서 유도 — COCO 17 키포인트 기준 박스 생성 규칙
# (키포인트 그룹, 여백 비율)
HUMAN_PARTS = {
    "head": ([0, 1, 2, 3, 4], 0.55),       # 코·눈·귀 — 머리 윤곽까지 여유
    "face": ([0, 1, 2], 0.45),
    "torso": ([5, 6, 11, 12], 0.15),        # 어깨·엉덩이
    "left_arm": ([5, 7, 9], 0.25),
    "right_arm": ([6, 8, 10], 0.25),
    "left_leg": ([11, 13, 15], 0.2),
    "right_leg": ([12, 14, 16], 0.2),
    "arm": ([5, 7, 9, 6, 8, 10], 0.25),
    "leg": ([11, 13, 15, 12, 14, 16], 0.2),
}

_pose = None


def get_pose():
    """인체 키포인트 모델 (Apache-2.0 계열 YOLO pose) — 사람 부위 전용 경로."""
    global _pose
    if _pose is None:
        from ultralytics import YOLO

        _pose = YOLO("yolo11n-pose.pt")
    return _pose


def parse_ontology(ontology: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """온톨로지를 부모 클래스와 part 클래스로 분리. part는 '부모.자식' 표기."""
    parents, parts = [], {}
    for c in ontology:
        if "." in c["name"]:
            p, child = c["name"].split(".", 1)
            parts.setdefault(p, []).append({**c, "child": child})
        else:
            parents.append(c)
    return parents, parts


def _human_parts_from_pose(image: Image.Image, box: list[float],
                           wanted: list[dict]) -> list[dict]:
    """사람 crop에서 키포인트 → 부위 박스."""
    x, y, w, h = box
    crop = image.crop((int(x), int(y), int(x + w), int(y + h)))
    res = get_pose().predict(crop, device=ml.DEVICE, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints.data) == 0:
        return []
    kp = res.keypoints.data[0].cpu().numpy()  # [17, 3] = x, y, conf
    out = []
    for spec in wanted:
        rule = HUMAN_PARTS.get(spec["child"])
        if not rule:
            continue
        idxs, pad = rule
        pts = [kp[i] for i in idxs if i < len(kp) and kp[i][2] > 0.3]
        if len(pts) < 2:
            continue
        # numpy 스칼라는 JSON 직렬화가 안 되므로 파이썬 float으로 캐스팅
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        px, py = max(bw * pad, 8), max(bh * pad, 8)
        x1 = max(0, min(xs) - px)
        y1 = max(0, min(ys) - py)
        x2 = min(crop.width, max(xs) + px)
        y2 = min(crop.height, max(ys) + py)
        conf = float(np.mean([float(p[2]) for p in pts]))
        out.append({
            "class_name": spec["name"],
            # crop 좌표 → 원본 좌표 역변환
            "bbox": [round(x + x1, 1), round(y + y1, 1),
                     round(x2 - x1, 1), round(y2 - y1, 1)],
            "confidence": round(conf, 4),
        })
    return out


@torch.no_grad()
def _generic_parts_from_crop(image: Image.Image, box: list[float],
                             wanted: list[dict], parent_name: str) -> list[dict]:
    """일반 객체: 부모 crop을 확대해 그 안에서만 part 프롬프트 검출."""
    x, y, w, h = box
    pad = 0.05
    cx1 = max(0, int(x - w * pad))
    cy1 = max(0, int(y - h * pad))
    cx2 = min(image.width, int(x + w * (1 + pad)))
    cy2 = min(image.height, int(y + h * (1 + pad)))
    crop = image.crop((cx1, cy1, cx2, cy2))
    # 작은 part 검출률을 올리려면 확대가 유효 (SAHI 근거)
    scale = 1.0
    if max(crop.size) < 640:
        scale = 640 / max(crop.size)
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))

    # part 프롬프트는 부모 수식 형태로 ("person arm")
    onto = [{"name": s["name"], "prompt": s.get("prompt") or f"{parent_name} {s['child']}",
             "threshold": s.get("threshold", 0.3)} for s in wanted]
    dets = ml.detect(crop, onto)
    crop_area = crop.width * crop.height
    out = []
    for d in dets:
        bx, by, bw, bh = d["bbox"]
        # part가 부모 전체를 덮으면 "부모를 다시 찾은 것" — 버린다.
        # grounding 모델이 crop 안에서 부모를 재검출하는 흔한 실패 모드
        if (bw * bh) / crop_area > 0.75:
            continue
        out.append({**d, "bbox": [round(cx1 + bx / scale, 1), round(cy1 + by / scale, 1),
                                  round(bw / scale, 1), round(bh / scale, 1)]})
    return out


def detect_with_parts(image: Image.Image, ontology: list[dict],
                      parent_dets: list[dict]) -> list[dict]:
    """부모 검출 결과를 받아 각 부모 안에서 part를 찾는다.

    반환 항목에는 _parent_index가 붙어 저장 시 parent_annotation_id로 연결된다.
    """
    _parents, parts_by_parent = parse_ontology(ontology)
    if not parts_by_parent:
        return []
    results = []
    for i, p in enumerate(parent_dets):
        wanted = parts_by_parent.get(p["class_name"])
        if not wanted:
            continue
        if p["class_name"] == "person":
            found = _human_parts_from_pose(image, p["bbox"], wanted)
            # 키포인트로 못 만든 부위는 일반 경로로 보완
            got = {f["class_name"] for f in found}
            rest = [w for w in wanted if w["name"] not in got]
            if rest:
                found += _generic_parts_from_crop(image, p["bbox"], rest, p["class_name"])
        else:
            found = _generic_parts_from_crop(image, p["bbox"], wanted, p["class_name"])
        for f in found:
            results.append({**f, "_parent_index": i})
    return results

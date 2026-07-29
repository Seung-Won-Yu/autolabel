"""제로샷 Grounding DINO 오토라벨 vs GT — test 분할 mAP50 베이스라인.

파인튜닝 모델(Colab)과 동일한 기준으로 비교하기 위한 수치.
"""
import json
from pathlib import Path

import numpy as np
import supervision as sv

TEST = Path("data/rfdetr_person/test/_annotations.coco.json")  # GT
AUTO = Path("output/coco128_person/annotations.json")  # 제로샷 오토라벨 (전체)

gt = json.loads(TEST.read_text())
auto = json.loads(AUTO.read_text())

auto_img_by_name = {im["file_name"]: im["id"] for im in auto["images"]}
auto_anns = {}
for a in auto["annotations"]:
    auto_anns.setdefault(a["image_id"], []).append(a)

targets, predictions = [], []
for im in gt["images"]:
    gt_boxes = np.array([
        [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
        for a in gt["annotations"] if a["image_id"] == im["id"]
    ]).reshape(-1, 4)
    targets.append(sv.Detections(xyxy=gt_boxes, class_id=np.zeros(len(gt_boxes), dtype=int)))

    aid = auto_img_by_name[im["file_name"]]
    pred = auto_anns.get(aid, [])
    boxes = np.array([
        [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
        for a in pred
    ]).reshape(-1, 4)
    scores = np.array([a.get("score", 1.0) for a in pred])
    predictions.append(sv.Detections(
        xyxy=boxes, class_id=np.zeros(len(pred), dtype=int), confidence=scores))

m = sv.MeanAveragePrecision.from_detections(predictions=predictions, targets=targets)
print(f"제로샷 Grounding DINO — GT 기준 mAP50: {m.map50:.3f}, mAP50-95: {m.map50_95:.3f}")
print(f"test 이미지 {len(gt['images'])}장, GT 박스 {len(gt['annotations'])}개")

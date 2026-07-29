"""PCB 도그푸딩 평가: 전용 모델 오토라벨 vs GT — 평가 10장 mAP50."""
import sys
from pathlib import Path

import numpy as np
import requests
import supervision as sv

sys.path.insert(0, str(Path(__file__).parent))
from dogfood_pcb import CLASSES, N_EVAL, N_SEED, collect_samples, parse_gt  # noqa: E402

API = "http://127.0.0.1:8899/api"
EVAL_IDS = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
NAMES = list(CLASSES.values())
cls_idx = {n: i for i, n in enumerate(NAMES)}

samples = collect_samples()[N_SEED:N_SEED + N_EVAL]

targets, preds = [], []
total_dets = 0
for iid, (_, ann_path) in zip(EVAL_IDS, samples):
    r = requests.post(f"{API}/images/{iid}/autolabel", json={"masks": False}).json()
    dets = r["detections"]
    total_dets += len(dets)
    boxes = np.array([[d["bbox"][0], d["bbox"][1],
                       d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]]
                      for d in dets]).reshape(-1, 4)
    preds.append(sv.Detections(
        xyxy=boxes,
        class_id=np.array([cls_idx[d["class_name"]] for d in dets], dtype=int),
        confidence=np.array([d["confidence"] for d in dets])))

    gt = parse_gt(ann_path)
    gboxes = np.array([[a["bbox"][0], a["bbox"][1],
                        a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                       for a in gt]).reshape(-1, 4)
    targets.append(sv.Detections(
        xyxy=gboxes,
        class_id=np.array([cls_idx[a["class_name"]] for a in gt], dtype=int)))

m = sv.MeanAveragePrecision.from_detections(predictions=preds, targets=targets)
gt_n = sum(len(t) for t in targets)
print(f"엔진: {r['engine']}")
print(f"평가 10장 — 검출 {total_dets}개 / GT {gt_n}개")
print(f"mAP50: {m.map50:.3f} | mAP50-95: {m.map50_95:.3f}")

"""signature 프로젝트 평가: 전용 모델 오토라벨 vs GT — 미접촉 43장 mAP50."""
import json

import numpy as np
import requests
import supervision as sv

API = "http://127.0.0.1:8899/api"
PID = 4
HOLDOUT = 43  # 뒤 43장 = 학습에 안 쓴 이미지

imgs = requests.get(f"{API}/projects/{PID}/images").json()[-HOLDOUT:]
targets, preds = [], []
n_det = n_gt = 0

for im in imgs:
    gt = requests.get(f"{API}/images/{im['id']}/annotations").json()
    gb = np.array([[a["bbox"][0], a["bbox"][1],
                    a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                   for a in gt]).reshape(-1, 4)
    targets.append(sv.Detections(xyxy=gb, class_id=np.zeros(len(gb), dtype=int)))
    n_gt += len(gt)

    r = requests.post(f"{API}/images/{im['id']}/autolabel", json={"masks": False}).json()
    d = r["detections"]
    n_det += len(d)
    pb = np.array([[x["bbox"][0], x["bbox"][1],
                    x["bbox"][0] + x["bbox"][2], x["bbox"][1] + x["bbox"][3]]
                   for x in d]).reshape(-1, 4)
    preds.append(sv.Detections(
        xyxy=pb, class_id=np.zeros(len(d), dtype=int),
        confidence=np.array([x["confidence"] for x in d])))

m = sv.MeanAveragePrecision.from_detections(predictions=preds, targets=targets)
print(f"엔진: {r['engine']}")
print(f"홀드아웃 {len(imgs)}장 — 검출 {n_det}개 / GT {n_gt}개")
print(f"mAP50: {m.map50:.3f} | mAP50-95: {m.map50_95:.3f}")

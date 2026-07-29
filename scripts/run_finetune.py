"""Phase 0-3: RF-DETR nano 파인튜닝 + GT 평가 — Colab T4용 원파일 러너."""
import glob
import json
import time
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano

print("=== 학습 시작 (RF-DETR nano, 오토라벨 데이터) ===")
model = RFDETRNano()
t0 = time.time()
model.train(dataset_dir="dataset", epochs=30, batch_size=4, grad_accum_steps=4, lr=1e-4)
print(f"학습 시간: {(time.time() - t0) / 60:.1f}분")

print("=== GT 테스트셋 평가 ===")
test_dir = Path("dataset/test")
coco = json.loads((test_dir / "_annotations.coco.json").read_text())
gt_by_img = {}
for a in coco["annotations"]:
    gt_by_img.setdefault(a["image_id"], []).append(a["bbox"])

targets, predictions = [], []
for im in coco["images"]:
    det = model.predict(Image.open(test_dir / im["file_name"]).convert("RGB"), threshold=0.4)
    predictions.append(det)
    boxes = np.array(
        [[x, y, x + w, y + h] for x, y, w, h in gt_by_img.get(im["id"], [])]
    ).reshape(-1, 4)
    targets.append(sv.Detections(xyxy=boxes, class_id=np.zeros(len(boxes), dtype=int)))

m = sv.MeanAveragePrecision.from_detections(predictions=predictions, targets=targets)
print("=" * 60)
print(f"파인튜닝 RF-DETR nano — GT 기준 mAP50: {m.map50:.3f}, mAP50-95: {m.map50_95:.3f}")
print("제로샷 Grounding DINO 베이스라인 — mAP50: 0.906, mAP50-95: 0.763")
print("=" * 60)
ckpts = glob.glob("output/**/checkpoint_best_total.pth", recursive=True)
print("체크포인트:", ckpts)

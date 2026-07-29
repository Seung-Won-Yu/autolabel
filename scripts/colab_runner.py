"""Colab 원셀 러너: coco128 다운로드 → GDINO 오토라벨 → 데이터셋 구성 → RF-DETR 학습 → GT 평가.
base64로 인코딩해 Colab 셀 한 줄로 주입된다.
"""
import glob
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

DEV = "cuda"
print("=== 1/4 오토라벨 (Grounding DINO, person) ===")
proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base").to(DEV).eval()
imgs = sorted(Path("coco128/images/train2017").glob("*.jpg"))
auto = {}
t0 = time.time()
for p in imgs:
    im = Image.open(p).convert("RGB")
    with torch.no_grad():
        inp = proc(images=im, text="person.", return_tensors="pt").to(DEV)
        out = dino(**inp)
    r = proc.post_process_grounded_object_detection(
        out, threshold=0.35, text_threshold=0.35, target_sizes=[im.size[::-1]])[0]
    auto[p.name] = [
        ([float(v) for v in b], float(s))
        for b, s, l in zip(r["boxes"], r["scores"], r["text_labels"]) if "person" in l
    ]
print(f"{len(imgs)}장 라벨링 {time.time()-t0:.0f}s, 박스 {sum(len(v) for v in auto.values())}개")

print("=== 2/4 데이터셋 구성 (train/valid=오토라벨, test=GT) ===")
random.seed(42)
names = sorted(auto)
random.shuffle(names)
nt, nv = int(len(names) * 0.7), int(len(names) * 0.1)
splits = {"train": names[:nt], "valid": names[nt:nt + nv], "test": names[nt + nv:]}
CATS = [{"id": 1, "name": "person", "supercategory": "none"}]
for split, sn in splits.items():
    d = Path("dataset") / split
    d.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": CATS}
    aid = 0
    for iid, name in enumerate(sn, 1):
        src = Path("coco128/images/train2017") / name
        shutil.copy(src, d / name)
        w, h = Image.open(src).size
        coco["images"].append({"id": iid, "file_name": name, "width": w, "height": h})
        if split == "test":
            txt = Path("coco128/labels/train2017") / (Path(name).stem + ".txt")
            rows = txt.read_text().splitlines() if txt.exists() else []
            for line in rows:
                q = line.split()
                if not q or q[0] != "0":
                    continue
                cx, cy, bw, bh = map(float, q[1:5])
                aid += 1
                coco["annotations"].append({
                    "id": aid, "image_id": iid, "category_id": 1,
                    "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                    "area": bw * w * bh * h, "iscrowd": 0})
        else:
            for box, s in auto[name]:
                x1, y1, x2, y2 = box
                aid += 1
                coco["annotations"].append({
                    "id": aid, "image_id": iid, "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    (d / "_annotations.coco.json").write_text(json.dumps(coco))
    print(split, len(coco["images"]), "장 /", len(coco["annotations"]), "박스")

print("=== 3/4 RF-DETR nano 파인튜닝 ===")
from rfdetr import RFDETRNano
model = RFDETRNano()
t0 = time.time()
model.train(dataset_dir="dataset", epochs=30, batch_size=4, grad_accum_steps=4, lr=1e-4)
print(f"학습 {(time.time()-t0)/60:.1f}분")

print("=== 4/4 GT 평가 ===")
td = Path("dataset/test")
coco = json.loads((td / "_annotations.coco.json").read_text())
gtb = {}
for a in coco["annotations"]:
    gtb.setdefault(a["image_id"], []).append(a["bbox"])
targets, preds = [], []
for im in coco["images"]:
    det = model.predict(Image.open(td / im["file_name"]).convert("RGB"), threshold=0.4)
    preds.append(det)
    bx = np.array([[x, y, x + w, y + h] for x, y, w, h in gtb.get(im["id"], [])]).reshape(-1, 4)
    targets.append(sv.Detections(xyxy=bx, class_id=np.zeros(len(bx), dtype=int)))
m = sv.MeanAveragePrecision.from_detections(predictions=preds, targets=targets)
print("=" * 60)
print(f"RESULT 파인튜닝 mAP50={m.map50:.3f} mAP50-95={m.map50_95:.3f}")
print("RESULT 제로샷 베이스라인 mAP50=0.906 mAP50-95=0.763 (동일 분할, 로컬 측정)")
print("=" * 60)
print("체크포인트:", glob.glob("output/**/*.pth", recursive=True))

"""RF-DETR 학습용 데이터셋 패키징.

- train/valid: 오토라벨(Grounding DINO) 박스 — 학습은 자동 라벨로만
- test: coco128 GT(사람 수작업 라벨) — 평가는 진짜 정답으로
구조: data/rfdetr_person/{train,valid,test}/_annotations.coco.json + 이미지
"""
import json
import random
import shutil
from pathlib import Path

from PIL import Image

SRC_IMGS = Path("data/coco128/images/train2017")
GT_LABELS = Path("data/coco128/labels/train2017")  # YOLO 포맷, class 0 = person
AUTO = Path("output/coco128_person/annotations.json")
OUT = Path("data/rfdetr_person")

random.seed(42)

auto = json.loads(AUTO.read_text())
img_by_name = {im["file_name"]: im for im in auto["images"]}
anns_by_img = {}
for a in auto["annotations"]:
    anns_by_img.setdefault(a["image_id"], []).append(a)

names = sorted(img_by_name)
random.shuffle(names)
n_train = int(len(names) * 0.7)
n_valid = int(len(names) * 0.1)
splits = {
    "train": names[:n_train],
    "valid": names[n_train:n_train + n_valid],
    "test": names[n_train + n_valid:],
}

CATS = [{"id": 1, "name": "person", "supercategory": "none"}]


def yolo_to_coco_boxes(txt: Path, w: int, h: int):
    boxes = []
    if not txt.exists():
        return boxes
    for line in txt.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != "0":  # person만
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h])
    return boxes


for split, split_names in splits.items():
    d = OUT / split
    d.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": CATS}
    ann_id = 0
    for new_id, name in enumerate(split_names, 1):
        src = SRC_IMGS / name
        shutil.copy(src, d / name)
        w, h = Image.open(src).size
        coco["images"].append({"id": new_id, "file_name": name, "width": w, "height": h})
        if split == "test":
            # GT 라벨 사용
            for box in yolo_to_coco_boxes(GT_LABELS / f"{Path(name).stem}.txt", w, h):
                ann_id += 1
                coco["annotations"].append({
                    "id": ann_id, "image_id": new_id, "category_id": 1,
                    "bbox": [round(v, 1) for v in box],
                    "area": round(box[2] * box[3], 1), "iscrowd": 0,
                })
        else:
            # 오토라벨 사용
            old = img_by_name[name]
            for a in anns_by_img.get(old["id"], []):
                ann_id += 1
                coco["annotations"].append({
                    "id": ann_id, "image_id": new_id, "category_id": 1,
                    "bbox": a["bbox"], "area": a["area"], "iscrowd": 0,
                })
    (d / "_annotations.coco.json").write_text(json.dumps(coco))
    print(f"{split}: 이미지 {len(coco['images'])}장, 박스 {len(coco['annotations'])}개")

shutil.make_archive("data/rfdetr_person", "zip", OUT)
print("생성:", "data/rfdetr_person.zip")

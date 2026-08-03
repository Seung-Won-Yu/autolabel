"""SAM 3 vs Grounding DINO 제로샷 홀드아웃 비교 — mAP50.

세 도메인(PCB 결함·서명·사람)에서 같은 온톨로지로 두 엔진을 직접 호출해 비교한다.
학생 모델은 제외 — 순수 제로샷 시드 엔진 선택 근거용.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from dogfood_pcb import N_EVAL, N_SEED, ONTOLOGY as PCB_ONTOLOGY, collect_samples, parse_gt  # noqa: E402

from server import ml  # noqa: E402


def yolo_gt(label_path: Path, w: int, h: int) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        _, cx, cy, bw, bh = map(float, p)
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h])
    return boxes


def load_datasets():
    datasets = []

    pcb = []
    for img_path, ann_path in collect_samples()[N_SEED:N_SEED + N_EVAL]:
        gt = parse_gt(ann_path)
        pcb.append((img_path, gt))
    pcb_names = [c["name"] for c in PCB_ONTOLOGY]
    datasets.append(("pcb", PCB_ONTOLOGY, pcb_names, pcb))

    sig_onto = [{"name": "signature", "prompt": "handwritten signature", "threshold": 0.3}]
    sig_imgs = sorted(Path("data/signature/images/train").glob("*.jpg"))[-12:]
    sig = []
    for p in sig_imgs:
        lbl = Path("data/signature/labels/train") / (p.stem + ".txt")
        if not lbl.exists():
            continue
        with Image.open(p) as im:
            w, h = im.size
        sig.append((p, [{"class_name": "signature", "bbox": b} for b in yolo_gt(lbl, w, h)]))
    datasets.append(("signature", sig_onto, ["signature"], sig))

    person_onto = [{"name": "person", "prompt": "person", "threshold": 0.3}]
    coco = json.loads(Path("data/rfdetr_person/test/_annotations.coco.json").read_text())
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    person = []
    for im in coco["images"][:15]:
        p = Path("data/rfdetr_person/test") / im["file_name"]
        gt = [{"class_name": "person", "bbox": a["bbox"]} for a in by_img.get(im["id"], [])]
        person.append((p, gt))
    datasets.append(("person", person_onto, ["person"], person))

    return datasets


def to_sv(items: list[dict], cls_idx: dict, with_conf: bool) -> sv.Detections:
    boxes = np.array([[d["bbox"][0], d["bbox"][1],
                       d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]]
                      for d in items]).reshape(-1, 4)
    kw = {"xyxy": boxes,
          "class_id": np.array([cls_idx[d["class_name"]] for d in items], dtype=int).reshape(-1)}
    if with_conf:
        kw["confidence"] = np.array([d["confidence"] for d in items]).reshape(-1)
    return sv.Detections(**kw)


def run_engine(fn, name: str, ontology: list[dict], cls_idx: dict, data) -> dict:
    targets, preds = [], []
    n_det = n_gt = 0
    t0 = time.time()
    for img_path, gt in data:
        image = Image.open(img_path).convert("RGB")
        dets = [d for d in fn(image, ontology) if d["class_name"] in cls_idx]
        preds.append(to_sv(dets, cls_idx, with_conf=True))
        targets.append(to_sv(gt, cls_idx, with_conf=False))
        n_det += len(dets)
        n_gt += len(gt)
    m = sv.MeanAveragePrecision.from_detections(predictions=preds, targets=targets)
    return {"engine": name, "map50": m.map50, "map5095": m.map50_95,
            "dets": n_det, "gt": n_gt, "sec": time.time() - t0}


def main():
    for ds_name, ontology, names, data in load_datasets():
        cls_idx = {n: i for i, n in enumerate(names)}
        print(f"\n== {ds_name} ({len(data)}장) ==")
        for fn, label in [(ml.detect, "gdino"), (ml.detect_sam3, "sam3")]:
            r = run_engine(fn, label, ontology, cls_idx, data)
            print(f"{r['engine']:>6}: mAP50 {r['map50']:.3f} | mAP50-95 {r['map5095']:.3f} "
                  f"| 검출 {r['dets']} / GT {r['gt']} | {r['sec']:.0f}s")


if __name__ == "__main__":
    main()

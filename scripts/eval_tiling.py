"""타일링 효과 실측 — PCB 홀드아웃 10장에서 타일링 on/off mAP 비교."""
import sys
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from dogfood_pcb import CLASSES, N_SEED, collect_samples, parse_gt  # noqa: E402

from server import ml, tiling, train  # noqa: E402

NAMES = list(CLASSES.values())
cls_idx = {n: i for i, n in enumerate(NAMES)}
PID = 2
student = train.active_model(PID)
ontology = [{"name": n, "threshold": 0.25} for n in NAMES]

# 홀드아웃 10장 — 업로드 순서와 샘플 순서가 1:1이므로 iid = 5 + index
samples = collect_samples()[N_SEED:]
UP = Path("/Users/est/Documents/claude/autolabel") / "data" / "uploads" / str(PID)
paths = [UP / f"{5 + N_SEED + i}_{img.name}" for i, (img, _) in enumerate(samples)]
missing = [p for p in paths if not p.exists()]
if missing:
    raise SystemExit(f"이미지 경로 불일치: {missing[:2]}")


def evaluate(use_tiles: bool, tile: int = 800):
    targets, preds = [], []
    n_det = 0
    for p, (_, ann_path) in zip(paths, samples):
        img = Image.open(p).convert("RGB")
        if use_tiles:
            d = tiling.detect_tiled(
                img, lambda im: ml.detect_student(im, student, ontology), tile=tile)
        else:
            d = ml.detect_student(img, student, ontology)
        n_det += len(d)
        pb = np.array([[x["bbox"][0], x["bbox"][1],
                        x["bbox"][0] + x["bbox"][2], x["bbox"][1] + x["bbox"][3]]
                       for x in d]).reshape(-1, 4)
        preds.append(sv.Detections(
            xyxy=pb, class_id=np.array([cls_idx[x["class_name"]] for x in d], dtype=int),
            confidence=np.array([x["confidence"] for x in d])))
        gt = parse_gt(ann_path)
        gb = np.array([[a["bbox"][0], a["bbox"][1],
                        a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                       for a in gt]).reshape(-1, 4)
        targets.append(sv.Detections(
            xyxy=gb, class_id=np.array([cls_idx[a["class_name"]] for a in gt], dtype=int)))
    m = sv.MeanAveragePrecision.from_detections(predictions=preds, targets=targets)
    gt_n = sum(len(t) for t in targets)
    return m.map50, m.map50_95, n_det, gt_n


base = evaluate(False)
print(f"타일링 OFF — mAP50 {base[0]:.3f} · mAP50-95 {base[1]:.3f} · 검출 {base[2]}/{base[3]}")
for tile in (800, 512):
    t = evaluate(True, tile)
    delta = (t[0] - base[0]) * 100
    print(f"타일링 ON({tile}px) — mAP50 {t[0]:.3f} ({delta:+.1f}%p) · "
          f"mAP50-95 {t[1]:.3f} · 검출 {t[2]}/{t[3]}")

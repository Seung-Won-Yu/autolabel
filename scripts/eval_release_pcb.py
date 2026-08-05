"""번들 PCB 모델을 번들 DeepPCB 홀드아웃 10장에서 직접 재평가한다.

학습 val 점수와 실제 운용 임계값 점수를 섞지 않도록 두 값을 함께 출력한다.
실행: .venv/bin/python scripts/eval_release_pcb.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from dogfood_pcb import N_EVAL, N_SEED, collect_samples, parse_gt  # noqa: E402

ROOT = Path(__file__).parent.parent


def _detections(boxes, class_ids, confidence=None):
    kw = {
        "xyxy": np.asarray(boxes, dtype=float).reshape(-1, 4),
        "class_id": np.asarray(class_ids, dtype=int).reshape(-1),
    }
    if confidence is not None:
        kw["confidence"] = np.asarray(confidence, dtype=float).reshape(-1)
    return sv.Detections(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ROOT / "release/pcb_defect_yolo11n_map50_0.738.pt",
                    type=Path)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--operating-conf", default=0.30, type=float)
    ap.add_argument("--recall-conf", default=0.10, type=float)
    args = ap.parse_args()

    samples = collect_samples()[N_SEED:N_SEED + N_EVAL]
    if len(samples) != N_EVAL:
        raise SystemExit(f"DeepPCB 홀드아웃 부족: {len(samples)}/{N_EVAL}")
    model = YOLO(str(args.model))
    name_to_id = {name: i for i, name in model.names.items()}
    # 첫 호출의 그래프/디바이스 초기화 시간을 한 프로필에만 떠넘기지 않는다.
    model.predict(str(samples[0][0]), conf=0.30, max_det=300,
                  device=args.device, verbose=False, save=False)
    t0 = time.time()
    results = model.predict(
        [str(path) for path, _ in samples], conf=0.001, max_det=300,
        device=args.device, verbose=False, save=False)
    balanced_seconds = time.time() - t0

    targets = []
    raw = []
    n_gt = 0
    for result, (_, ann_path) in zip(results, samples):
        gt = [a for a in parse_gt(ann_path) if a["class_name"] in name_to_id]
        targets.append(_detections(
            [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
              a["bbox"][1] + a["bbox"][3]] for a in gt],
            [name_to_id[a["class_name"]] for a in gt]))
        n_gt += len(gt)
        raw.append((result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy()))

    def evaluate(source, min_conf: float):
        preds = []
        n_pred = 0
        for boxes, classes, confs in source:
            keep = confs >= min_conf
            preds.append(_detections(boxes[keep], classes[keep], confs[keep]))
            n_pred += int(keep.sum())
        metric = MeanAveragePrecision().update(predictions=preds, targets=targets).compute()
        return {"conf": min_conf, "detections": n_pred,
                "map50": round(float(metric.map50), 4),
                "map50_95": round(float(metric.map50_95), 4)}

    recall_started = time.time()
    recall_results = model.predict(
        [str(path) for path, _ in samples], conf=0.001, max_det=300,
        device=args.device, verbose=False, save=False, augment=True)
    recall_raw = [(result.boxes.xyxy.cpu().numpy(),
                   result.boxes.cls.cpu().numpy().astype(int),
                   result.boxes.conf.cpu().numpy()) for result in recall_results]
    recall_seconds = time.time() - recall_started

    print(json.dumps({
        "model": str(args.model), "images": len(samples), "ground_truth": n_gt,
        "classes": model.names, "full_pr": evaluate(raw, 0.001),
        "operating_point": evaluate(raw, args.operating_conf),
        "recall_profile": evaluate(recall_raw, args.recall_conf),
        "seconds": {"balanced": round(balanced_seconds, 2),
                    "recall": round(recall_seconds, 2)},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""배치 프리라벨링 스파이크: Grounding DINO(텍스트→박스) + SAM(박스→마스크).

사용법:
    python scripts/prelabel.py --images data/samples --out output/run1 \
        --classes "person,car,dog" --threshold 0.35

출력:
    output/run1/annotations.json   # COCO 포맷 (bbox + segmentation RLE)
    output/run1/yolo/*.txt         # YOLO 포맷 bbox
    output/run1/viz/*.jpg          # 오버레이 시각화
    output/run1/report.json        # 장당 지연시간·검출 수 리포트
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM_MODEL = "facebook/sam-vit-base"


def load_models():
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        SamModel,
        SamProcessor,
    )

    dino_proc = AutoProcessor.from_pretrained(DINO_MODEL)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(DEVICE).eval()
    sam_proc = SamProcessor.from_pretrained(SAM_MODEL)
    sam = SamModel.from_pretrained(SAM_MODEL).to(DEVICE).eval()
    return dino_proc, dino, sam_proc, sam


@torch.no_grad()
def detect_boxes(dino_proc, dino, image: Image.Image, classes: list[str], threshold: float):
    # Grounding DINO는 "class1. class2. class3." 형태의 프롬프트를 기대
    prompt = ". ".join(classes) + "."
    inputs = dino_proc(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    outputs = dino(**inputs)
    results = dino_proc.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes, scores, labels = [], [], []
    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        label = label.strip()
        # 프롬프트 병합 검출("person car" 등) 은 첫 매칭 클래스로 정규화
        matched = next((c for c in classes if c in label), None)
        if matched is None:
            continue
        boxes.append([round(v, 1) for v in box.tolist()])
        scores.append(round(score.item(), 4))
        labels.append(matched)
    return boxes, scores, labels


@torch.no_grad()
def boxes_to_masks(sam_proc, sam, image: Image.Image, boxes: list) -> list[np.ndarray]:
    if not boxes:
        return []
    raw = sam_proc(image, input_boxes=[boxes], return_tensors="pt")
    # MPS는 float64 미지원 — 프로세서가 만드는 float64 텐서를 float32로 강등
    inputs = {
        k: (v.to(torch.float32) if v.dtype == torch.float64 else v).to(DEVICE)
        for k, v in raw.items()
    }
    outputs = sam(**inputs, multimask_output=False)
    masks = sam_proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    return [m[0].numpy().astype(np.uint8) for m in masks]


def mask_to_coco_rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def draw_viz(image: Image.Image, boxes, labels, scores, masks) -> np.ndarray:
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rng = np.random.default_rng(0)
    palette = {l: tuple(int(c) for c in rng.integers(60, 255, 3)) for l in set(labels)}
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        color = palette[label]
        if i < len(masks):
            overlay = img.copy()
            overlay[masks[i] > 0] = color
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", required=True, help="쉼표 구분 클래스 목록")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--no-masks", action="store_true", help="SAM 마스크 생략(박스만)")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    out = Path(args.out)
    (out / "viz").mkdir(parents=True, exist_ok=True)
    (out / "yolo").mkdir(exist_ok=True)

    image_paths = sorted(
        p for p in Path(args.images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_paths:
        raise SystemExit(f"이미지 없음: {args.images}")

    print(f"디바이스: {DEVICE} | 이미지 {len(image_paths)}장 | 클래스 {classes}")
    t0 = time.perf_counter()
    dino_proc, dino, sam_proc, sam = load_models()
    print(f"모델 로드: {time.perf_counter() - t0:.1f}s")

    coco = {
        "images": [], "annotations": [],
        "categories": [{"id": i + 1, "name": c} for i, c in enumerate(classes)],
    }
    cat_id = {c: i + 1 for i, c in enumerate(classes)}
    ann_id = 0
    per_image = []

    for img_id, path in enumerate(image_paths, 1):
        image = Image.open(path).convert("RGB")
        w, h = image.size

        t_det = time.perf_counter()
        boxes, scores, labels = detect_boxes(dino_proc, dino, image, classes, args.threshold)
        det_ms = (time.perf_counter() - t_det) * 1000

        t_seg = time.perf_counter()
        masks = [] if args.no_masks else boxes_to_masks(sam_proc, sam, image, boxes)
        seg_ms = (time.perf_counter() - t_seg) * 1000

        coco["images"].append({"id": img_id, "file_name": path.name, "width": w, "height": h})
        yolo_lines = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box
            ann_id += 1
            ann = {
                "id": ann_id, "image_id": img_id, "category_id": cat_id[label],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "score": score, "iscrowd": 0,
                # 라벨 출처 메타데이터 — 재현성 요구사항
                "meta": {"model": DINO_MODEL, "sam": None if args.no_masks else SAM_MODEL,
                         "threshold": args.threshold},
            }
            if i < len(masks):
                ann["segmentation"] = mask_to_coco_rle(masks[i])
            coco["annotations"].append(ann)
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            yolo_lines.append(
                f"{cat_id[label] - 1} {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )

        (out / "yolo" / f"{path.stem}.txt").write_text("\n".join(yolo_lines))
        cv2.imwrite(str(out / "viz" / f"{path.stem}.jpg"),
                    draw_viz(image, boxes, labels, scores, masks))

        per_image.append({"file": path.name, "detections": len(boxes),
                          "det_ms": round(det_ms), "seg_ms": round(seg_ms)})
        print(f"[{img_id}/{len(image_paths)}] {path.name}: {len(boxes)}개 "
              f"(검출 {det_ms:.0f}ms + 분할 {seg_ms:.0f}ms)")

    (out / "annotations.json").write_text(json.dumps(coco, ensure_ascii=False))
    det_avg = np.mean([r["det_ms"] for r in per_image])
    seg_avg = np.mean([r["seg_ms"] for r in per_image])
    report = {
        "device": DEVICE, "images": len(per_image),
        "avg_det_ms": round(det_avg), "avg_seg_ms": round(seg_avg),
        "avg_total_ms": round(det_avg + seg_avg),
        "est_sec_per_1k_images": round((det_avg + seg_avg)),
        "per_image": per_image,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n평균: 검출 {det_avg:.0f}ms + 분할 {seg_avg:.0f}ms = 장당 {det_avg + seg_avg:.0f}ms")
    print(f"1,000장 추정: {(det_avg + seg_avg) / 60:.0f}분")
    print(f"결과: {out}/")


if __name__ == "__main__":
    main()

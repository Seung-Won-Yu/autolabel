"""모델 서비스: SAM 인코더(임베딩) + Grounding DINO(텍스트→박스) + SAM(박스→마스크).

Phase 0에서 검증된 코드 이식. 모델은 최초 사용 시 lazy 로드 (기동 속도 확보).
"""
import base64
import hashlib
import io
import os
import threading
import time

import numpy as np
import torch
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAM_CKPT = "models/sam_vit_l_0b3195.pth"
DINO_MODEL = "IDEA-Research/grounding-dino-base"
from server.tiling import drop_frame_filling  # noqa: E402

_lock = threading.Lock()
# SAM predictor는 전역 하나를 여러 스레드가 쓴다 (배치 스레드 + FastAPI 스레드풀).
# set_image→predict 구간이 원자적이지 않으면 A의 박스가 B의 특징맵으로 디코딩돼
# 엉뚱한 마스크가 저장되고, 잘못된 임베딩이 해시 키로 캐시에 눌러앉는다.
# torch 연산이 GIL을 놓기 때문에 진짜로 인터리빙된다. 디바이스가 하나뿐이라
# 직렬화해도 처리량 손해는 사실상 없다.
_infer_lock = threading.Lock()
_sam_predictor = None
_dino = None
_embed_cache: dict[str, dict] = {}
_current_key: str | None = None


class ModelsDisabled(RuntimeError):
    """무거운 모델 로딩이 꺼진 상태에서 모델을 요구했다."""


# 프론트 e2e는 SAM·GDINO가 필요 없다. 그런데도 이미지를 열 때마다 임베딩을
# 계산하려 들어, 메인 서버와 테스트 서버가 동시에 SAM ViT-L(1.2GB)을 MPS에
# 올리다 프로세스가 통째로 죽었다 (macOS 크래시 리포트까지 떴다).
NO_MODELS = os.environ.get("AUTOLABEL_NO_MODELS") == "1"


def get_sam():
    if NO_MODELS:
        raise ModelsDisabled("모델 로딩이 꺼져 있습니다 (AUTOLABEL_NO_MODELS=1)")
    global _sam_predictor
    with _lock:
        if _sam_predictor is None:
            from segment_anything import SamPredictor, sam_model_registry

            sam = sam_model_registry["vit_l"](checkpoint=SAM_CKPT).to(DEVICE)
            _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def get_dino():
    if NO_MODELS:
        raise ModelsDisabled("모델 로딩이 꺼져 있습니다 (AUTOLABEL_NO_MODELS=1)")
    global _dino
    with _lock:
        if _dino is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            proc = AutoProcessor.from_pretrained(DINO_MODEL)
            model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL)
                .to(DEVICE).eval()
            )
            _dino = (proc, model)
    return _dino


def embed_image(data: bytes) -> dict:
    """SAM 임베딩 — 이미지 해시 캐시. 브라우저 디코더용."""
    global _current_key
    key = hashlib.sha256(data).hexdigest()
    if key in _embed_cache:
        return {**_embed_cache[key], "cached": True, "encode_ms": 0}

    image = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    predictor = get_sam()
    t0 = time.perf_counter()
    with _infer_lock:
        predictor.set_image(image)
        _current_key = key
        emb = predictor.get_image_embedding().cpu().numpy().astype(np.float32)
    encode_ms = round((time.perf_counter() - t0) * 1000)
    result = {
        "key": key,
        "embedding": base64.b64encode(emb.tobytes()).decode(),
        "shape": list(emb.shape),
        "orig_size": [image.shape[0], image.shape[1]],
    }
    _embed_cache[key] = result
    return {**result, "cached": False, "encode_ms": encode_ms}


@torch.no_grad()
def exemplar_rerank(image_np: np.ndarray, bbox: list[float],
                    candidates: list[dict], sim_thr: float = 0.55) -> list[dict]:
    """후보 박스들을 예시와의 특징 유사도로 걸러낸다.

    특징맵 피크로 후보를 찾는 방식은 미세 객체(35px급)에서 무력하다.
    전용 모델이 이미 후보를 뽑아준 상황이면, 그 후보 중 예시와 닮은 것만
    남기는 편이 훨씬 정확하다 — "이런 것만 골라줘" 필터로 동작.
    """
    global _current_key
    predictor = get_sam()
    with _infer_lock:
        predictor.set_image(image_np)
        _current_key = None
        feats = predictor.features[0]  # 텐서를 잡아두면 잠금 밖에서도 안전
    C, FH, FW = feats.shape
    H, W = image_np.shape[:2]
    scale = 1024 / max(H, W)

    def pooled(bx, by, bw, bh):
        gx1 = int(bx * scale / 1024 * FW)
        gy1 = int(by * scale / 1024 * FH)
        gx2 = max(gx1 + 1, int((bx + bw) * scale / 1024 * FW))
        gy2 = max(gy1 + 1, int((by + bh) * scale / 1024 * FH))
        f = feats[:, gy1:gy2, gx1:gx2].mean(dim=(1, 2))
        return f / (f.norm() + 1e-6)

    ex = pooled(*bbox)
    out = []
    for c in candidates:
        sim = float((pooled(*c["bbox"]) * ex).sum())
        if sim >= sim_thr:
            out.append({**c, "confidence": round(sim, 4),
                        "meta": {**(c.get("meta") or {}), "exemplar_sim": round(sim, 4)}})
    return out


@torch.no_grad()
def exemplar_detect(image_np: np.ndarray, bbox: list[float],
                    topk: int = 20, sim_thr: float = 0.6) -> list[dict]:
    """시각 예시 검출 (PerSAM 패턴, 학습 프리):
    예시 박스의 SAM 특징 평균 → 특징맵 코사인 유사도 → 피크 → 포인트 프롬프트 → 마스크 → NMS.
    """
    global _current_key
    predictor = get_sam()
    # 피크→마스크 predict 루프까지 predictor의 set_image 상태에 의존한다 —
    # 전 구간을 잠금 안에서 돌린다 (다른 스레드가 중간에 이미지를 갈아끼우면
    # 예시와 무관한 이미지에서 마스크가 나온다)
    with _infer_lock:
        predictor.set_image(image_np)
        _current_key = None  # 임베딩 캐시와 별개 경로 — 상태 오염 방지

        feats = predictor.features[0]  # [256, 64, 64]
        C, FH, FW = feats.shape
        H, W = image_np.shape[:2]
        x, y, w, h = bbox
        # 박스를 특징맵 격자로 사영 (SAM은 긴 변 1024 + 패딩 — 특징맵은 패딩 포함 정사각)
        scale = 1024 / max(H, W)
        fx1 = int(x * scale / 1024 * FW); fy1 = int(y * scale / 1024 * FH)
        fx2 = max(fx1 + 1, int((x + w) * scale / 1024 * FW))
        fy2 = max(fy1 + 1, int((y + h) * scale / 1024 * FH))
        ex_feat = feats[:, fy1:fy2, fx1:fx2].mean(dim=(1, 2))  # [256]

        fmap = feats / (feats.norm(dim=0, keepdim=True) + 1e-6)
        ex_feat = ex_feat / (ex_feat.norm() + 1e-6)
        sim = torch.einsum("c,chw->hw", ex_feat, fmap)  # [64, 64] 코사인 유사도

        # 유효 영역(패딩 제외)만
        vh = max(1, round(H * scale / 1024 * FH))
        vw = max(1, round(W * scale / 1024 * FW))
        sim_valid = sim[:vh, :vw].clone()

        # 적응 임계값: 텍스처 빈약 도메인(PCB 등)은 유사도가 전체적으로 높게 번짐 —
        # 절대값과 분포 기반(mean+2σ) 중 높은 쪽 사용
        adaptive = (sim_valid.mean() + 2 * sim_valid.std()).item()
        thr = max(sim_thr, adaptive)

        # 로컬 피크 상위 K개 → 이미지 좌표 포인트
        flat = sim_valid.flatten()
        order = torch.argsort(flat, descending=True)
        picked_pts = []
        taken = torch.zeros_like(sim_valid, dtype=torch.bool)
        for idx in order[: topk * 8]:
            v = flat[idx].item()
            if v < thr or len(picked_pts) >= topk:
                break
            py, px = divmod(idx.item(), sim_valid.shape[1])
            if taken[max(0, py - 2):py + 3, max(0, px - 2):px + 3].any():
                continue  # 근접 피크 억제
            taken[py, px] = True
            picked_pts.append(((px + 0.5) / FW * 1024 / scale, (py + 0.5) / FH * 1024 / scale, v))

        def pooled_sim(bx, by, bw, bh) -> float:
            """후보 박스 영역 특징 풀링 → 예시와 코사인 (재검증 점수)."""
            gx1 = int(bx * scale / 1024 * FW); gy1 = int(by * scale / 1024 * FH)
            gx2 = max(gx1 + 1, int((bx + bw) * scale / 1024 * FW))
            gy2 = max(gy1 + 1, int((by + bh) * scale / 1024 * FH))
            f = feats[:, gy1:gy2, gx1:gx2].mean(dim=(1, 2))
            f = f / (f.norm() + 1e-6)
            return float((f * ex_feat).sum())

        # 각 피크 → SAM 마스크 → 박스 → 재검증
        candidates = []
        for cx, cy, peak in picked_pts:
            masks, ious, _ = predictor.predict(
                point_coords=np.array([[cx, cy]]), point_labels=np.array([1]),
                multimask_output=False)
            m = masks[0]
            ys, xs = np.where(m)
            if len(xs) < 8:
                continue
            bx1, by1, bx2, by2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            bw, bh = bx2 - bx1, by2 - by1
            # 크기 필터 (완화): 마스크가 결함 일부만 잡는 케이스 허용
            area_ratio = (bw * bh) / max(w * h, 1)
            if not (0.08 <= area_ratio <= 12.0):
                continue
            # 재검증: 후보 영역 특징이 예시와 실제로 닮았는지 (피크는 한 점, 이건 영역 전체)
            score = pooled_sim(bx1, by1, bw, bh)
            if score < sim_thr:
                continue
            candidates.append({
                "bbox": [bx1, by1, bw, bh],
                "confidence": round(score, 4),
            })

    # NMS (IoU 0.5)
    def iou(a, b):
        ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return inter / (aw * ah + bw * bh - inter + 1e-6)

    candidates.sort(key=lambda d: -d["confidence"])
    kept = []
    for c in candidates:
        if all(iou(c["bbox"], k["bbox"]) < 0.5 for k in kept):
            kept.append(c)
    return kept


SAM3_PATH = "models/sam3.pt"
_sam3 = None


def sam3_available() -> bool:
    from pathlib import Path

    return Path(SAM3_PATH).exists()


def get_sam3():
    """SAM 3 — 텍스트 명사구 하나로 이미지 내 모든 인스턴스를 분할한다.

    가중치는 Meta HF에서 접근 승인 후 받아 models/sam3.pt 로 두면 자동 사용된다.
    없으면 기존 Grounding DINO + SAM 경로를 그대로 쓴다.
    """
    global _sam3
    if _sam3 is None:
        from ultralytics.models.sam import SAM3SemanticPredictor

        _sam3 = SAM3SemanticPredictor(
            overrides={"conf": 0.25, "task": "segment", "mode": "predict",
                       "model": SAM3_PATH, "save": False, "verbose": False})
    return _sam3


def detect_sam3(image: Image.Image, ontology: list[dict]) -> list[dict]:
    """SAM 3 개념 분할 → 박스 목록. 클래스별 임계값 적용."""
    predictor = get_sam3()
    prompts = [c.get("prompt") or c["name"] for c in ontology]
    name_of = {(c.get("prompt") or c["name"]): c["name"] for c in ontology}
    thresholds = {c["name"]: float(c.get("threshold", 0.35)) for c in ontology}

    with _infer_lock:  # set_image→predict가 2단계라 동시 요청이 이미지를 바꿔치기할 수 있음
        predictor.set_image(image)
        results = predictor(text=prompts)
    dets = []
    for r in (results if isinstance(results, list) else [results]):
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        names = getattr(r, "names", None)
        labels = dict(enumerate(names)) if isinstance(names, (list, tuple)) else (names or {})
        for b in boxes:
            raw = labels.get(int(b.cls), prompts[0]) if labels else prompts[0]
            cls = name_of.get(raw, raw)
            conf = float(b.conf)
            if conf < thresholds.get(cls, 0.35):
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            dets.append({"class_name": cls,
                         "bbox": [round(x1, 1), round(y1, 1),
                                  round(x2 - x1, 1), round(y2 - y1, 1)],
                         "confidence": round(conf, 4)})
    return dets


_students: dict[str, object] = {}


def detect_student(image: Image.Image, model_row: dict, ontology: list[dict]) -> list[dict]:
    """파인튜닝된 학생 모델(YOLO) 추론 — 활성 champion이 있으면 이쪽이 기본."""
    from ultralytics import YOLO

    path = model_row["path"]
    if path not in _students:
        _students[path] = YOLO(path)
    names = model_row["meta"]["names"]
    thresholds = {c["name"]: float(c.get("threshold", 0.35)) for c in ontology}
    result = _students[path].predict(image, device=DEVICE, verbose=False)[0]
    detections = []
    for box in result.boxes:
        name = names[int(box.cls)]
        conf = float(box.conf)
        if conf < thresholds.get(name, 0.35):
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append({
            "class_name": name,
            "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
            "confidence": round(conf, 4),
        })
    return detections


@torch.no_grad()
def detect(image: Image.Image, ontology: list[dict]) -> list[dict]:
    """온톨로지 기반 검출: 클래스별 프롬프트·임계값 적용.

    프레임을 거의 다 덮는 박스는 버린다 (MAX_FRAME_COVERAGE). 프레임을 꽉
    채우는 객체를 찾는 용도라면 이 상수를 올릴 것.
    """
    proc, dino = get_dino()
    prompts = {c["name"]: c.get("prompt") or c["name"] for c in ontology}
    thresholds = {c["name"]: float(c.get("threshold", 0.35)) for c in ontology}
    text = ". ".join(prompts.values()) + "."
    min_thr = min(thresholds.values()) if thresholds else 0.35

    inputs = proc(images=image, text=text, return_tensors="pt").to(DEVICE)
    outputs = dino(**inputs)
    results = proc.post_process_grounded_object_detection(
        outputs, threshold=min_thr, text_threshold=min_thr,
        target_sizes=[image.size[::-1]],
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        label = label.strip()
        matched = next(
            (name for name, p in prompts.items() if p in label or name in label), None)
        if matched is None or score.item() < thresholds[matched]:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        detections.append({
            "class_name": matched,
            "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
            "confidence": round(score.item(), 4),
        })
    # 프롬프트에 맞는 것을 못 찾으면 GDINO는 입력 전체를 박스로 뱉는다.
    # 타일링과 만나면 이미지가 타일 격자 모양 박스로 덮인다 (실측: 서명 12장
    # 제로샷에서 박스 78개 중 69개). 학습된 학생 모델은 이 실패 모드가 없어
    # 여기, 즉 GDINO 경로에서만 막는다. 타일 검출이면 image가 곧 타일이다.
    return drop_frame_filling(detections, image.width, image.height)


@torch.no_grad()
def boxes_to_masks(image: Image.Image, boxes_xywh: list[list[float]]) -> list[dict]:
    """박스 배치 → COCO RLE 마스크 목록."""
    from pycocotools import mask as mask_utils

    if not boxes_xywh:
        return []
    arr = np.array(image.convert("RGB"))
    predictor = get_sam()
    with _infer_lock:
        predictor.set_image(arr)
        boxes_xyxy = torch.tensor(
            [[x, y, x + w, y + h] for x, y, w, h in boxes_xywh],
            dtype=torch.float32, device=DEVICE)
        tb = predictor.transform.apply_boxes_torch(boxes_xyxy, arr.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=tb, multimask_output=False)
    rles = []
    for m in masks[:, 0].cpu().numpy():
        rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    return rles

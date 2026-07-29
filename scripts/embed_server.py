"""SAM 인코더 서버: 이미지 업로드 → 임베딩 1회 계산 → 브라우저 디코더가 클릭마다 재사용.

실행: .venv/bin/python scripts/embed_server.py  (포트 8765)
"""
import base64
import hashlib
import io
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
sam = sam_model_registry["vit_l"](checkpoint="models/sam_vit_l_0b3195.pth").to(DEVICE)
predictor = SamPredictor(sam)

app = FastAPI()
embed_cache: dict[str, dict] = {}  # 이미지 해시 → 임베딩 (프로덕션은 Redis)
image_store: dict[str, np.ndarray] = {}  # 이미지 해시 → RGB 배열
current_key = None  # predictor에 set_image된 이미지

_dino = None


def get_dino():
    global _dino
    if _dino is None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        model = (
            AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
            .to(DEVICE).eval()
        )
        _dino = (proc, model)
    return _dino


@app.post("/api/embed")
async def embed(file: UploadFile):
    global current_key
    data = await file.read()
    key = hashlib.sha256(data).hexdigest()
    if key in embed_cache:
        return {**embed_cache[key], "cached": True, "encode_ms": 0}

    image = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    image_store[key] = image
    t0 = time.perf_counter()
    predictor.set_image(image)
    current_key = key
    encode_ms = round((time.perf_counter() - t0) * 1000)

    emb = predictor.get_image_embedding().cpu().numpy().astype(np.float32)
    result = {
        "key": key,
        "embedding": base64.b64encode(emb.tobytes()).decode(),
        "shape": list(emb.shape),
        "orig_size": [image.shape[0], image.shape[1]],  # H, W
        "input_size": list(predictor.input_size),
    }
    embed_cache[key] = result
    return {**result, "cached": False, "encode_ms": encode_ms}


@app.post("/api/autolabel")
async def autolabel(req: dict):
    """텍스트 프롬프트로 현재 이미지 전체 자동 라벨: DINO 박스 → SAM 마스크."""
    global current_key
    key, classes = req["key"], [c.strip() for c in req["classes"] if c.strip()]
    threshold = float(req.get("threshold", 0.35))
    image = image_store.get(key)
    if image is None:
        return {"error": "이미지 없음 — 먼저 이미지를 로드하세요"}

    t0 = time.perf_counter()
    proc, dino = get_dino()
    prompt = ". ".join(classes) + "."
    pil = Image.fromarray(image)
    with torch.no_grad():
        inputs = proc(images=pil, text=prompt, return_tensors="pt").to(DEVICE)
        outputs = dino(**inputs)
    results = proc.post_process_grounded_object_detection(
        outputs, threshold=threshold, text_threshold=threshold,
        target_sizes=[pil.size[::-1]],
    )[0]
    detections = []
    boxes = []
    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        matched = next((c for c in classes if c in label.strip()), None)
        if matched is None:
            continue
        boxes.append(box.tolist())
        detections.append({"label": matched, "score": round(score.item(), 3),
                           "box": [round(v, 1) for v in box.tolist()]})
    det_ms = round((time.perf_counter() - t0) * 1000)

    # SAM: 박스 배치 → 마스크
    t1 = time.perf_counter()
    overlay_b64 = None
    if boxes:
        if current_key != key:
            predictor.set_image(image)
            current_key = key
        boxes_t = torch.tensor(boxes, dtype=torch.float32, device=DEVICE)
        tb = predictor.transform.apply_boxes_torch(boxes_t, image.shape[:2])
        with torch.no_grad():
            masks, _, _ = predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=tb, multimask_output=False)
        palette = [(46, 204, 113), (52, 152, 219), (231, 76, 60), (241, 196, 15),
                   (155, 89, 182), (26, 188, 156)]
        h, w = image.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        for i, m in enumerate(masks[:, 0].cpu().numpy()):
            c = palette[i % len(palette)]
            overlay[m > 0] = [c[0], c[1], c[2], 150]
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        overlay_b64 = base64.b64encode(buf.getvalue()).decode()
    seg_ms = round((time.perf_counter() - t1) * 1000)

    return {"detections": detections, "overlay": overlay_b64,
            "det_ms": det_ms, "seg_ms": seg_ms}


@app.get("/")
async def index():
    return FileResponse("web/index.html")


app.mount("/static", StaticFiles(directory="web"), name="static")
app.mount("/samples", StaticFiles(directory="data/samples"), name="samples")


@app.get("/api/samples")
async def samples():
    from pathlib import Path

    return sorted(p.name for p in Path("data/samples").glob("*.jpg"))

if __name__ == "__main__":
    print(f"SAM 인코더 디바이스: {DEVICE}")
    uvicorn.run(app, host="127.0.0.1", port=8765)

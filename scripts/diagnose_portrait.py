"""세로 이미지에서 ONNX 디코더 출력이 torch 기준과 어긋나는지 진단.

비교: SamOnnxModel(torch) vs 익스포트된 ONNX — 동일 임베딩·포인트 입력.
대상: data/samples/000000000724.jpg (STOP 표지판, 세로 375x500).
"""
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

CKPT = "models/sam_vit_l_0b3195.pth"
IMG = "data/samples/000000000724.jpg"

image = np.array(Image.open(IMG).convert("RGB"))
H, W = image.shape[:2]
print(f"이미지: {W}x{H} (WxH) — 세로 여부: {H > W}")

sam = sam_model_registry["vit_l"](checkpoint=CKPT)
predictor = SamPredictor(sam)
predictor.set_image(image)
emb = predictor.get_image_embedding()

# 표지판 중앙 근처 포인트 (이미지 좌표)
pt = np.array([[190.0, 180.0]])
scale = 1024.0 / max(H, W)
coords = np.concatenate([pt * scale, [[0.0, 0.0]]])[None].astype(np.float32)
labels = np.array([[1.0, -1.0]], dtype=np.float32)

inputs = {
    "image_embeddings": emb.cpu().numpy().astype(np.float32),
    "point_coords": coords,
    "point_labels": labels,
    "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
    "has_mask_input": np.array([0.0], dtype=np.float32),
    "orig_im_size": np.array([H, W], dtype=np.float32),
}

# 1) torch 기준
model = SamOnnxModel(sam, return_single_mask=True)
with torch.no_grad():
    t_out = model(**{k: torch.from_numpy(v) for k, v in inputs.items()})
t_mask = (t_out[0][0, 0].numpy() > 0)

# 2) ONNX
sess = ort.InferenceSession("web/sam_decoder.onnx")
o_out = sess.run(None, inputs)
o_mask = o_out[0][0, 0] > 0

def bbox(m):
    ys, xs = np.where(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None

print(f"torch 마스크 shape: {t_out[0].shape}, bbox(x1,y1,x2,y2): {bbox(t_mask)}")
print(f"ONNX  마스크 shape: {o_out[0].shape}, bbox(x1,y1,x2,y2): {bbox(o_mask)}")
inter = (t_mask & o_mask).sum()
union = (t_mask | o_mask).sum()
print(f"torch vs ONNX IoU: {inter / union:.4f}")
print(f"클릭 포인트 (이미지 좌표): {pt[0]}")

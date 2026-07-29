"""SAM ViT-B 마스크 디코더를 ONNX로 익스포트 — 브라우저(onnxruntime-web) 실행용."""
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

CKPT = "models/sam_vit_l_0b3195.pth"
OUT = "web/sam_decoder.onnx"

sam = sam_model_registry["vit_l"](checkpoint=CKPT)
model = SamOnnxModel(sam, return_single_mask=True)

embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
dummy = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(0, 1024, (1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(0, 4, (1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, 4 * embed_size[0], 4 * embed_size[1], dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
}

with open(OUT, "wb") as f:
    torch.onnx.export(
        model,
        tuple(dummy.values()),
        f,
        export_params=True,
        opset_version=17,
        input_names=list(dummy.keys()),
        output_names=["masks", "iou_predictions", "low_res_masks"],
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
        dynamo=False,
    )
print(f"저장: {OUT}")

"""타일링 추론 — 작은 객체·고해상도 이미지 대응 (SAHI 패턴).

원리: 모델은 입력을 640px 등으로 축소해 보기 때문에, 1920px 이미지의 35px 결함은
12px로 쪼그라들어 사라진다. 이미지를 겹치는 타일로 쪼개 각각 추론하면 그 결함이
타일 안에서는 상대적으로 커져 검출된다. 조사 근거: SAHI가 소형 객체 AP +5~7%p.

비용: 타일 수만큼 추론 횟수가 늘어난다 (2x2면 5배 — 전체 1회 + 타일 4회).
그래서 작은 이미지나 큰 객체만 있는 경우엔 자동으로 건너뛴다.
"""
from PIL import Image

MIN_SIDE_FOR_TILING = 1000  # 이보다 작은 이미지는 타일링 이득이 없다


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter + 1e-6)


def merge_nms(dets: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """타일 경계에서 중복 검출된 박스를 합친다 (클래스별 NMS)."""
    out = []
    for d in sorted(dets, key=lambda x: -x.get("confidence", 0)):
        dup = False
        for k in out:
            if k["class_name"] == d["class_name"] and _iou(k["bbox"], d["bbox"]) > iou_thr:
                dup = True
                break
        if not dup:
            out.append(d)
    return out


def tile_boxes(w: int, h: int, tile: int = 800, overlap: float = 0.2):
    """겹치는 타일 좌표 생성. 겹침이 있어야 경계에 걸친 객체를 놓치지 않는다."""
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y, min(tile, w - x), min(tile, h - y)) for y in ys for x in xs]


def detect_tiled(image: Image.Image, detect_fn, tile: int = 800,
                 overlap: float = 0.2, include_full: bool = True) -> list[dict]:
    """전체 이미지 + 타일들에서 각각 검출한 뒤 합친다.

    detect_fn(PIL.Image) -> [{class_name, bbox, confidence}] 형태의 콜러블.
    """
    w, h = image.size
    dets = list(detect_fn(image)) if include_full else []
    if max(w, h) < MIN_SIDE_FOR_TILING:
        return dets  # 작은 이미지는 타일링 이득 없음

    for (tx, ty, tw, th) in tile_boxes(w, h, tile, overlap):
        crop = image.crop((tx, ty, tx + tw, ty + th))
        for d in detect_fn(crop):
            bx, by, bw, bh = d["bbox"]
            dets.append({**d, "bbox": [round(tx + bx, 1), round(ty + by, 1),
                                       round(bw, 1), round(bh, 1)],
                         "meta": {**(d.get("meta") or {}), "tiled": True}})
    return merge_nms(dets)


def should_tile(image: Image.Image, ontology: list[dict] | None = None) -> bool:
    """타일링이 이득일지 판단 — 큰 이미지에서만 켠다."""
    return max(image.size) >= MIN_SIDE_FOR_TILING

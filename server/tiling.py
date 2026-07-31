"""타일링 추론 — 작은 객체·고해상도 이미지 대응 (SAHI 패턴).

원리: 모델은 입력을 640px 등으로 축소해 보기 때문에, 1920px 이미지의 35px 결함은
12px로 쪼그라들어 사라진다. 이미지를 겹치는 타일로 쪼개 각각 추론하면 그 결함이
타일 안에서는 상대적으로 커져 검출된다. 조사 근거: SAHI가 소형 객체 AP +5~7%p.

비용: 타일 수만큼 추론 횟수가 늘어난다 (2x2면 5배 — 전체 1회 + 타일 4회).
그래서 작은 이미지나 큰 객체만 있는 경우엔 자동으로 건너뛴다.
"""
from PIL import Image

MIN_SIDE_FOR_TILING = 1000  # 이보다 작은 이미지는 타일링 이득이 없다

# 가로·세로 둘 다 이 비율 이상이면 "프레임 전체"로 보고 버린다.
# Grounding DINO는 프롬프트에 맞는 것을 못 찾으면 입력 전체를 박스로 뱉는다.
# 타일마다 그게 하나씩 나오면 이미지가 타일 격자 모양 박스로 덮인다
# (실측: 서명 12장 제로샷에서 박스 78개 중 69개가 이 쓰레기였다).
#
# 면적이 아니라 차원별로 보는 이유: 800x800 타일에서 800x720 박스는 면적비가
# 0.90뿐이라 면적 기준을 빠져나가지만, 폭이 타일과 정확히 같은 시점에서 이미
# 퇴화다. 반대로 가로로 길고 납작한 진짜 객체는 한 축만 크므로 살아남는다.
MAX_FRAME_COVERAGE = 0.85


def drop_frame_filling(dets: list[dict], w: int, h: int) -> list[dict]:
    """프레임을 거의 다 덮는 퇴화 검출 제거.

    호출 지점은 GDINO 검출부 한 곳(server/ml.py)이다. 학습된 학생 모델은 이
    실패 모드가 없고, 프레임을 채우는 검출이 정당할 수 있어 적용하지 않는다.
    """
    return [d for d in dets
            if not (d["bbox"][2] >= w * MAX_FRAME_COVERAGE
                    and d["bbox"][3] >= h * MAX_FRAME_COVERAGE)]


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
        # detect_fn은 타일을 받으므로 프레임 전체 판정도 타일 기준으로 이뤄진다
        for d in detect_fn(crop):
            bx, by, bw, bh = d["bbox"]
            dets.append({**d, "bbox": [round(tx + bx, 1), round(ty + by, 1),
                                       round(bw, 1), round(bh, 1)],
                         "meta": {**(d.get("meta") or {}), "tiled": True}})
    return merge_nms(dets)


def should_tile(image: Image.Image, ontology: list[dict] | None = None) -> bool:
    """타일링이 이득일지 판단 — 큰 이미지에서만 켠다."""
    return max(image.size) >= MIN_SIDE_FOR_TILING

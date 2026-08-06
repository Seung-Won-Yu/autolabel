"""콜드스타트 파운데이션 모델 합의 병합.

SAM3와 Grounding DINO의 confidence는 서로 보정된 값이 아니므로 점수를
부풀리지 않는다. 같은 클래스·같은 객체로 겹친 후보만 합의로 표시하고,
나머지는 삭제하지 않은 채 사람이 먼저 볼 단독 후보로 남긴다.
"""

MATCH_IOU = 0.45

# 양쪽 엔진을 다 돌려 감사 표본을 만드는 앞부분. 이 표본이 승인되면
# foundation.build_profile이 클래스별 경로를 정하는 근거가 된다.
SEED_IMAGES = 30
# 경로가 정해진 뒤에도 이 주기로 한 장은 양쪽을 다시 돌린다. 없으면 감사
# 표본이 더 안 쌓여 build_profile이 초기 표본에 영구히 갇힌다 (자기 봉인).
EXPLORE_EVERY = 10


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(aw * ah + bw * bh - inter, 1e-6)


def _with_ensemble(det: dict, agreement: str, **evidence) -> dict:
    return {
        **det,
        "meta": {
            **(det.get("meta") or {}),
            "ensemble": {"agreement": agreement, **evidence},
        },
    }


def fuse_foundation_detections(sam3: list[dict], gdino: list[dict],
                               match_iou: float = MATCH_IOU) -> list[dict]:
    """동일 클래스 후보를 일대일로 합의 처리하고 단독 후보도 보존한다.

    합의 박스는 개념 분할 경계를 이용하는 SAM3 박스를 대표로 쓴다. confidence는
    서로 다른 척도의 평균을 신뢰도로 오해하지 않도록 두 값 중 낮은 쪽을 쓴다.
    """
    matched_gdino: set[int] = set()
    fused: list[dict] = []
    sam_only: list[dict] = []

    for sam_det in sorted(sam3, key=lambda d: -float(d.get("confidence", 0))):
        candidates = [
            (idx, iou(sam_det["bbox"], dino_det["bbox"]))
            for idx, dino_det in enumerate(gdino)
            if idx not in matched_gdino
            and dino_det.get("class_name") == sam_det.get("class_name")
        ]
        best_idx, best_iou = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        if best_iou >= match_iou:
            dino_det = gdino[best_idx]
            matched_gdino.add(best_idx)
            fused.append(_with_ensemble(
                {**sam_det, "confidence": round(min(
                    float(sam_det.get("confidence", 0)),
                    float(dino_det.get("confidence", 0))), 4)},
                "consensus",
                sam3_confidence=sam_det.get("confidence"),
                gdino_confidence=dino_det.get("confidence"),
                # 최종 표시 박스는 SAM3 경계를 쓰지만, 사용자가 수정·삭제한 뒤에도
                # 각 엔진을 독립 채점할 수 있게 두 원본을 모두 보존한다.
                sam3_bbox=list(sam_det["bbox"]),
                gdino_bbox=list(dino_det["bbox"]),
                match_iou=round(best_iou, 4),
            ))
        else:
            sam_only.append(_with_ensemble(
                sam_det, "sam3_only", sam3_confidence=sam_det.get("confidence")))

    gdino_only = [
        _with_ensemble(det, "gdino_only", gdino_confidence=det.get("confidence"))
        for idx, det in enumerate(gdino) if idx not in matched_gdino
    ]
    # 불일치부터 검수하도록 단독 후보를 앞에, 각 그룹 안에서는 높은 점수부터.
    key = lambda det: -float(det.get("confidence", 0))  # noqa: E731
    return sorted(sam_only + gdino_only, key=key) + sorted(fused, key=key)


def agreement_counts(detections: list[dict]) -> dict[str, int]:
    counts = {"consensus": 0, "sam3_only": 0, "gdino_only": 0}
    for det in detections:
        value = (det.get("meta") or {}).get("ensemble", {}).get("agreement")
        if value in counts:
            counts[value] += 1
    return counts


def batch_engine(index: int, settled: str, audited_both: int,
                 seed_images: int = SEED_IMAGES,
                 explore_every: int = EXPLORE_EVERY) -> str:
    """배치의 index번째(0-based) 이미지에서 실제로 돌릴 파운데이션 모드.

    settled: 승인 근거로 정해진 기본 모드('routed') 또는 근거가 없을 때의
        기본 단일 엔진('sam3').
    audited_both: 배치 시작 시점에 이미 양쪽을 돌려둔 이미지 수.

    이중 추론을 이어갈지 말지는 품질로 추측하지 않는다. 합의율로 판단하던
    이전 방식은 재는 값이 틀렸다 — 합의 박스는 한쪽 엔진만 돌려도 나오므로
    이중 추론의 이득이 아니다. 이득은 상대 엔진만 찾은 후보인데, 합의율이
    낮을수록(= 서로 다른 것을 찾을수록) 이중 추론을 껐다. 방향이 반대였다.

    그래서 품질 판단을 빼고 명시적 예산으로 바꾼다. 앞의 seed_images장은
    근거를 만들기 위해 양쪽을 돌리고, 그 뒤에도 explore_every마다 한 장은
    양쪽을 돌려 근거가 계속 자라게 한다.
    """
    if audited_both + index < seed_images:
        return "ensemble"
    return "ensemble" if index % explore_every == 0 else settled

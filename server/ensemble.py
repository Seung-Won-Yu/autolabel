"""콜드스타트 파운데이션 모델 합의 병합.

SAM3와 Grounding DINO의 confidence는 서로 보정된 값이 아니므로 점수를
부풀리지 않는다. 같은 클래스·같은 객체로 겹친 후보만 합의로 표시하고,
나머지는 삭제하지 않은 채 사람이 먼저 볼 단독 후보로 남긴다.
"""

MATCH_IOU = 0.45
PILOT_IMAGES = 3
MIN_PILOT_AGREEMENT = 0.10


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


def pilot_should_continue(counts: dict[str, int], images_seen: int,
                          min_images: int = PILOT_IMAGES,
                          min_agreement: float = MIN_PILOT_AGREEMENT) -> bool | None:
    """초기 표본에서 실제 보완 신호가 있을 때만 느린 이중 추론을 계속한다."""
    if images_seen < min_images:
        return None
    total = sum(counts.get(key, 0) for key in ("consensus", "sam3_only", "gdino_only"))
    if total == 0:
        return False
    return counts.get("consensus", 0) / total >= min_agreement

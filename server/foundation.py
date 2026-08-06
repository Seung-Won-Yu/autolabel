"""콜드스타트 파운데이션 추론·검수 기반 엔진 보정.

SAM3와 Grounding DINO를 무조건 오래 함께 돌리는 대신, 처음 몇 장에서 양쪽
원본 후보를 보존한다. 사용자가 그 이미지를 승인하면 현재 라벨을 정답으로
재채점해 클래스별로 어느 엔진이 검수 작업량을 덜 만드는지 선택한다.
"""
import json
from collections import defaultdict

from server import ensemble, ml, tiling

MATCH_IOU = 0.50
MIN_REVIEWED_IMAGES = 3
MIN_CLASS_INSTANCES = 2
MISSING_BOX_COST = 3


def _mark_source(detections: list[dict], engine: str) -> list[dict]:
    """단일 엔진 결과에 감사 로그용 출처를 붙인다."""
    marked = []
    for det in detections:
        marked.append({
            **det,
            "meta": {**(det.get("meta") or {}), "foundation_engine": engine},
        })
    return marked


def detect(image, ontology: list[dict], engine: str = "ensemble",
           class_routes: dict[str, str] | None = None) -> tuple[list[dict], str]:
    """파운데이션 엔진 실행과 장애 시 폴백을 한 경계에서 처리한다.

    engine: 'sam3' | 'gdino' | 'foundation'은 단일 엔진을 강제한다. 그 외
    ('ensemble', 'routed', 호출부가 학생 모델을 못 찾고 넘긴 'auto')는 모두
    두 엔진 교차 경로로 간다 — 알 수 없는 값이 조용히 한쪽으로 새지 않는다.

    class_routes 값은 sam3/gdino/ensemble이다. 미결정 클래스는 ensemble로
    취급해 어느 클래스도 조용히 누락하지 않는다.
    """
    use_tiles = tiling.should_tile(image)

    if engine == "sam3" and ml.sam3_available():
        try:
            return _mark_source(ml.detect_sam3(image, ontology), "sam3"), "sam3(pilot 선택)"
        except Exception:
            # 초기 3장 뒤 SAM3를 선택했더라도 이후 한 장의 추론 실패가 배치
            # 전체를 중단시키면 안 된다. 같은 온톨로지를 GDINO로 즉시 보충한다.
            dets = (tiling.detect_tiled(image, lambda im: ml.detect(im, ontology))
                    if use_tiles else ml.detect(image, ontology))
            used = "foundation(sam3 선택 후 실패)" + ("+tiled" if use_tiles else "")
            return _mark_source(dets, "gdino"), used
    if engine in ("gdino", "foundation") or not ml.sam3_available():
        dets = (tiling.detect_tiled(image, lambda im: ml.detect(im, ontology))
                if use_tiles else ml.detect(image, ontology))
        return _mark_source(dets, "gdino"), "foundation" + ("+tiled" if use_tiles else "")

    routes = class_routes or {}
    sam_onto = [item for item in ontology
                if routes.get(item.get("name"), "ensemble") != "gdino"]
    gdino_onto = [item for item in ontology
                  if routes.get(item.get("name"), "ensemble") != "sam3"]
    routed = bool(class_routes)

    sam3_dets = gdino_dets = None
    sam3_error = gdino_error = None
    if sam_onto:
        try:
            sam3_dets = ml.detect_sam3(image, sam_onto)
        except Exception as exc:  # 한 엔진 장애가 전체 배치를 죽이면 안 된다
            sam3_error = exc
    if gdino_onto:
        try:
            gdino_dets = (tiling.detect_tiled(image, lambda im: ml.detect(im, gdino_onto))
                           if use_tiles else ml.detect(image, gdino_onto))
        except Exception as exc:
            gdino_error = exc

    # 클래스별 단일 엔진 경로가 선택된 뒤 그 엔진만 장애 나면 해당 클래스가
    # 조용히 사라져서는 안 된다. 살아 있는 반대 엔진으로 선택 클래스만 보충한다.
    if routed and sam3_error is not None and gdino_dets is not None:
        fallback = [item for item in ontology if routes.get(item.get("name")) == "sam3"]
        if fallback:
            extra = (tiling.detect_tiled(image, lambda im: ml.detect(im, fallback))
                     if use_tiles else ml.detect(image, fallback))
            gdino_dets.extend(extra)
    if routed and gdino_error is not None and sam3_dets is not None:
        fallback = [item for item in ontology if routes.get(item.get("name")) == "gdino"]
        if fallback:
            sam3_dets.extend(ml.detect_sam3(image, fallback))

    if sam3_dets is not None and gdino_dets is not None:
        dets = ensemble.fuse_foundation_detections(sam3_dets, gdino_dets)
        prefix = "routed" if routed else "ensemble"
        used = f"{prefix}(sam3+gdino)" + ("+tiled" if use_tiles else "")
    elif sam3_dets is not None:
        dets = (_mark_source(sam3_dets, "sam3") if routed
                else ensemble.fuse_foundation_detections(sam3_dets, []))
        used = ("routed(sam3)" if routed else "sam3(gdino 실패)")
    elif gdino_dets is not None:
        dets = (_mark_source(gdino_dets, "gdino") if routed
                else ensemble.fuse_foundation_detections([], gdino_dets))
        used = ("routed(gdino)" if routed else "foundation(sam3 실패)")
        if use_tiles:
            used += "+tiled"
    else:
        raise RuntimeError(
            f"SAM3와 Grounding DINO가 모두 실패했습니다: {sam3_error}; {gdino_error}")
    return dets, used


def _ran_engines(used_engine: str) -> set[str]:
    """표시용 엔진 문자열을 실제 실행된 파운데이션 엔진 집합으로 바꾼다."""
    if used_engine.startswith(("ensemble(", "routed(sam3+gdino)")):
        return {"sam3", "gdino"}
    if used_engine.startswith(("sam3(", "routed(sam3)")):
        return {"sam3"}
    if used_engine.startswith(("foundation", "routed(gdino)")):
        return {"gdino"}
    return set()


def raw_candidates(detections: list[dict]) -> list[dict]:
    """표시용 합의 결과에서 SAM3/GDINO 원본 후보를 다시 펼친다."""
    raw = []
    for det in detections:
        meta = det.get("meta") or {}
        ev = meta.get("ensemble") or {}
        agreement = ev.get("agreement")
        common = {"class_name": det.get("class_name")}
        if agreement == "consensus":
            for engine in ("sam3", "gdino"):
                bbox = ev.get(f"{engine}_bbox")
                if bbox is not None:
                    raw.append({**common, "engine": engine, "bbox": list(bbox),
                                "confidence": ev.get(f"{engine}_confidence")})
        elif agreement in ("sam3_only", "gdino_only"):
            engine = agreement.removesuffix("_only")
            raw.append({**common, "engine": engine, "bbox": list(det["bbox"]),
                        "confidence": ev.get(f"{engine}_confidence", det.get("confidence"))})
        elif meta.get("foundation_engine") in ("sam3", "gdino"):
            raw.append({**common, "engine": meta["foundation_engine"],
                        "bbox": list(det["bbox"]), "confidence": det.get("confidence")})
    return [item for item in raw if item["class_name"]]


def replace_audit(conn, project_id: int, image_id: int, detections: list[dict],
                  used_engine: str) -> bool:
    """이미지의 최신 파운데이션 원본 후보를 같은 트랜잭션에 저장한다."""
    ran = _ran_engines(used_engine)
    if not ran:  # 학생 모델 결과는 파운데이션 비교 표본이 아니다
        return False
    conn.execute("DELETE FROM foundation_candidates WHERE image_id=?", (image_id,))
    conn.execute(
        "INSERT INTO foundation_audits (image_id,project_id,sam3_ran,gdino_ran) "
        "VALUES (?,?,?,?) ON CONFLICT(image_id) DO UPDATE SET "
        "project_id=excluded.project_id,sam3_ran=excluded.sam3_ran,"
        "gdino_ran=excluded.gdino_ran,created_at=datetime('now')",
        (image_id, project_id, int("sam3" in ran), int("gdino" in ran)))
    for item in raw_candidates(detections):
        conn.execute(
            "INSERT INTO foundation_candidates "
            "(image_id,project_id,engine,class_name,bbox,confidence) VALUES (?,?,?,?,?,?)",
            (image_id, project_id, item["engine"], item["class_name"],
             json.dumps(item["bbox"]), item.get("confidence")))
    return True


def _score(image_ids: list[int], truths: dict[int, list[dict]],
           candidates: dict[int, list[dict]]) -> dict:
    tp = fp = fn = 0
    for image_id in image_ids:
        gt = truths.get(image_id, [])
        pred = candidates.get(image_id, [])
        pairs = sorted(
            ((ensemble.iou(p["bbox"], g["bbox"]), pi, gi)
             for pi, p in enumerate(pred) for gi, g in enumerate(gt)),
            reverse=True)
        used_pred, used_gt = set(), set()
        for iou, pi, gi in pairs:
            if iou < MATCH_IOU:
                break
            if pi not in used_pred and gi not in used_gt:
                used_pred.add(pi)
                used_gt.add(gi)
        tp += len(used_pred)
        fp += len(pred) - len(used_pred)
        fn += len(gt) - len(used_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(f1, 3), "review_cost": fp + MISSING_BOX_COST * fn,
    }


def _selection(reviewed_images: int, support: int, scores: dict[str, dict]) -> str:
    if reviewed_images < MIN_REVIEWED_IMAGES or support < MIN_CLASS_INSTANCES:
        return "comparing"
    sam_cost = scores["sam3"]["review_cost"]
    gdino_cost = scores["gdino"]["review_cost"]
    if sam_cost == gdino_cost:
        return "ensemble"
    best = "sam3" if sam_cost < gdino_cost else "gdino"
    saving = abs(sam_cost - gdino_cost)
    worse = max(sam_cost, gdino_cost, 1)
    # 작은 표본의 1건 차이로 경로를 고정하지 않는다. 2건 이상이면서 검수 비용
    # 20% 이상 절감되는 경우만 단일 엔진을 선택한다.
    return best if saving >= 2 and saving / worse >= 0.20 else "ensemble"


def build_profile(conn, project_id: int, ontology: list[dict]) -> dict:
    """양쪽 엔진을 실행했고 사람이 승인한 이미지로 클래스별 성능을 계산한다."""
    audited = conn.execute(
        "SELECT a.image_id FROM foundation_audits a JOIN images i ON i.id=a.image_id "
        "WHERE a.project_id=? AND a.sam3_ran=1 AND a.gdino_ran=1 "
        "AND i.status='approved' ORDER BY a.image_id", (project_id,)).fetchall()
    image_ids = [row["image_id"] for row in audited]
    truths: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    candidates: dict[str, dict[str, dict[int, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))

    if image_ids:
        marks = ",".join("?" * len(image_ids))
        for row in conn.execute(
                f"SELECT image_id,class_name,bbox FROM annotations WHERE image_id IN ({marks})",
                image_ids):
            truths[row["class_name"]][row["image_id"]].append({"bbox": json.loads(row["bbox"])})
        for row in conn.execute(
                f"SELECT image_id,engine,class_name,bbox FROM foundation_candidates "
                f"WHERE image_id IN ({marks})", image_ids):
            candidates[row["class_name"]][row["engine"]][row["image_id"]].append(
                {"bbox": json.loads(row["bbox"])})

    classes = []
    for item in ontology:
        name = item.get("name")
        if not name:
            continue
        support = sum(len(v) for v in truths[name].values())
        scores = {
            engine: _score(image_ids, truths[name], candidates[name][engine])
            for engine in ("sam3", "gdino")
        }
        classes.append({
            "name": name, "support": support,
            "selection": _selection(len(image_ids), support, scores),
            "sam3": scores["sam3"], "gdino": scores["gdino"],
        })

    decided = sum(c["selection"] != "comparing" for c in classes)
    status = "ready" if classes and decided == len(classes) else ("partial" if decided else "learning")
    routes = {c["name"]: c["selection"] for c in classes
              if c["selection"] != "comparing"}
    return {
        "status": status,
        "reviewed_images": len(image_ids),
        "required_images": MIN_REVIEWED_IMAGES,
        "remaining_images": max(0, MIN_REVIEWED_IMAGES - len(image_ids)),
        "classes": classes,
        "routes": routes,
    }


def useful_routes(profile: dict) -> dict[str, str]:
    """검수 근거가 생긴 클래스만 반환한다. 미결정 클래스는 양쪽을 유지한다."""
    return profile.get("routes") or {}

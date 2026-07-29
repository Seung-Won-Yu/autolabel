"""QA층: 라벨 의심 랭킹 + 자동 승인 임계값 추천.

원리 (confident-learning 경량판): 학생 모델 예측과 저장된 라벨의 불일치를 스코어링.
모델과 라벨이 싸우는 이미지 = 라벨 오류이거나 모델 약점 — 둘 다 리뷰 가치 최상.
"""
import json
from pathlib import Path

from PIL import Image

from server import ml, train
from server.db import get_db, row_to_dict

ROOT = Path(__file__).parent.parent


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter + 1e-6)


def _match(preds, labels, iou_thr=0.5):
    """예측↔라벨 그리디 매칭. 반환: (매칭쌍, 미매칭 예측, 미매칭 라벨)."""
    used = set()
    pairs = []
    for p in sorted(preds, key=lambda d: -d["confidence"]):
        best, best_iou = None, iou_thr
        for i, l in enumerate(labels):
            if i in used:
                continue
            v = _iou(p["bbox"], l["bbox"])
            if v >= best_iou:
                best, best_iou = i, v
        if best is not None:
            used.add(best)
            pairs.append((p, labels[best]))
        else:
            pairs.append((p, None))
    unmatched_labels = [l for i, l in enumerate(labels) if i not in used]
    matched = [(p, l) for p, l in pairs if l is not None]
    spurious = [p for p, l in pairs if l is None]
    return matched, spurious, unmatched_labels


def analyze(pid: int) -> dict:
    """전체 라벨 이미지에 학생 모델 실행 → 이미지별 의심 점수 저장 + 임계값 추천."""
    student = train.active_model(pid)
    if not student:
        return {"error": "활성 학생 모델 없음 — 먼저 파인튜닝 필요"}

    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    ontology = json.loads(proj["ontology"])
    images = conn.execute(
        "SELECT * FROM images WHERE project_id=? AND status IN ('approved','prelabeled')",
        (pid,)).fetchall()

    # 추천 임계값 산출용: 골드 val에서 (conf, 정답 여부) 수집
    calib = {c["name"]: [] for c in ontology}
    scored = []
    # 낮은 임계값으로 예측 뽑아 매칭 (평가용이라 관대하게)
    loose_ontology = [{**c, "threshold": 0.1} for c in ontology]

    for im in images:
        labels = [row_to_dict(a) for a in conn.execute(
            "SELECT * FROM annotations WHERE image_id=?", (im["id"],))]
        img_path = ROOT / "data" / "uploads" / str(pid) / f"{im['id']}_{im['file_name']}"
        preds = ml.detect_student(Image.open(img_path).convert("RGB"), student, loose_ontology)

        matched, spurious, missing = _match(preds, labels)
        mismatch = [(p, l) for p, l in matched if p["class_name"] != l["class_name"]]
        spurious_hi = [p for p in spurious if p["confidence"] >= 0.5]
        # 확신도 가중: 모델이 "확신하며" 라벨과 싸울수록 라벨 오류 가능성↑.
        # 저확신 불일치는 모델 오류일 가능성이 높아 자연히 감쇠 — 약한 모델의 노이즈 억제
        score = (4.0 * sum(p["confidence"] for p, _ in mismatch)
                 + 2.0 * sum(p["confidence"] for p in spurious_hi)
                 + 0.3 * len(missing))
        conn.execute("UPDATE images SET qa_score=? WHERE id=?", (round(score, 2), im["id"]))
        scored.append({"image_id": im["id"], "file_name": im["file_name"],
                       "score": round(score, 2), "mismatch": len(mismatch),
                       "possible_missing_label": len(spurious_hi),
                       "model_missed": len(missing)})

        if im["is_val"]:  # 골드 val에서만 캘리브레이션 수집 (라벨=정답 가정)
            for p, l in matched:
                calib[p["class_name"]].append((p["confidence"], p["class_name"] == l["class_name"]))
            for p in spurious:
                calib[p["class_name"]].append((p["confidence"], False))
    conn.commit()
    conn.close()

    # 클래스별 추천 임계값: 정밀도 95% 달성하는 최소 conf (골드 val 기준)
    # 표본 하한 8, 임계값 바닥 0.25 — 소표본 요행·무의미한 저값 추천 방지
    MIN_SUPPORT = 8
    thresholds = {}
    for cls, pts in calib.items():
        if len(pts) < MIN_SUPPORT:
            thresholds[cls] = {"tau": None, "note": f"val 표본 부족 ({len(pts)}<{MIN_SUPPORT}) — 승인 더 필요"}
            continue
        best = None
        for tau in [i / 20 for i in range(5, 20)]:  # 0.25부터
            sel = [ok for conf, ok in pts if conf >= tau]
            if len(sel) >= MIN_SUPPORT:
                prec = sum(sel) / len(sel)
                if prec >= 0.95:
                    best = {"tau": tau, "precision": round(prec, 3), "support": len(sel)}
                    break
        thresholds[cls] = best or {"tau": None, "note": "정밀도 95% 달성 불가 — 더 학습 필요"}

    scored.sort(key=lambda s: -s["score"])
    return {"images_analyzed": len(scored), "top_suspects": scored[:10],
            "recommended_thresholds": thresholds}

"""QA층: 라벨 의심 랭킹 + 자동 승인 임계값 추천.

원리 (confident-learning 경량판): 학생 모델 예측과 저장된 라벨의 불일치를 스코어링.
모델과 라벨이 싸우는 이미지 = 라벨 오류이거나 모델 약점 — 둘 다 리뷰 가치 최상.
"""
import json
import threading
from pathlib import Path

from PIL import Image

from server import ml, train
from server.db import get_db, row_to_dict

ROOT = Path(__file__).parent.parent

_jobs: dict[int, dict] = {}


def job_status(pid: int) -> dict:
    return _jobs.get(pid, {"status": "idle"})


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter + 1e-6)


def _overlap(a, b):
    """겹침 정도 — IoU와 양방향 포함률 중 최대.

    IoU만 보면 큰 라벨 안에 작은 예측이 통째로 들어가도 값이 낮게 나온다
    (예: 라벨 면적의 1/4을 차지하며 완전히 포함 → IoU 0.25). 그건 새 객체가
    아니라 같은 객체를 좁게 잡은 것이다.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return max(inter / (union + 1e-6),
               inter / (aw * ah + 1e-6), inter / (bw * bh + 1e-6))


def filter_new_objects(spurious, labels, overlap_thr=0.3):
    """'누락 라벨' 후보에서 기존 라벨과 겹치는 것을 걷어낸다.

    _match는 mAP 관례대로 IoU 0.5로 정탐/오탐을 가른다. 오류율 추정에는 맞지만
    "라벨에 없는 새 객체냐"를 묻는 데 그대로 쓰면 안 된다 — IoU 0.4로 겹친
    같은 객체가 '누락'으로 나가 원클릭 반영 시 중복 라벨이 박힌다
    (실측: 서명 데이터셋에서 IoU 0.406짜리 중복 생성).
    """
    return [p for p in spurious
            if all(_overlap(p["bbox"], l["bbox"]) < overlap_thr for l in labels)]


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


def analyze(pid: int, progress: dict | None = None) -> dict:
    """전체 라벨 이미지에 학생 모델 실행 → 이미지별 의심 점수 저장 + 임계값 추천.

    progress dict를 넘기면 done/total을 갱신한다 (대규모 심판 잡용).
    """
    student = train.active_model(pid)
    if not student:
        return {"error": "활성 학생 모델 없음 — 먼저 파인튜닝하거나 외부 모델을 등록하세요"}

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

    if progress is not None:
        progress.update(total=len(images), done=0)

    n_labels = n_mismatch = n_spurious = n_loose = n_missing = 0
    for n, im in enumerate(images, 1):
        labels = [row_to_dict(a) for a in conn.execute(
            "SELECT * FROM annotations WHERE image_id=?", (im["id"],))]
        # 연결 임포트 이미지는 원본 경로, 업로드 이미지는 복사본 경로
        src = im["src_path"] if "src_path" in im.keys() and im["src_path"] else None
        img_path = Path(src) if src else (
            ROOT / "data" / "uploads" / str(pid) / f"{im['id']}_{im['file_name']}")
        if not img_path.exists():
            continue
        preds = ml.detect_student(Image.open(img_path).convert("RGB"), student, loose_ontology)

        matched, spurious, missing = _match(preds, labels)
        mismatch = [(p, l) for p, l in matched if p["class_name"] != l["class_name"]]
        spurious_hi = [p for p in spurious if p["confidence"] >= 0.5]
        # 라벨에 아예 없는 새 객체와, 같은 객체를 다르게 잡은 것은 다른 문제다.
        # 전자는 진짜 누락, 후자는 박스가 헐거운 것 — 섞으면 오류율이 부풀려진다.
        new_hi = filter_new_objects(spurious_hi, labels)
        new_ids = {id(p) for p in new_hi}
        loose_hi = [p for p in spurious_hi if id(p) not in new_ids]
        # 확신도 가중: 모델이 "확신하며" 라벨과 싸울수록 라벨 오류 가능성↑.
        # 저확신 불일치는 모델 오류일 가능성이 높아 자연히 감쇠 — 약한 모델의 노이즈 억제
        score = (4.0 * sum(p["confidence"] for p, _ in mismatch)
                 + 2.0 * sum(p["confidence"] for p in new_hi)
                 + 0.8 * sum(p["confidence"] for p in loose_hi)
                 + 0.3 * len(missing))
        conn.execute("UPDATE images SET qa_score=? WHERE id=?", (round(score, 2), im["id"]))
        scored.append({"image_id": im["id"], "file_name": im["file_name"],
                       "score": round(score, 2), "mismatch": len(mismatch),
                       "possible_missing_label": len(new_hi),
                       "loose_box": len(loose_hi),
                       "model_missed": len(missing)})
        n_labels += len(labels)
        n_mismatch += len(mismatch)
        n_spurious += len(new_hi)
        n_loose += len(loose_hi)
        n_missing += len(missing)
        if progress is not None and n % 20 == 0:
            progress.update(done=n)
            conn.commit()

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
    flagged = [s for s in scored if s["score"] > 0]
    # 라벨 오류율 추정: 모델이 라벨과 다투는 비율 (클래스 불일치 + 진짜 누락).
    # 박스가 헐거운 것(loose)은 객체 자체는 라벨돼 있으므로 오류율에 안 넣는다 —
    # 넣으면 같은 객체를 두 번 세어 오류율이 부풀려진다.
    est_rate = (n_mismatch + n_spurious) / max(n_labels, 1)
    return {
        "images_analyzed": len(scored),
        "labels_checked": n_labels,
        "flagged_images": len(flagged),
        "estimated_label_error_rate": round(est_rate, 4),
        "breakdown": {"class_mismatch": n_mismatch,
                      "possible_missing_label": n_spurious,
                      "loose_box": n_loose,
                      "model_missed": n_missing},
        "top_suspects": scored[:20],
        "recommended_thresholds": thresholds,
    }


def _run_judge(pid: int):
    from server import jobs

    job = _jobs[pid]
    try:
        result = analyze(pid, progress=job)
        if result.get("error"):
            job.update(status="failed", error=result["error"])
            jobs.update("qa", pid, status="failed", error=result["error"])
        else:
            job.update(status="completed", result=result, done=job.get("total", 0))
            # 결과 본문은 크니 디스크 기록에는 상태만 남긴다 (중단 판별용)
            jobs.update("qa", pid, status="completed", done=job.get("total", 0))
    except Exception as e:
        job.update(status="failed", error=str(e))
        jobs.update("qa", pid, status="failed", error=str(e))


def start_judge(pid: int) -> dict:
    """대규모 심판 — 백그라운드 실행 (수만 장 대응)."""
    from server import jobs

    if _jobs.get(pid, {}).get("status") == "running":
        return _jobs[pid]
    _jobs[pid] = {"status": "running", "done": 0, "total": 0}
    # 서버 재시작 시 기록이 사라져 "완료"로 오독되지 않게 디스크에도 남긴다
    jobs.start("qa", pid, done=0, total=0)
    threading.Thread(target=_run_judge, args=(pid,), daemon=True).start()
    return _jobs[pid]

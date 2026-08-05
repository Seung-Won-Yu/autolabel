"""자동 파인튜닝 루프: 승인 라벨 → YOLO 포맷 → 학습 → champion 게이트 → 등록.

학습 자체는 별도 프로세스(server.train_worker)에서 — 서버 재시작(--reload 포함)에도 생존.
상태는 data/runs/train_status_<pid>.json 파일 경유 (서버는 읽기 전용).

트레이너는 플러그인 개념 — 현재 YOLO11n(ultralytics, AGPL: 내부 도구 용도).
제품 배포 시 RF-DETR(Apache-2.0, CUDA) 트레이너로 교체 지점은 train_worker 하나.
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from server.db import get_db, row_to_dict

CODE_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(os.environ.get("AUTOLABEL_DATA_ROOT") or CODE_ROOT)
RUNS = DATA_ROOT / "data" / "runs"
UPLOADS = Path(os.environ.get("AUTOLABEL_DATA") or DATA_ROOT / "data") / "uploads"
MIN_APPROVED = 8        # 자동 트리거 최소 승인 이미지 수
RETRAIN_DELTA = 5       # 마지막 학습 이후 신규 승인 N장마다 재학습
VAL_RATIO = 0.2
MIN_VAL = 30   # 이보다 작은 val은 모델 품질을 가리지 못한다 (실측: 12장 val이 오판)
TEST_RATIO = 0.15
MIN_TEST = 20  # 홀드아웃 — 학습·게이트에서 완전 배제, 진짜 성능 보고 전용
MIN_CLASS_IMAGES = 3  # 클래스당 최소 다양성. 그 미만은 점수보다 누락 위험이 더 크다

_lock = threading.Lock()
_procs: dict[int, subprocess.Popen] = {}
_timers: dict[int, threading.Timer] = {}
DEBOUNCE_SEC = 20  # 연속 승인이 끝나길 기다렸다 학습 (대량 승인 시 조기 발동 방지)


def _status_path(pid: int) -> Path:
    return RUNS / f"train_status_{pid}.json"


def _training_counts(pid: int) -> tuple[int, int]:
    """현재 승인 수와 마지막 학습 스냅샷 크기."""
    conn = get_db()
    approved = conn.execute(
        "SELECT COUNT(*) c FROM images WHERE project_id=? AND status='approved'",
        (pid,)).fetchone()["c"]
    last = conn.execute(
        "SELECT MAX(train_images) m FROM models WHERE project_id=?", (pid,)).fetchone()["m"] or 0
    conn.close()
    return approved, last


def _run_scheduled(pid: int) -> None:
    with _lock:
        _timers.pop(pid, None)
    maybe_start_training(pid)


def plan_splits(assigned: dict, groups: dict | None = None) -> dict:
    """이미지별 train/val/test 배정. 입력 {id: 기존 split 또는 None} → {id: split}.

    val·test 하한(MIN_VAL·MIN_TEST)은 예산 안에서만 채운다 — train에 항상
    절반 이상을 남긴다. 실측 사고: 승인 8장에서 need_val=30이 pool을 전부
    val로 소진해 train 0장으로 학습이 실패했고, 배정은 DB에 고착이라 승인을
    아무리 늘려도 초기 이미지들이 학습에서 영영 배제됐다.
    """
    n = len(assigned)
    if n and all(s in ("val", "test") for s in assigned.values()):
        # 위 사고를 이미 겪은 DB — 전량이 val/test에 고착돼 train이 영원히
        # 0장이다. 이 경우만 배정을 처음부터 다시 한다 (라운드 간 비교 기준
        # 유지보다 학습이 되는 것이 먼저다).
        assigned = dict.fromkeys(assigned)
    groups = groups or {i: f"image:{i}" for i in assigned}
    members: dict[str, list[int]] = {}
    for iid in assigned:
        members.setdefault(str(groups.get(iid) or f"image:{iid}"), []).append(iid)

    # 기존 배정이 한 그룹 안에서 갈렸다면 다수결로 한쪽에 모은다. 비디오의
    # 인접 프레임이 train/val에 섞인 상태를 유지하는 것보다 라운드 기준을 한 번
    # 바로잡는 편이 낫다. 동률은 train을 우선해 학습 몫을 고갈시키지 않는다.
    group_split = {}
    priority = {"train": 2, "val": 1, "test": 0}
    for key, ids in members.items():
        known = [assigned[i] for i in ids if assigned[i] in priority]
        if known:
            group_split[key] = max(priority, key=lambda s: (known.count(s), priority[s]))

    counts = {s: sum(len(members[g]) for g, v in group_split.items() if v == s)
              for s in ("train", "val", "test")}
    pool = [g for g in members if g not in group_split]
    random.Random(42).shuffle(pool)
    budget = max(0, n - max(1, n // 2) - counts["val"] - counts["test"])
    for split, need in (("val", max(MIN_VAL, int(n * VAL_RATIO))),
                        ("test", max(MIN_TEST, int(n * TEST_RATIO)))):
        while counts[split] < need and pool and budget > 0:
            # 그룹 전체가 예산에 들어와야만 평가 셋으로 보낸다. 긴 영상 하나뿐인
            # 데이터는 같은 영상을 양쪽에 섞어 가짜 점수를 만드는 대신 train에 둔다.
            pick = next((g for g in reversed(pool) if len(members[g]) <= budget), None)
            if pick is None:
                break
            pool.remove(pick)
            group_split[pick] = split
            counts[split] += len(members[pick])
            budget -= len(members[pick])
    for key in pool:
        group_split[key] = "train"
    return {i: group_split[str(groups.get(i) or f"image:{i}")] for i in assigned}


def _image_group(im: dict) -> str:
    if im.get("group_key"):
        return im["group_key"]
    # migration 이전 비디오 프레임도 파일명으로 묶는다.
    m = re.match(r"^(.*)_f\d{6}\.[^.]+$", im["file_name"])
    return f"video:{m.group(1)}" if m else f"image:{im['id']}"


def _alive(os_pid) -> bool:
    """OS 프로세스 생존 확인 — 서버 재시작 후엔 Popen 핸들이 없다."""
    if not os_pid:
        return False   # 기록이 없으면 판단할 수 없으니 죽은 것으로 본다
    try:
        os.kill(int(os_pid), 0)
        return True
    except (OSError, ValueError):
        return False


def job_status(pid: int) -> dict:
    p = _status_path(pid)
    if not p.exists():
        return {"status": "idle"}
    st = json.loads(p.read_text())
    # 워커 프로세스가 죽었는데 running으로 남아 있으면 실패 처리 (고아 상태 감지)
    if st.get("status") == "running":
        proc = _procs.get(pid)
        if proc is not None and proc.poll() is not None:
            st.update(status="failed", error=f"워커 비정상 종료 (exit {proc.returncode})")
            p.write_text(json.dumps(st, ensure_ascii=False))
        elif proc is None and not _alive(st.get("pid_os")):
            # 서버가 재시작해 핸들이 없다 — OS에 직접 물어본다. 워커가 살아
            # 있으면 그대로 두고(별도 프로세스라 생존한다), 죽었으면 실패로.
            st.update(status="failed",
                      error="학습 워커가 사라졌습니다 (서버 재시작 중 종료) — 다시 시도하세요")
            p.write_text(json.dumps(st, ensure_ascii=False))
    return st


def _export_yolo_dataset(pid: int, out: Path) -> tuple[int, list[str]]:
    """승인된 이미지만 YOLO 포맷으로 export. train/val/test 분할은 고정.

    반환 장수는 test까지 포함한 승인 스냅샷 전체다. 이를 train+val만 세면 모델
    기록이 승인 수보다 작아져, 신규 승인 1장만 추가해도 RETRAIN_DELTA를 이미
    넘은 것으로 오인하고 불필요한 재학습이 시작된다.
    라벨 0개 승인 이미지 = 배경 네거티브 샘플로 포함."""
    conn = get_db()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    ontology = json.loads(proj["ontology"])
    names = [c["name"] for c in ontology]
    cls_id = {n: i for i, n in enumerate(names)}
    images = conn.execute(
        "SELECT * FROM images WHERE project_id=? AND status='approved'", (pid,)).fetchall()

    rows = []
    for im in images:
        anns = [row_to_dict(a) for a in conn.execute(
            "SELECT * FROM annotations WHERE image_id=?", (im["id"],))]
        rows.append((dict(im), anns))
    if not rows:
        conn.close()
        return 0, names

    # 3분할 고정: train(학습) / val(게이트 판정) / test(홀드아웃 — 학습·게이트 모두 배제).
    # 한 번 정해진 소속은 유지해 라운드 간 비교 기준이 흔들리지 않게 하고,
    # 데이터가 늘면 val·test도 함께 키운다 — 12장짜리 val이 홀드아웃 0.96짜리
    # 모델을 0.57로 오판해 승격을 막은 사고에서 나온 규칙.
    assigned = {r[0]["id"]: (r[0]["split"] or ("val" if r[0]["is_val"] else None))
                for r in rows}
    groups = {r[0]["id"]: _image_group(r[0]) for r in rows}
    plan = plan_splits(assigned, groups)
    val_ids = {i for i, s in plan.items() if s == "val"}
    test_ids = {i for i, s in plan.items() if s == "test"}
    conn.executemany("UPDATE images SET split=? WHERE id=?",
                     [(s, i) for i, s in plan.items()])
    conn.execute("UPDATE images SET is_val=1 WHERE split='val' AND project_id=?", (pid,))
    # 복구 재배정으로 val에서 빠진 이미지의 is_val도 정리 — 남겨두면 QA
    # 캘리브레이션이 train 이미지를 골드 val로 오인한다 (라벨 누출)
    conn.execute("UPDATE images SET is_val=0 WHERE split!='val' AND project_id=?", (pid,))
    conn.commit()
    conn.close()

    splits = {
        "val": [r for r in rows if r[0]["id"] in val_ids],
        # test는 학습 데이터에서 제외 — 진짜 성능 측정용으로만 남긴다
        "train": [r for r in rows if r[0]["id"] not in val_ids and r[0]["id"] not in test_ids],
        "test": [r for r in rows if r[0]["id"] in test_ids],
    }

    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for im, anns in items:
            # 연결 임포트 이미지는 원본 경로, 업로드 이미지는 복사본 경로
            src = Path(im["src_path"]) if im.get("src_path") else (
                UPLOADS / str(pid) / f"{im['id']}_{im['file_name']}")
            if not src.exists():
                continue  # 원본이 사라진 연결 이미지는 건너뜀
            dst = out / "images" / split / f"{im['id']}_{im['file_name']}"
            shutil.copy(src, dst)
            lines = []
            for a in anns:
                if a["class_name"] not in cls_id:
                    continue
                x, y, w, h = a["bbox"]
                lines.append(
                    f"{cls_id[a['class_name']]} "
                    f"{(x + w / 2) / im['width']:.6f} {(y + h / 2) / im['height']:.6f} "
                    f"{w / im['width']:.6f} {h / im['height']:.6f}")
            (out / "labels" / split / f"{Path(dst.name).stem}.txt").write_text("\n".join(lines))

    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names: {json.dumps(names)}\n")
    return sum(len(items) for items in splits.values()), names


def training_readiness(pid: int) -> dict | None:
    """학습 시작 전에 UI가 보여줄 읽기 전용 준비도와 예상 분할."""
    conn = get_db()
    project = conn.execute("SELECT ontology FROM projects WHERE id=?", (pid,)).fetchone()
    if not project:
        conn.close()
        return None
    rows = [dict(r) for r in conn.execute(
        "SELECT id, file_name, split, is_val, group_key FROM images "
        "WHERE project_id=? AND status='approved'", (pid,))]
    last = conn.execute(
        "SELECT MAX(train_images) m FROM models WHERE project_id=?", (pid,)).fetchone()["m"] or 0
    has_model = conn.execute(
        "SELECT 1 FROM models WHERE project_id=? LIMIT 1", (pid,)).fetchone() is not None
    assigned = {r["id"]: (r["split"] or ("val" if r["is_val"] else None)) for r in rows}
    groups = {r["id"]: _image_group(r) for r in rows}
    planned = plan_splits(assigned, groups) if assigned else {}
    split_counts = {s: sum(v == s for v in planned.values())
                    for s in ("train", "val", "test")}
    approved = len(rows)
    next_auto_at = max(MIN_APPROVED, last + RETRAIN_DELTA)
    ontology = json.loads(project["ontology"] or "[]")
    classes = ontology if isinstance(ontology, list) else ontology.get("classes", [])
    class_names = [c.get("name") for c in classes if isinstance(c, dict) and c.get("name")]
    coverage_rows = conn.execute(
        "SELECT a.class_name, COUNT(*) boxes, COUNT(DISTINCT a.image_id) images "
        "FROM annotations a JOIN images i ON i.id=a.image_id "
        "WHERE i.project_id=? AND i.status='approved' GROUP BY a.class_name", (pid,)).fetchall()
    coverage_found = {r["class_name"]: {"boxes": r["boxes"], "images": r["images"]}
                      for r in coverage_rows}
    class_coverage = {name: coverage_found.get(name, {"boxes": 0, "images": 0})
                      for name in class_names}
    conn.close()
    from server.train_worker import pick_arch

    blockers = []
    if approved < 4:
        blockers.append(f"수동 학습까지 승인 {4 - approved}장 더 필요")
    if split_counts["train"] == 0 and approved:
        blockers.append("학습용 train 분할이 비어 있음")
    evaluation_ready = (split_counts["val"] >= MIN_VAL
                        and split_counts["test"] >= MIN_TEST)
    sparse_classes = [name for name, c in class_coverage.items()
                      if c["images"] < MIN_CLASS_IMAGES]
    coverage_ready = bool(class_names) and not sparse_classes
    professional_ready = evaluation_ready and coverage_ready
    warnings = []
    if approved >= 4 and split_counts["test"] == 0:
        warnings.append("독립 홀드아웃(test)이 0장이라 실제 성능을 판정할 수 없습니다")
    elif split_counts["test"] < MIN_TEST:
        warnings.append(f"전문 평가용 홀드아웃이 {MIN_TEST - split_counts['test']}장 부족합니다")
    if approved >= 4 and split_counts["val"] < MIN_VAL:
        warnings.append(f"안정적인 검증셋이 {MIN_VAL - split_counts['val']}장 부족합니다")
    if sparse_classes:
        preview = ", ".join(sparse_classes[:4]) + (" 외" if len(sparse_classes) > 4 else "")
        warnings.append(f"클래스별 승인 다양성이 부족합니다: {preview} (각 {MIN_CLASS_IMAGES}장 권장)")
    stage = "collecting" if approved < 4 else ("validated" if professional_ready else "experiment")
    min_professional_approved = 2 * (MIN_VAL + MIN_TEST)
    return {
        "approved": approved,
        "last_trained": last,
        "new_since_last": max(0, approved - last),
        "min_manual": 4,
        "min_auto": MIN_APPROVED,
        "next_auto_at": next_auto_at,
        "remaining_auto": max(0, next_auto_at - approved),
        "ready_manual": approved >= 4 and split_counts["train"] > 0,
        "ready_auto": approved >= next_auto_at and split_counts["train"] > 0,
        "stage": stage,
        "evaluation_ready": evaluation_ready,
        "coverage_ready": coverage_ready,
        "professional_ready": professional_ready,
        "min_professional_approved": min_professional_approved,
        "remaining_professional": max(0, min_professional_approved - approved),
        "has_model": has_model,
        "recommended_arch": pick_arch(approved),
        "expected_epochs": 60 if split_counts["train"] < 100 else 100,
        "split_counts": split_counts,
        "class_count": len(classes),
        "class_coverage": class_coverage,
        "blockers": blockers,
        "warnings": warnings,
    }


def maybe_start_training(pid: int, force: bool = False, debounce: bool = False) -> dict:
    """승인 수 조건 충족 시 워커 프로세스 시작.

    debounce=True면 승인이 멈춘 뒤에 시작한다. 240장을 연속 승인하는 동안
    첫 몇 장만으로 학습이 시작돼 나머지가 반영되지 않던 문제를 막는다.
    """
    if debounce and not force:
        running = job_status(pid)
        if running.get("status") == "running":
            return running
        approved, last = _training_counts(pid)
        need = max(MIN_APPROVED, last + RETRAIN_DELTA)
        if approved < need:
            return {"status": "skipped", "approved": approved, "need": need}
        with _lock:
            t = _timers.get(pid)
            if t:
                t.cancel()
            timer = threading.Timer(DEBOUNCE_SEC, lambda: _run_scheduled(pid))
            timer.daemon = True
            _timers[pid] = timer
            timer.start()
        return {"status": "scheduled", "in_sec": DEBOUNCE_SEC}
    with _lock:
        if job_status(pid).get("status") == "running":
            return job_status(pid)
        approved, last = _training_counts(pid)
        if not force:
            if approved < MIN_APPROVED or approved < last + RETRAIN_DELTA:
                return {"status": "skipped", "approved": approved,
                        "need": max(MIN_APPROVED, last + RETRAIN_DELTA)}
        RUNS.mkdir(parents=True, exist_ok=True)
        started_at = time.time()
        _status_path(pid).write_text(json.dumps(
            {"status": "running", "phase": "starting", "approved": approved,
             "started_at": started_at, "updated_at": started_at,
             "elapsed_sec": 0, "progress": 0}))
        log = open(RUNS / f"train_log_{pid}.txt", "a")
        proc = subprocess.Popen(
            [sys.executable, "-m", "server.train_worker", str(pid)],
            # 데이터 저장 루트가 외장 디스크·임시 경로여도 Python 모듈은 코드
            # 루트에서 실행해야 한다. 둘을 같은 ROOT로 쓰면 워커가 server 모듈을
            # 못 찾아 즉시 종료한다.
            cwd=CODE_ROOT, stdout=log, stderr=log)
        _procs[pid] = proc
        # OS pid를 상태 파일에 남긴다. _procs는 인메모리라 서버가 재시작하면
        # 워커 생사를 알 수 없어 상태가 영원히 running으로 멈췄다.
        _status_path(pid).write_text(json.dumps(
            {"status": "running", "phase": "starting", "approved": approved,
             "pid_os": proc.pid, "started_at": started_at,
             "updated_at": time.time(), "elapsed_sec": 0, "progress": 0}))
        return {"status": "running", "phase": "starting", "approved": approved,
                "started_at": started_at, "elapsed_sec": 0, "progress": 0}


def active_model(pid: int) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM models WHERE project_id=? AND active=1", (pid,)).fetchone()
    conn.close()
    return row_to_dict(row) if row else None

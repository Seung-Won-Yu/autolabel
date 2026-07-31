"""자동 파인튜닝 루프: 승인 라벨 → YOLO 포맷 → 학습 → champion 게이트 → 등록.

학습 자체는 별도 프로세스(server.train_worker)에서 — 서버 재시작(--reload 포함)에도 생존.
상태는 data/runs/train_status_<pid>.json 파일 경유 (서버는 읽기 전용).

트레이너는 플러그인 개념 — 현재 YOLO11n(ultralytics, AGPL: 내부 도구 용도).
제품 배포 시 RF-DETR(Apache-2.0, CUDA) 트레이너로 교체 지점은 train_worker 하나.
"""
import json
import os
import random
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from server.db import get_db, row_to_dict

import os as _os

ROOT = Path(_os.environ.get("AUTOLABEL_DATA_ROOT") or Path(__file__).parent.parent)
RUNS = ROOT / "data" / "runs"
MIN_APPROVED = 8        # 자동 트리거 최소 승인 이미지 수
RETRAIN_DELTA = 5       # 마지막 학습 이후 신규 승인 N장마다 재학습
VAL_RATIO = 0.2
MIN_VAL = 30   # 이보다 작은 val은 모델 품질을 가리지 못한다 (실측: 12장 val이 오판)
TEST_RATIO = 0.15
MIN_TEST = 20  # 홀드아웃 — 학습·게이트에서 완전 배제, 진짜 성능 보고 전용

_lock = threading.Lock()
_procs: dict[int, subprocess.Popen] = {}
_timers: dict[int, threading.Timer] = {}
DEBOUNCE_SEC = 20  # 연속 승인이 끝나길 기다렸다 학습 (대량 승인 시 조기 발동 방지)


def _status_path(pid: int) -> Path:
    return RUNS / f"train_status_{pid}.json"


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
    """승인된 이미지만 YOLO 포맷으로 export. train/val 분할은 결정적(시드 고정).
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
    val_ids = {i for i, s in assigned.items() if s == "val"}
    test_ids = {i for i, s in assigned.items() if s == "test"}
    need_val = max(MIN_VAL, int(len(rows) * VAL_RATIO))
    need_test = max(MIN_TEST, int(len(rows) * TEST_RATIO))
    pool = [i for i, s in assigned.items() if s is None]
    random.Random(42).shuffle(pool)

    updates = []
    for target, ids, need in (("val", val_ids, need_val), ("test", test_ids, need_test)):
        while len(ids) < need and pool:
            i = pool.pop()
            ids.add(i)
            updates.append((target, i))
    # 기존 is_val 기반 멤버도 split 컬럼에 반영 (표시·조회 일관성)
    updates += [("val", i) for i in val_ids] + [("test", i) for i in test_ids]
    updates += [("train", i) for i in pool]
    if updates:
        conn.executemany("UPDATE images SET split=? WHERE id=?", updates)
        conn.execute("UPDATE images SET is_val=1 WHERE split='val' AND project_id=?", (pid,))
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
                ROOT / "data" / "uploads" / str(pid) / f"{im['id']}_{im['file_name']}")
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
    return len(splits["train"]) + len(splits["val"]), names


def maybe_start_training(pid: int, force: bool = False, debounce: bool = False) -> dict:
    """승인 수 조건 충족 시 워커 프로세스 시작.

    debounce=True면 승인이 멈춘 뒤에 시작한다. 240장을 연속 승인하는 동안
    첫 몇 장만으로 학습이 시작돼 나머지가 반영되지 않던 문제를 막는다.
    """
    if debounce and not force:
        with _lock:
            t = _timers.get(pid)
            if t:
                t.cancel()
            timer = threading.Timer(DEBOUNCE_SEC, lambda: maybe_start_training(pid))
            timer.daemon = True
            _timers[pid] = timer
            timer.start()
        return {"status": "scheduled", "in_sec": DEBOUNCE_SEC}
    with _lock:
        if job_status(pid).get("status") == "running":
            return job_status(pid)
        conn = get_db()
        approved = conn.execute(
            "SELECT COUNT(*) c FROM images WHERE project_id=? AND status='approved'",
            (pid,)).fetchone()["c"]
        last = conn.execute(
            "SELECT MAX(train_images) m FROM models WHERE project_id=?", (pid,)).fetchone()["m"] or 0
        conn.close()
        if not force:
            if approved < MIN_APPROVED or approved < last + RETRAIN_DELTA:
                return {"status": "skipped", "approved": approved,
                        "need": max(MIN_APPROVED, last + RETRAIN_DELTA)}
        RUNS.mkdir(parents=True, exist_ok=True)
        _status_path(pid).write_text(json.dumps(
            {"status": "running", "phase": "starting", "approved": approved}))
        log = open(RUNS / f"train_log_{pid}.txt", "a")
        proc = subprocess.Popen(
            [sys.executable, "-m", "server.train_worker", str(pid)],
            cwd=ROOT, stdout=log, stderr=log)
        _procs[pid] = proc
        # OS pid를 상태 파일에 남긴다. _procs는 인메모리라 서버가 재시작하면
        # 워커 생사를 알 수 없어 상태가 영원히 running으로 멈췄다.
        _status_path(pid).write_text(json.dumps(
            {"status": "running", "phase": "starting", "approved": approved,
             "pid_os": proc.pid}))
        return {"status": "running", "phase": "starting", "approved": approved}


def active_model(pid: int) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM models WHERE project_id=? AND active=1", (pid,)).fetchone()
    conn.close()
    return row_to_dict(row) if row else None

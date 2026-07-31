"""학습 워커 — 서버와 분리된 프로세스로 실행 (서버 재시작에도 생존).

실행: python -m server.train_worker <project_id>
상태는 data/runs/train_status_<pid>.json 에 기록, 서버는 읽기만 한다.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "data" / "runs"


def pick_arch(n_images: int, override: str | None = None) -> str:
    """데이터 규모별 학생 모델 크기 자동 선택.

    소량에 큰 모델은 과적합·시간 낭비, 대량에 작은 모델은 성능 손해.
    프로젝트에서 override를 지정하면 그것을 따른다.
    """
    if override:
        return override
    if n_images < 150:
        return "yolo11n"
    if n_images < 800:
        return "yolo11s"
    return "yolo11m"


def write_status(pid: int, **kw):
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"train_status_{pid}.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(kw)
    path.write_text(json.dumps(cur, ensure_ascii=False))


def main(pid: int):
    from server.db import get_db
    from server.train import _export_yolo_dataset

    write_status(pid, status="running", phase="export", pid_os=None)
    try:
        run_dir = RUNS / f"p{pid}_{int(time.time())}"
        n_images, names = _export_yolo_dataset(pid, run_dir / "dataset")
        if n_images < 4:
            write_status(pid, status="failed", error=f"승인 이미지 부족 ({n_images}장)")
            return
        write_status(pid, phase="training", images=n_images)

        from ultralytics import YOLO

        small = n_images < 100
        # 데이터가 많을수록 큰 모델이 이득 — 승인 장수로 자동 승급
        # (n=2.6M · s=9.4M · m=20M 파라미터. 로컬 MPS 학습 시간과 정확도의 균형)
        # 프로젝트 설정에 arch 지정이 있으면 우선
        from server.db import get_db as _gdb

        _c = _gdb()
        _p = _c.execute("SELECT ontology FROM projects WHERE id=?", (pid,)).fetchone()
        _c.close()
        override = None
        try:
            _o = json.loads(_p["ontology"])
            if isinstance(_o, dict):
                override = _o.get("arch")
        except Exception:
            pass
        arch = pick_arch(n_images, override)
        write_status(pid, arch=arch)
        model = YOLO(f"{arch}.pt")
        model.train(
            data=str(run_dir / "dataset" / "data.yaml"),
            epochs=60 if small else 100,
            patience=10,
            freeze=10 if small else None,
            mosaic=0.0 if small else 1.0,
            imgsz=640, device="mps", batch=8 if arch == "yolo11n" else 4,
            project=str(run_dir), name="train", verbose=False, plots=False,
        )
        best = run_dir / "train" / "weights" / "best.pt"
        metrics = YOLO(str(best)).val(
            data=str(run_dir / "dataset" / "data.yaml"), device="mps", verbose=False)
        map50 = float(metrics.box.map50)
        write_status(pid, phase="gating", map50=round(map50, 4))

        # champion/challenger + 품질 하한선 (콜드스타트 쓰레기 모델 승격 방지)
        QUALITY_FLOOR = 0.3
        conn = get_db()
        champ = conn.execute(
            "SELECT * FROM models WHERE project_id=? AND active=1", (pid,)).fetchone()
        promote = map50 >= QUALITY_FLOOR and (champ is None or map50 >= (champ["map50"] or 0))
        if promote:
            conn.execute("UPDATE models SET active=0 WHERE project_id=?", (pid,))
        conn.execute(
            "INSERT INTO models (project_id, path, map50, train_images, active, meta) "
            "VALUES (?,?,?,?,?,?)",
            (pid, str(best), map50, n_images, int(promote),
             json.dumps({"arch": arch, "names": names})))
        conn.commit()
        conn.close()
        write_status(pid, status="completed", promoted=promote)
    except Exception as e:
        write_status(pid, status="failed", error=str(e))


if __name__ == "__main__":
    main(int(sys.argv[1]))

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
    # 실측(PCB 300장): yolo11n·640이 yolo11s·960보다 홀드아웃에서 크게 우세했다.
    # 로컬 학습에서는 큰 모델·고해상도가 배치와 수렴을 희생시켜 손해인 경우가 많아
    # 승급 기준을 보수적으로 잡는다. 확신이 있으면 프로젝트 설정으로 올릴 것.
    if n_images < 800:
        return "yolo11n"
    if n_images < 3000:
        return "yolo11s"
    return "yolo11m"


def pick_imgsz(dataset_dir, n_images: int = 0, default: int = 640) -> int:
    """라벨 크기 분포 + 데이터 양으로 학습 해상도 결정.

    작은 객체가 지배적이면 해상도를 올리는 게 정석이지만, 실측 결과 소량
    데이터에서는 역효과였다 (PCB 60장·960px → val 0.44 vs 640px 0.74).
    고해상도는 배치가 줄고 수렴이 느려 소량에서 손해다. 따라서 데이터가
    충분할 때만 올린다.
    """
    import statistics
    from pathlib import Path as _P

    # 실측에서 해상도 상향이 일관되게 손해였다 (60장 0.74→0.44, 300장에서도 열세).
    # 데이터가 아주 많을 때만, 그것도 객체가 극히 작을 때만 올린다.
    if n_images < 1000:
        return default
    ratios = []
    for txt in list((_P(dataset_dir) / "labels" / "train").glob("*.txt"))[:300]:
        for line in txt.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                # YOLO 정규화 좌표라 그대로 이미지 대비 비율
                ratios.append((float(p[3]) * float(p[4])) ** 0.5)
    if not ratios:
        return default
    med = statistics.median(ratios)
    if med < 0.04:
        return 1280
    if med < 0.08:
        return 960
    return default


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
        imgsz = pick_imgsz(run_dir / "dataset", n_images)
        write_status(pid, arch=arch, imgsz=imgsz)
        model = YOLO(f"{arch}.pt")
        model.train(
            data=str(run_dir / "dataset" / "data.yaml"),
            epochs=60 if small else 100,
            patience=10,
            freeze=10 if small else None,
            mosaic=0.0 if small else 1.0,
            imgsz=imgsz, device="mps",
            batch=(8 if arch == "yolo11n" else 4) // (2 if imgsz > 800 else 1) or 1,
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

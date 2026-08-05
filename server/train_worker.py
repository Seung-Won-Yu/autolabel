"""학습 워커 — 서버와 분리된 프로세스로 실행 (서버 재시작에도 생존).

실행: python -m server.train_worker <project_id>
상태는 data/runs/train_status_<pid>.json 에 기록, 서버는 읽기만 한다.
"""
import json
import os
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(os.environ.get("AUTOLABEL_DATA_ROOT") or CODE_ROOT)
RUNS = DATA_ROOT / "data" / "runs"


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


def operational_calibration(metrics, names: list[str]) -> tuple[dict[str, float], float]:
    """val의 F1-confidence 곡선에서 클래스별 실제 추론 임계값을 고른다.

    AP는 예측의 순위가 좋으면 confidence가 전부 0.01대여도 높게 나올 수 있다.
    그 체크포인트를 온톨로지 기본값 0.3으로 배포하면 검출은 0건이다. 모델마다
    confidence 스케일이 다르므로 운영 F1과 그 최적점을 별도로 기록한다.
    """
    box = getattr(metrics, "box", None)
    raw_curves = getattr(box, "f1_curve", None)
    raw_px = getattr(box, "px", None)
    raw_class_ids = getattr(box, "ap_class_index", None)
    curves = [] if raw_curves is None else list(raw_curves)
    px = [] if raw_px is None else list(raw_px)
    class_ids = list(range(len(curves))) if raw_class_ids is None else list(raw_class_ids)
    thresholds: dict[str, float] = {}
    f1s: list[float] = []
    for row_i, curve in enumerate(curves):
        values = [float(v) for v in curve]
        if not values or not px:
            continue
        best_i = max(range(len(values)), key=values.__getitem__)
        cls_i = int(class_ids[row_i]) if row_i < len(class_ids) else row_i
        if not 0 <= cls_i < len(names):
            continue
        thresholds[names[cls_i]] = round(max(0.001, min(float(px[best_i]), 0.95)), 4)
        f1s.append(values[best_i])
    return thresholds, (sum(f1s) / len(f1s) if f1s else 0.0)


def choose_deploy_checkpoint(best_f1: float, last_f1: float,
                             margin: float = 0.01) -> str:
    """mAP용 best가 실제 임계값에서 약하면 더 잘 수렴한 last를 배포한다."""
    return "last" if last_f1 > best_f1 + margin else "best"


def epoch_progress(epoch_index: int, total_epochs: int, started_at: float,
                   now: float | None = None) -> dict:
    """0-based epoch를 사용자용 진행률과 ETA로 변환한다.

    첫 epoch 전에는 샘플이 없어 ETA를 계산하지 않는다. 한 epoch라도 끝나면
    현재까지의 평균 속도로 남은 시간을 추정한다. UI는 이 값을 그대로 보여
    주므로, 가짜 퍼센트나 고정 시간을 만들지 않는다.
    """
    total = max(1, int(total_epochs))
    done = max(0, min(int(epoch_index) + 1, total))
    elapsed = max(0.0, float(now if now is not None else time.time()) - started_at)
    eta = round(elapsed / done * (total - done)) if done else None
    return {
        "epoch": done,
        "epochs": total,
        "progress": round(done / total, 4),
        "elapsed_sec": round(elapsed),
        "eta_sec": eta,
    }


def write_status(pid: int, **kw):
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"train_status_{pid}.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    # pid_os는 생사 판정 키다. None으로 병합하면 런처가 남긴 pid가 지워져
    # 서버 재시작 후 살아있는 학습을 failed로 오보하고, 재시도가 워커 2개를
    # 겹치게 한다 (실측). None은 버리고 기존 값을 지킨다.
    if kw.get("pid_os") is None:
        kw.pop("pid_os", None)
    kw.setdefault("updated_at", time.time())
    cur.update(kw)
    path.write_text(json.dumps(cur, ensure_ascii=False))


def should_promote(challenger_map50: float, champion_map50: float | None,
                   champion_required: bool, champion_eval_ok: bool,
                   floor: float = 0.3) -> bool:
    """같은 val에서 비교 가능한 경우에만 challenger를 승격한다."""
    if challenger_map50 < floor:
        return False
    if not champion_required:
        return True
    return champion_eval_ok and champion_map50 is not None and challenger_map50 >= champion_map50


def main(pid: int):
    from server.db import get_db
    from server.train import _export_yolo_dataset

    job_started = time.time()
    write_status(pid, status="running", phase="export", started_at=job_started,
                 elapsed_sec=0, progress=0)
    try:
        run_dir = RUNS / f"p{pid}_{int(time.time())}"
        n_images, names = _export_yolo_dataset(pid, run_dir / "dataset")
        n_train = len(list((run_dir / "dataset" / "images" / "train").glob("*")))
        if n_images < 4 or n_train == 0:
            write_status(pid, status="failed",
                         error=f"학습 데이터 부족 (train {n_train}장 / 전체 {n_images}장)",
                         finished_at=time.time(), eta_sec=None,
                         elapsed_sec=round(time.time() - job_started))
            return
        write_status(pid, phase="training", images=n_images)

        from ultralytics import YOLO

        # 증강/동결 정책은 실제 optimizer가 보는 train 장수 기준이다. 반환
        # n_images는 재학습 bookkeeping을 위해 test까지 포함한다.
        small = n_train < 100
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
        epochs = 60 if small else 100
        training_started = time.time()
        write_status(pid, arch=arch, imgsz=imgsz, epoch=0, epochs=epochs,
                     progress=0, training_started_at=training_started,
                     elapsed_sec=round(training_started - job_started), eta_sec=None)
        model = YOLO(f"{arch}.pt")

        def report_epoch(trainer):
            """Ultralytics 실제 epoch 종료 이벤트를 상태 파일로 전달한다."""
            try:
                total = int(getattr(trainer, "epochs", epochs) or epochs)
                update = epoch_progress(int(getattr(trainer, "epoch", -1)), total,
                                        training_started)
                # 전체 작업 경과 시간은 export까지 포함하고, ETA는 학습 구간만
                # 평균낸다. metrics는 숫자인 것만 작은 스냅샷으로 남긴다.
                update["elapsed_sec"] = round(time.time() - job_started)
                live_metrics = {}
                for key, value in (getattr(trainer, "metrics", {}) or {}).items():
                    try:
                        live_metrics[str(key)] = round(float(value), 4)
                    except (TypeError, ValueError):
                        continue
                write_status(pid, phase="training", live_metrics=live_metrics, **update)
            except Exception:
                # 진행 표시 실패가 실제 학습까지 중단시키면 안 된다.
                pass

        model.add_callback("on_fit_epoch_end", report_epoch)
        model.train(
            data=str(run_dir / "dataset" / "data.yaml"),
            epochs=epochs,
            patience=10,
            freeze=10 if small else None,
            mosaic=0.0 if small else 1.0,
            imgsz=imgsz, device="mps",
            batch=(8 if arch == "yolo11n" else 4) // (2 if imgsz > 800 else 1) or 1,
            project=str(run_dir), name="train", verbose=False, plots=False,
        )
        best = run_dir / "train" / "weights" / "best.pt"
        last = run_dir / "train" / "weights" / "last.pt"
        trained = YOLO(str(best))
        data_yaml = str(run_dir / "dataset" / "data.yaml")
        write_status(pid, phase="validation", progress=1, eta_sec=None,
                     elapsed_sec=round(time.time() - job_started))
        metrics = trained.val(data=data_yaml, device="mps", verbose=False)
        calibrated_thresholds, operational_f1 = operational_calibration(metrics, names)

        # Ultralytics best는 mAP50-95 중심이다. 실제 사고에서는 epoch 2 best가
        # AP 0.86이면서 모든 confidence가 0.01대라 기본 임계값에서 0건이었고,
        # epoch 12 last는 새 이미지 20장 중 17장을 찾았다. 둘을 동일 val에서
        # 운영 F1로 비교해 실제 서비스에 쓸 체크포인트를 따로 선택한다.
        deploy_path = best
        checkpoint = "best"
        if last.exists():
            last_model = YOLO(str(last))
            last_metrics = last_model.val(data=data_yaml, device="mps", verbose=False)
            last_thresholds, last_operational_f1 = operational_calibration(last_metrics, names)
            if choose_deploy_checkpoint(operational_f1, last_operational_f1) == "last":
                deploy_path, checkpoint = last, "last"
                trained, metrics = last_model, last_metrics
                calibrated_thresholds, operational_f1 = last_thresholds, last_operational_f1
        map50 = float(metrics.box.map50)
        # 홀드아웃(test)은 학습·게이트 어디에도 안 쓴 데이터 — 진짜 성능
        test_map50 = None
        try:
            if list((run_dir / "dataset" / "images" / "test").glob("*")):
                write_status(pid, phase="holdout",
                             elapsed_sec=round(time.time() - job_started))
                t = trained.val(data=data_yaml, split="test", device="mps", verbose=False)
                test_map50 = round(float(t.box.map50), 4)
        except Exception:
            pass
        write_status(pid, phase="gating", map50=round(map50, 4), test_map50=test_map50,
                     operational_f1=round(operational_f1, 4), checkpoint=checkpoint,
                     calibrated_thresholds=calibrated_thresholds,
                     elapsed_sec=round(time.time() - job_started))

        # champion/challenger + 품질 하한선 (콜드스타트 쓰레기 모델 승격 방지)
        QUALITY_FLOOR = 0.3
        conn = get_db()
        champ = conn.execute(
            "SELECT * FROM models WHERE project_id=? AND active=1", (pid,)).fetchone()
        # val 셋은 승인 데이터가 늘며 보강될 수 있다. 과거 val에서 저장한 champion
        # 점수와 현재 val의 challenger 점수를 비교하면 서로 다른 시험지 비교다.
        # 활성 champion도 현재 라운드의 정확히 같은 val에서 다시 평가한다.
        champion_map50 = None
        champion_eval_error = None
        champion_required = champ is not None and Path(champ["path"]).exists()
        if champion_required:
            try:
                champion = YOLO(champ["path"])
                cm = champion.val(data=data_yaml, device="mps", verbose=False)
                champion_map50 = float(cm.box.map50)
            except Exception as e:
                champion_eval_error = str(e)
        elif champ is not None:
            champion_eval_error = f"활성 champion 파일 없음: {champ['path']}"
        promote = operational_f1 >= 0.3 and should_promote(
            map50, champion_map50, champion_required,
            champion_eval_ok=champion_map50 is not None, floor=QUALITY_FLOOR)
        val_images = len(list((run_dir / "dataset" / "images" / "val").glob("*")))
        write_status(pid, champion_map50=(round(champion_map50, 4)
                                          if champion_map50 is not None else None),
                     champion_eval_error=champion_eval_error, val_images=val_images)
        if promote:
            conn.execute("UPDATE models SET active=0 WHERE project_id=?", (pid,))
        conn.execute(
            "INSERT INTO models (project_id, path, map50, test_map50, train_images, "
            "active, meta) VALUES (?,?,?,?,?,?,?)",
            (pid, str(deploy_path), map50, test_map50, n_images, int(promote),
             json.dumps({"arch": arch, "names": names, "imgsz": imgsz,
                         "checkpoint": checkpoint,
                         "operational_f1": round(operational_f1, 4),
                         "calibrated_thresholds": calibrated_thresholds,
                         "gate_val_images": val_images,
                         "gate_champion_map50": champion_map50,
                         "gate_champion_error": champion_eval_error})))
        conn.commit()
        conn.close()
        write_status(pid, status="completed", phase="completed", promoted=promote,
                     progress=1, eta_sec=0, finished_at=time.time(),
                     elapsed_sec=round(time.time() - job_started))
    except Exception as e:
        write_status(pid, status="failed", error=str(e), finished_at=time.time(),
                     eta_sec=None, elapsed_sec=round(time.time() - job_started))


if __name__ == "__main__":
    main(int(sys.argv[1]))

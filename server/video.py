"""비디오 트래킹 레인: 업로드된 비디오 → 프레임 추출 → SAM 3 전파 트래킹.

프레임은 일반 이미지로 등록된다 — 리뷰 큐·심판·학습·익스포트 등 기존 레인을
전부 재사용하기 위해서다. 트래킹은 SAM 3 비디오 시맨틱 예측기(텍스트 프롬프트
→ 검출 + 메모리 전파 + 트랙 정합)를 쓰고, 결과는 프레임별 prelabel로 저장된다.
같은 객체가 프레임을 넘어 이어졌다는 정보는 meta.track_id로 남긴다.

가중치(models/sam3.pt)가 없으면 프레임 등록까지만 하고 정직하게 알린다 —
그 상태에서도 배치 오토라벨(GDINO 폴백)로 라벨링은 가능하다.
"""
import json
import os
import threading
from pathlib import Path

from server import jobs
from server.db import get_db

_jobs: dict[int, dict] = {}
_start_lock = threading.Lock()

MAX_FRAMES_DEFAULT = 300  # 30fps 영상 기준 stride 5로 약 50초 — 그 이상은 나눠서


def job_status(pid: int) -> dict:
    return _jobs.get(pid) or jobs.get("video", pid)


def extract_frames(video_path: Path, pid: int, data_dir: Path,
                   stride: int, max_frames: int) -> list[int]:
    """stride 간격 프레임을 이미지로 등록한다. 등록된 image id 목록 반환.

    프레임 인덱스 규칙(0, stride, 2*stride…)은 ultralytics vid_stride와 같아
    트래킹 결과 k번째와 여기 k번째가 1:1로 맞는다.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("비디오를 열 수 없습니다 — 파일/코덱을 확인하세요")
    conn = get_db()
    pdir = data_dir / str(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem.removeprefix("src_")  # 저장 시 붙인 충돌 방지 프리픽스 제거
    ids: list[int] = []
    idx = 0
    try:
        while len(ids) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                h, w = frame.shape[:2]
                fname = f"{stem}_f{idx:06d}.jpg"
                cur = conn.execute(
                    "INSERT INTO images (project_id, file_name, width, height) "
                    "VALUES (?,?,?,?)", (pid, fname, w, h))
                iid = cur.lastrowid
                cv2.imwrite(str(pdir / f"{iid}_{fname}"), frame)
                ids.append(iid)
            idx += 1
        conn.commit()
    finally:
        conn.close()
        cap.release()
    return ids


def _track(pid: int, video_path: Path, frame_ids: list[int],
           ontology: list[dict], stride: int, job: dict) -> int:
    """SAM 3 비디오 전파 트래킹 → 프레임별 어노테이션 저장. 박스 수 반환."""
    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    from server.ml import SAM3_PATH

    prompts = [c.get("prompt") or c["name"] for c in ontology]
    name_of = {(c.get("prompt") or c["name"]): c["name"] for c in ontology}
    thresholds = {c["name"]: float(c.get("threshold", 0.35)) for c in ontology}
    predictor = SAM3VideoSemanticPredictor(overrides={
        "conf": 0.25, "task": "segment", "mode": "predict", "model": SAM3_PATH,
        "save": False, "verbose": False, "vid_stride": stride})
    results = predictor(source=str(video_path), text=prompts, stream=True)

    conn = get_db()
    n_boxes = 0
    try:
        for k, r in enumerate(results):
            if k >= len(frame_ids):
                break  # max_frames 초과분 — 추출 안 한 프레임의 결과는 버린다
            iid = frame_ids[k]
            names = getattr(r, "names", None)
            labels = (dict(enumerate(names)) if isinstance(names, (list, tuple))
                      else (names or {}))
            dets = 0
            for b in (getattr(r, "boxes", None) or []):
                raw = labels.get(int(b.cls), prompts[0]) if labels else prompts[0]
                cls = name_of.get(raw, raw)
                conf = float(b.conf)
                if conf < thresholds.get(cls, 0.35):
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                tid = getattr(b, "id", None)
                meta = {"track_id": int(tid)} if tid is not None else {}
                conn.execute(
                    "INSERT INTO annotations (image_id, class_name, bbox, "
                    "confidence, source, meta) VALUES (?,?,?,?,?,?)",
                    (iid, cls,
                     json.dumps([round(x1, 1), round(y1, 1),
                                 round(x2 - x1, 1), round(y2 - y1, 1)]),
                     round(conf, 4), "model", json.dumps(meta)))
                dets += 1
            if dets:
                conn.execute("UPDATE images SET status='prelabeled' WHERE id=?", (iid,))
            n_boxes += dets
            conn.commit()  # 프레임마다 커밋 — 긴 쓰기 트랜잭션 금지 (심판 사고 참조)
            job.update(done=k + 1)
            if (k + 1) % 10 == 0:
                jobs.update("video", pid, done=k + 1)
    finally:
        conn.close()
    return n_boxes


def _run(pid: int, video_path: Path, ontology: list[dict], data_dir: Path,
         stride: int, max_frames: int):
    job = _jobs[pid]
    try:
        frame_ids = extract_frames(video_path, pid, data_dir, stride, max_frames)
        if not frame_ids:
            raise ValueError("추출된 프레임이 없습니다")
        job.update(total=len(frame_ids), phase="track")
        jobs.update("video", pid, total=len(frame_ids))

        from server import ml
        if os.environ.get("AUTOLABEL_NO_MODELS") or not ml.sam3_available():
            advice = (f"프레임 {len(frame_ids)}장 등록 (트래킹 생략 — "
                      + ("모델 비활성 모드" if os.environ.get("AUTOLABEL_NO_MODELS")
                         else "models/sam3.pt 없음, 배치 오토라벨을 대신 쓰세요") + ")")
            job.update(status="completed", done=len(frame_ids), advice=advice)
            jobs.update("video", pid, status="completed", done=len(frame_ids),
                        advice=advice)
            return

        n_boxes = _track(pid, video_path, frame_ids, ontology, stride, job)
        advice = (f"프레임 {len(frame_ids)}장 · 트래킹 박스 {n_boxes}개 — "
                  "리뷰를 시작하세요 (같은 객체는 track_id로 이어져 있습니다)")
        job.update(status="completed", advice=advice)
        jobs.update("video", pid, status="completed", done=len(frame_ids),
                    advice=advice)
    except Exception as e:
        job.update(status="failed", error=str(e))
        jobs.update("video", pid, status="failed", error=str(e))


def start(pid: int, video_path: Path, ontology: list[dict], data_dir: Path,
          stride: int = 5, max_frames: int = MAX_FRAMES_DEFAULT) -> dict:
    stride = max(1, int(stride))
    max_frames = max(1, min(int(max_frames), 2000))
    with _start_lock:
        if _jobs.get(pid, {}).get("status") == "running":
            return _jobs[pid]
        _jobs[pid] = {"status": "running", "phase": "extract", "done": 0, "total": 0}
        jobs.start("video", pid, done=0, total=0)
        threading.Thread(
            target=_run, args=(pid, video_path, ontology, data_dir, stride, max_frames),
            daemon=True).start()
    return _jobs[pid]

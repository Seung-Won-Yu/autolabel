"""비디오 트래킹 레인: 업로드된 비디오 → 프레임 추출 → SAM 3 전파 트래킹.

프레임은 일반 이미지로 등록된다 — 리뷰 큐·심판·학습·익스포트 등 기존 레인을
전부 재사용하기 위해서다. 트래킹은 SAM 3 비디오 시맨틱 예측기(텍스트 프롬프트
→ 검출 + 메모리 전파 + 트랙 정합)를 쓰고, 결과는 프레임별 prelabel로 저장된다.
같은 객체가 프레임을 넘어 이어졌다는 정보는 meta.track_id로 남긴다.

가중치(models/sam3.pt)가 없으면 프레임 등록까지만 하고 정직하게 알린다 —
그 상태에서도 배치 오토라벨(GDINO 폴백)로 라벨링은 가능하다.

잡 상태는 jobs 모듈 단일 경로 — 모듈 로컬 사본을 두면 두 저장소가 드리프트한다
(실측: 심판의 박스 진행률이 로컬에만 기록돼 재시작 복원에서 빠졌다).
"""
import json
import os
import threading
from pathlib import Path

from server import jobs
from server.db import get_db

_start_lock = threading.Lock()

MAX_FRAMES_DEFAULT = 300  # 30fps 영상 기준 stride 5로 약 50초 — 그 이상은 나눠서
EXTRACT_COMMIT_EVERY = 25  # 추출 내내 쓰기 트랜잭션을 쥐지 않게 주기 커밋


def job_status(pid: int) -> dict:
    return jobs.get("video", pid)


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
    stem = video_path.stem.removeprefix("src_")  # 저장 시 붙인 충돌 방지 프리픽스 제거
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
                # 추출 전체를 한 트랜잭션으로 쥐면 그동안 다른 쓰기(2초 자동
                # 저장 등)가 busy_timeout을 넘겨 죽는다 — 주기적으로 놓아준다
                if len(ids) % EXTRACT_COMMIT_EVERY == 0:
                    conn.commit()
            idx += 1
        conn.commit()
    finally:
        conn.close()
        cap.release()
    return ids


def _track(pid: int, video_path: Path, frame_ids: list[int],
           ontology: list[dict], stride: int) -> int:
    """SAM 3 비디오 전파 트래킹 → 프레임별 어노테이션 저장. 박스 수 반환."""
    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    from server import ml

    prompts = [c.get("prompt") or c["name"] for c in ontology]
    predictor = SAM3VideoSemanticPredictor(
        overrides={**ml.SAM3_OVERRIDES, "vid_stride": stride})
    results = predictor(source=str(video_path), text=prompts, stream=True)

    conn = get_db()
    n_boxes = 0
    try:
        for k, r in enumerate(results):
            if k >= len(frame_ids):
                break  # max_frames 초과분 — 추출 안 한 프레임의 결과는 버린다
            iid = frame_ids[k]
            dets = ml.sam3_result_to_dets(r, ontology, with_track=True)
            for d in dets:
                meta = ({"track_id": d["track_id"]} if "track_id" in d else {})
                conn.execute(
                    "INSERT INTO annotations (image_id, class_name, bbox, "
                    "confidence, source, meta) VALUES (?,?,?,?,?,?)",
                    (iid, d["class_name"], json.dumps(d["bbox"]),
                     d["confidence"], "model", json.dumps(meta)))
            if dets:
                conn.execute("UPDATE images SET status='prelabeled' WHERE id=?", (iid,))
            n_boxes += len(dets)
            conn.commit()  # 프레임마다 커밋 — 긴 쓰기 트랜잭션 금지 (심판 사고 참조)
            jobs.update("video", pid, done=k + 1)
    finally:
        conn.close()
    return n_boxes


def _run(pid: int, video_path: Path, ontology: list[dict], data_dir: Path,
         stride: int, max_frames: int):
    try:
        frame_ids = extract_frames(video_path, pid, data_dir, stride, max_frames)
        if not frame_ids:
            raise ValueError("추출된 프레임이 없습니다")
        jobs.update("video", pid, total=len(frame_ids), phase="track")

        from server import ml
        if os.environ.get("AUTOLABEL_NO_MODELS") or not ml.sam3_available():
            advice = (f"프레임 {len(frame_ids)}장 등록 (트래킹 생략 — "
                      + ("모델 비활성 모드" if os.environ.get("AUTOLABEL_NO_MODELS")
                         else "models/sam3.pt 없음, 배치 오토라벨을 대신 쓰세요") + ")")
            jobs.update("video", pid, status="completed", done=len(frame_ids),
                        advice=advice)
            return

        n_boxes = _track(pid, video_path, frame_ids, ontology, stride)
        advice = (f"프레임 {len(frame_ids)}장 · 트래킹 박스 {n_boxes}개 — "
                  "리뷰를 시작하세요 (같은 객체는 track_id로 이어져 있습니다)")
        jobs.update("video", pid, status="completed", done=len(frame_ids),
                    advice=advice)
    except Exception as e:
        jobs.update("video", pid, status="failed", error=str(e))


def start(pid: int, video_path: Path, ontology: list[dict], data_dir: Path,
          stride: int = 5, max_frames: int = MAX_FRAMES_DEFAULT) -> dict:
    stride = max(1, int(stride))
    max_frames = max(1, min(int(max_frames), 2000))
    with _start_lock:  # 더블클릭·중복 탭이 같은 비디오를 두 번 처리하지 않게
        st = jobs.get("video", pid)
        if st.get("status") == "running":
            return st
        jobs.start("video", pid, done=0, total=0, phase="extract")
        threading.Thread(
            target=_run, args=(pid, video_path, ontology, data_dir, stride, max_frames),
            daemon=True).start()
    return jobs.get("video", pid)

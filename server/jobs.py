"""백그라운드 잡 상태를 디스크에 남긴다.

배치 오토라벨·임포트·심판은 서버 프로세스 안의 스레드로 돈다. 상태를 메모리에만
두면 서버가 재시작하는 순간 기록이 사라지고, 프론트는 그걸 "완료"로 읽어
"배치 오토라벨 완료: undefined/undefined장"을 띄웠다 — 절반만 라벨된 데이터를
두고 사용자는 끝난 줄 안다.

학습(train_worker)은 별도 프로세스라 원래 상태 파일을 쓴다. 여기서는 같은
방식을 인프로세스 잡에도 적용하고, 기동 시 running으로 남은 기록을
interrupted로 정리한다 — 그 스레드는 이전 프로세스와 함께 죽었다.
"""
import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
JOBS_DIR = Path(os.environ.get("AUTOLABEL_DATA") or (ROOT / "data" / "uploads")).parent / "jobs"

_lock = threading.Lock()
_cache: dict[str, dict] = {}
_last_write: dict[str, float] = {}
# 비종결 update의 디스크 쓰기 간격 — 박스·프레임마다 fsync하지 않게.
# 캐시는 항상 최신이라 상태 API는 정확하고, 스로틀로 잃는 것은 크래시 직전
# 0.5초치 진행 숫자뿐이다 (재시작 시 어차피 interrupted로 정리된다).
WRITE_INTERVAL = 0.5


def _path(kind: str, pid: int) -> Path:
    return JOBS_DIR / f"{kind}_{pid}.json"


def _key(kind: str, pid: int) -> str:
    return f"{kind}_{pid}"


def _write(kind: str, pid: int, state: dict) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(kind, pid).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False))
    tmp.replace(_path(kind, pid))  # 원자적 교체 — 반쯤 쓰인 파일을 읽지 않게


def start(kind: str, pid: int, **fields) -> dict:
    with _lock:
        state = {"status": "running", **fields}
        _cache[_key(kind, pid)] = state
        _write(kind, pid, state)
        _last_write[_key(kind, pid)] = time.monotonic()
        return dict(state)


def try_start(kind: str, pid: int, **fields) -> tuple[bool, dict]:
    """원자적 검사-등록 — 이미 실행 중이면 (False, 현재 상태).

    검사와 등록이 분리돼 있으면 더블클릭·중복 탭이 검사를 동시에 통과해
    같은 배치가 두 번 돌아 비용이 2배가 된다. 모듈마다 자기 락으로 감싸던
    것을 여기로 일반화.
    """
    with _lock:
        key = _key(kind, pid)
        cur = _cache.get(key)
        if cur and cur.get("status") == "running":
            return False, dict(cur)
        state = {"status": "running", **fields}
        _cache[key] = state
        _write(kind, pid, state)
        _last_write[key] = time.monotonic()
        return True, dict(state)


def update(kind: str, pid: int, **fields) -> dict:
    with _lock:
        key = _key(kind, pid)
        state = _cache.setdefault(key, {"status": "running"})
        state.update(fields)
        # 상태 전이(완료·실패 등)는 즉시 기록, 진행 카운터는 스로틀
        now = time.monotonic()
        if (state.get("status") != "running"
                or now - _last_write.get(key, 0.0) >= WRITE_INTERVAL):
            _write(kind, pid, state)
            _last_write[key] = now
        return dict(state)


def get(kind: str, pid: int) -> dict:
    with _lock:
        cached = _cache.get(_key(kind, pid))
        if cached:
            return dict(cached)
    path = _path(kind, pid)
    if not path.exists():
        return {"status": "idle"}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"status": "idle"}


def sweep_stale() -> int:
    """기동 시 호출 — running으로 남은 인프로세스 잡을 interrupted로 정리한다.

    그 스레드는 이전 프로세스와 함께 죽었으므로 절대 완료되지 않는다.
    """
    if not JOBS_DIR.exists():
        return 0
    n = 0
    for path in JOBS_DIR.glob("*.json"):
        try:
            state = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("status") != "running":
            continue
        state["status"] = "interrupted"
        state["error"] = "서버가 재시작되어 작업이 중단됐습니다 — 다시 실행하세요"
        try:
            path.write_text(json.dumps(state, ensure_ascii=False))
            n += 1
        except OSError:
            pass
    return n

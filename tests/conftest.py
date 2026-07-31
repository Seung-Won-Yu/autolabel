"""테스트 공통 설정 — 임시 DB로 격리해 실제 데이터를 건드리지 않는다."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def isolated_db():
    """서버 모듈이 로드되기 전에 DB 경로를 임시 파일로 바꾼다."""
    tmp = Path(tempfile.mkdtemp(prefix="autolabel-test-"))
    os.environ["AUTOLABEL_DB"] = str(tmp / "test.db")
    os.environ["AUTOLABEL_DATA"] = str(tmp / "data")
    yield tmp


@pytest.fixture
def client(isolated_db):
    from fastapi.testclient import TestClient

    from server.main import app

    return TestClient(app)


@pytest.fixture
def make_image():
    """테스트용 이미지 파일 생성기."""
    from PIL import Image

    def _make(dirpath: Path, name: str, size=(640, 480), color=(120, 120, 120)):
        dirpath.mkdir(parents=True, exist_ok=True)
        p = dirpath / name
        Image.new("RGB", size, color).save(p)
        return p

    return _make

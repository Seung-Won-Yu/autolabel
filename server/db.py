"""SQLite 스키마 + 커넥션. MVP는 단일 사용자 로컬 — 마이그레이션 도구 없이 idempotent DDL."""
import json
import os
import sqlite3
from pathlib import Path

# 테스트는 AUTOLABEL_DB로 임시 경로를 지정해 실제 데이터를 건드리지 않는다
DB_PATH = Path(os.environ.get("AUTOLABEL_DB")
               or Path(__file__).parent.parent / "autolabel.db")
# 없는 디렉터리를 가리켜도 떠야 한다 — 안 그러면 서버가 조용히 기동에 실패한다
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DDL = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    -- 온톨로지: [{name, prompt, threshold, color}] — 프롬프트↔클래스 분리(Autodistill 패턴)
    ontology TEXT NOT NULL DEFAULT '[]',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    file_name TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    -- 리뷰 상태: unlabeled | prelabeled | approved | rejected
    status TEXT NOT NULL DEFAULT 'unlabeled',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id),
    class_name TEXT NOT NULL,
    -- bbox: [x, y, w, h] 이미지 픽셀 좌표
    bbox TEXT NOT NULL,
    -- 마스크: COCO RLE JSON (없으면 NULL)
    segmentation TEXT,
    confidence REAL,
    -- 계층 라벨(객체 안의 객체) 대비
    parent_annotation_id INTEGER REFERENCES annotations(id),
    -- 출처: model | human. 재현성 메타(모델명·프롬프트·임계값)는 meta JSON에
    source TEXT NOT NULL DEFAULT 'model',
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id, status);
CREATE INDEX IF NOT EXISTS idx_ann_image ON annotations(image_id);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    path TEXT NOT NULL,
    -- champion/challenger 게이트용 지표 (val split 기준)
    map50 REAL,
    train_images INTEGER,
    -- active: 현재 라벨 어시스트로 쓰는 champion 여부
    active INTEGER NOT NULL DEFAULT 0,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript(DDL)
    # 증분 마이그레이션 (idempotent)
    for stmt in (
        "ALTER TABLE images ADD COLUMN is_val INTEGER NOT NULL DEFAULT 0",  # 고정 골드 val
        "ALTER TABLE images ADD COLUMN qa_score REAL",  # 라벨 의심 점수 (높을수록 의심)
        # 외부 폴더 연결: 복사 없이 원본 경로 참조 (대용량 데이터셋용)
        "ALTER TABLE images ADD COLUMN src_path TEXT",
        # 3분할: train | val(게이트) | test(홀드아웃 — 학습·게이트에서 완전 배제)
        "ALTER TABLE images ADD COLUMN split TEXT",
        # 라운드별 실성능 기록 (게이트용 val이 아니라 홀드아웃 기준)
        "ALTER TABLE models ADD COLUMN test_map50 REAL",
        # VLM 문맥 심판용 판정 기준 문서 (예: '사고 연루 차량만 accident_vehicle')
        "ALTER TABLE projects ADD COLUMN rubric TEXT NOT NULL DEFAULT ''",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # 이미 적용됨
    conn.commit()
    conn.close()


def row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in ("ontology", "bbox", "segmentation", "meta"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except (json.JSONDecodeError, TypeError):
                pass
    return d

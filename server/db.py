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

-- 사람이 삭제·수정한 제로샷 후보도 나중에 엔진 품질 계산에 써야 한다.
-- annotations는 현재 정답만 유지하므로, "그때 어떤 엔진을 실제로 돌렸는지"와
-- 원본 후보를 별도 감사 로그로 남긴다. 이미지별 최신 실행만 보존한다.
CREATE TABLE IF NOT EXISTS foundation_audits (
    image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    sam3_ran INTEGER NOT NULL DEFAULT 0,
    gdino_ran INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS foundation_candidates (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    engine TEXT NOT NULL CHECK(engine IN ('sam3', 'gdino')),
    class_name TEXT NOT NULL,
    bbox TEXT NOT NULL,
    confidence REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_foundation_audit_project
    ON foundation_audits(project_id, sam3_ran, gdino_ran);
CREATE INDEX IF NOT EXISTS idx_foundation_candidates_image
    ON foundation_candidates(image_id, engine, class_name);
"""


def get_db() -> sqlite3.Connection:
    # timeout: 다른 커넥션이 쓰기 트랜잭션을 쥐고 있으면 즉시 "database is
    # locked"로 죽지 않고 기다린다 (실측: 심판이 판정을 기록하는 동안
    # 프로젝트 생성이 500). WAL 전환은 배타 접근이 필요해 다른 커넥션이
    # 활동 중이면 그 자체가 잠긴다 — init_db에서 1회만 설정한다.
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def init_db():
    conn = get_db()
    # WAL: 읽기가 쓰기에 안 막힌다. 영속 설정이라 시작 시 1회면 충분 —
    # get_db마다 실행하면 다른 커넥션 활동 중 전환이 잠겨 오히려 죽는다
    conn.execute("PRAGMA journal_mode = WAL")
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
        # 같은 영상·연속 촬영 묶음이 train/val/test에 갈라지는 평가 누출 방지
        "ALTER TABLE images ADD COLUMN group_key TEXT",
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

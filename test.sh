#!/usr/bin/env bash
# 회귀 테스트 — 코드를 고친 뒤 항상 이걸 돌린다 (1초 안에 끝난다)
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python -m pytest tests/ -q "$@"

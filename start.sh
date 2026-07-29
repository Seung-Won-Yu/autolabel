#!/usr/bin/env bash
# 원커맨드 실행: 백엔드 + 프론트 동시 기동, 종료는 Ctrl+C 한 번
set -euo pipefail
cd "$(dirname "$0")"

.venv/bin/python -m uvicorn server.main:app --port 8899 &
BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT

( sleep 2 && open http://localhost:5173 ) &
cd webapp && npm run dev

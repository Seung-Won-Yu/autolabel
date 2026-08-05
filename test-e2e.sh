#!/usr/bin/env bash
# 프론트 회귀 테스트 (Playwright) — 백엔드·프론트를 격리 DB로 자동 기동한다.
# 모델 추론이 필요한 흐름은 scripts/qa_*_e2e.py 담당.
set -euo pipefail
cd "$(dirname "$0")/webapp"
npm run test:unit
npx playwright test "$@"

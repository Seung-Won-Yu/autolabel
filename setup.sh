#!/usr/bin/env bash
# 최초 1회 셋업: 가상환경 + 의존성 + SAM 가중치 + 브라우저 디코더
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] Python 가상환경"
if command -v uv >/dev/null; then
  uv venv --python 3.11 2>/dev/null || true
  uv pip install -r requirements.txt
else
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

echo "[2/4] SAM ViT-L 인코더 (1.2GB — 최초 1회)"
mkdir -p models
[ -f models/sam_vit_l_0b3195.pth ] || curl -L -o models/sam_vit_l_0b3195.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

echo "[3/4] 브라우저 SAM 디코더"
mkdir -p webapp/public
[ -f webapp/public/sam_decoder.onnx ] || cp release/sam_decoder_vit_l.onnx webapp/public/sam_decoder.onnx

echo "[4/4] 프론트 의존성"
cd webapp && npm install --silent

echo "✅ 셋업 완료 — ./start.sh 로 실행"

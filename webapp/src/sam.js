// SAM 브라우저 디코더 — Phase 0에서 검증된 패턴 이식.
// 핵심 교훈 반영: 원본 크기 복원은 ONNX 그래프가 아니라 JS에서(세로 이미지 crop 버그 회피),
// 이미지 전환 시 임베딩 무효화(레이스 차단).
import * as ort from 'onnxruntime-web'
import { api } from './api'

// wasmPaths 미설정 — Vite 번들러 모드에서 ORT가 자체 경로로 wasm 로드 (버전 불일치·public 임포트 문제 회피)

let session = null
let embedState = null // { imageId, tensor, origSize, scale }
let loadSeq = 0

export async function loadDecoder() {
  if (!session) {
    session = await ort.InferenceSession.create('/sam_decoder.onnx', {
      executionProviders: ['wasm'],
    })
  }
  return session
}

export function resetEmbed() {
  embedState = null
  loadSeq++
}

export async function ensureEmbed(imageId) {
  if (embedState?.imageId === imageId) return embedState
  const seq = ++loadSeq
  embedState = null
  const j = await api.embed(imageId)
  if (seq !== loadSeq) return null // 그 사이 다른 이미지 로드 — 폐기
  const bytes = Uint8Array.from(atob(j.embedding), (c) => c.charCodeAt(0))
  embedState = {
    imageId,
    tensor: new ort.Tensor('float32', new Float32Array(bytes.buffer), j.shape),
    origSize: j.orig_size, // [H, W]
    scale: 1024 / Math.max(...j.orig_size),
    encodeMs: j.encode_ms,
  }
  return embedState
}

// points: [{x, y, label}] 이미지 픽셀 좌표. 반환: { mask: Uint8Array(W*H), iou }
export async function decodeMask(imageId, points) {
  const emb = await ensureEmbed(imageId)
  if (!emb) return null
  await loadDecoder()
  const n = points.length
  const coords = new Float32Array((n + 1) * 2)
  const labels = new Float32Array(n + 1)
  points.forEach((p, i) => {
    coords[i * 2] = p.x * emb.scale
    coords[i * 2 + 1] = p.y * emb.scale
    labels[i] = p.label
  })
  coords[n * 2] = 0; coords[n * 2 + 1] = 0; labels[n] = -1 // 패딩 포인트

  const out = await session.run({
    image_embeddings: emb.tensor,
    point_coords: new ort.Tensor('float32', coords, [1, n + 1, 2]),
    point_labels: new ort.Tensor('float32', labels, [1, n + 1]),
    mask_input: new ort.Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]),
    has_mask_input: new ort.Tensor('float32', [0], [1]),
    orig_im_size: new ort.Tensor('float32', [emb.origSize[0], emb.origSize[1]], [2]),
  })
  return {
    mask: lowResToMask(out.low_res_masks.data, emb.origSize, emb.scale),
    iou: out.iou_predictions.data[0],
  }
}

// 256x256 로짓(1024 패딩 프레임) → 원본 크기 이진 마스크. 종횡비 무관.
function lowResToMask(low, [H, W], scale) {
  const s = document.createElement('canvas')
  s.width = s.height = 256
  const sctx = s.getContext('2d')
  const id = sctx.createImageData(256, 256)
  for (let i = 0; i < 256 * 256; i++) {
    id.data[i * 4 + 3] = Math.max(0, Math.min(255, low[i] * 32 + 128))
    id.data[i * 4] = 255
  }
  sctx.putImageData(id, 0, 0)
  const validW = Math.round((256 * (W * scale)) / 1024)
  const validH = Math.round((256 * (H * scale)) / 1024)
  const d = document.createElement('canvas')
  d.width = W; d.height = H
  const dctx = d.getContext('2d')
  dctx.imageSmoothingEnabled = true
  dctx.imageSmoothingQuality = 'high'
  dctx.drawImage(s, 0, 0, validW, validH, 0, 0, W, H)
  const px = dctx.getImageData(0, 0, W, H).data
  const m = new Uint8Array(W * H)
  for (let i = 0; i < W * H; i++) m[i] = px[i * 4 + 3] >= 128 ? 1 : 0
  return m
}

// 마스크 → 타이트 bbox [x, y, w, h]
export function maskToBbox(mask, W, H) {
  let x1 = W, y1 = H, x2 = -1, y2 = -1
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (mask[y * W + x]) {
        if (x < x1) x1 = x
        if (x > x2) x2 = x
        if (y < y1) y1 = y
        if (y > y2) y2 = y
      }
    }
  }
  if (x2 < 0) return null
  return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
}

// 마스크 → COCO 비압축 RLE (Fortran/열 우선, 0부터 시작하는 교대 카운트)
export function maskToRLE(mask, W, H) {
  const counts = []
  let cur = 0, run = 0
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) {
      const v = mask[y * W + x]
      if (v === cur) run++
      else { counts.push(run); cur = v; run = 1 }
    }
  }
  counts.push(run)
  return { counts, size: [H, W] }
}

// COCO 비압축 RLE → 마스크 (기존 어노테이션 렌더용)
export function rleToMask(rle) {
  const [H, W] = rle.size
  const m = new Uint8Array(W * H)
  let cur = 0, pos = 0
  for (const run of rle.counts) {
    if (cur === 1) {
      for (let k = 0; k < run; k++) {
        const p = pos + k
        m[(p % H) * W + Math.floor(p / H)] = 1
      }
    }
    pos += run
    cur = 1 - cur
  }
  return m
}

// 마스크 → 색칠된 오프스크린 캔버스 (Konva Image 소스)
export function maskToCanvas(mask, W, H, colorHex, alpha = 110) {
  const c = document.createElement('canvas')
  c.width = W; c.height = H
  const ctx = c.getContext('2d')
  const id = ctx.createImageData(W, H)
  const r = parseInt(colorHex.slice(1, 3), 16)
  const g = parseInt(colorHex.slice(3, 5), 16)
  const b = parseInt(colorHex.slice(5, 7), 16)
  for (let i = 0; i < W * H; i++) {
    if (mask[i]) {
      id.data[i * 4] = r; id.data[i * 4 + 1] = g
      id.data[i * 4 + 2] = b; id.data[i * 4 + 3] = alpha
    }
  }
  ctx.putImageData(id, 0, 0)
  return c
}

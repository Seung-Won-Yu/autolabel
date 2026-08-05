// COCO RLE 변환. 브라우저 SAM은 비압축 counts 배열을 만들고, pycocotools는
// 압축 counts 문자열을 만든다. 저장 출처와 무관하게 둘 다 읽어야 한다.

export function decodeRleCounts(counts) {
  if (Array.isArray(counts)) return counts
  if (typeof counts !== 'string') throw new Error('RLE counts 형식 오류')

  // pycocotools maskApi.c의 rleFrString과 같은 가변 길이 5-bit 디코딩.
  const out = []
  let p = 0
  while (p < counts.length) {
    let x = 0
    let k = 0
    let more
    do {
      const c = counts.charCodeAt(p++) - 48
      if (c < 0 || c > 63) throw new Error('압축 RLE 문자 오류')
      x |= (c & 0x1f) << (5 * k)
      more = c & 0x20
      k++
      if (!more && (c & 0x10)) x |= -1 << (5 * k)
    } while (more && p < counts.length)
    if (more) throw new Error('잘린 압축 RLE')
    if (out.length > 2) x += out[out.length - 2]
    if (!Number.isInteger(x) || x < 0) throw new Error('압축 RLE run 오류')
    out.push(x)
  }
  return out
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

// COCO 압축/비압축 RLE → 행 우선 브라우저 마스크.
export function rleToMask(rle) {
  const [H, W] = rle.size
  const total = W * H
  const m = new Uint8Array(total)
  const counts = decodeRleCounts(rle.counts)
  let cur = 0, pos = 0
  for (const run of counts) {
    if (!Number.isInteger(run) || run < 0 || pos + run > total) {
      throw new Error('RLE run 범위 오류')
    }
    if (cur === 1) {
      for (let p = pos; p < pos + run; p++) {
        m[(p % H) * W + Math.floor(p / H)] = 1
      }
    }
    pos += run
    cur = 1 - cur
  }
  if (pos !== total) throw new Error('RLE 크기가 이미지와 다릅니다')
  return m
}

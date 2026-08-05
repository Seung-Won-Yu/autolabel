import test from 'node:test'
import assert from 'node:assert/strict'

import { decodeRleCounts, maskToRLE, rleToMask } from '../src/rle.js'

test('pycocotools 압축 RLE 문자열을 브라우저 마스크로 복원한다', () => {
  // 4x5 마스크의 rows 1..2, cols 2..3이 1. pycocotools encode 결과.
  const rle = { size: [4, 5], counts: '92203' }
  assert.deepEqual(decodeRleCounts(rle.counts), [9, 2, 2, 2, 5])
  const mask = rleToMask(rle)
  const on = [...mask].flatMap((v, i) => (v ? [i] : []))
  assert.deepEqual(on, [7, 8, 12, 13])
})

test('브라우저 비압축 RLE는 왕복 보존된다', () => {
  const W = 5, H = 4
  const mask = new Uint8Array(W * H)
  for (const i of [1, 6, 13, 18]) mask[i] = 1
  assert.deepEqual(rleToMask(maskToRLE(mask, W, H)), mask)
})

test('이미지 크기를 넘는 손상 RLE를 거부한다', () => {
  assert.throws(() => rleToMask({ size: [2, 2], counts: [5] }), /범위 오류/)
})

import test from 'node:test'
import assert from 'node:assert/strict'

import { ensembleReviewSummary, mergeAutolabelDrafts } from '../src/autolabel.js'

test('재실행은 낡은 모델 초안을 교체하고 사람 라벨을 보존한다', () => {
  const existing = [
    { class_name: 'defect', bbox: [0, 0, 20, 20], source: 'human' },
    { class_name: 'defect', bbox: [50, 50, 20, 20], source: 'model' },
  ]
  const detected = [
    { class_name: 'defect', bbox: [1, 1, 18, 18], source: 'model' },
    { class_name: 'defect', bbox: [80, 80, 20, 20], source: 'model' },
    // 다른 클래스는 계층 부위나 실제 중첩 객체일 수 있으므로 살아야 한다.
    { class_name: 'defect.part', bbox: [2, 2, 5, 5], source: 'model' },
  ]
  const result = mergeAutolabelDrafts(existing, detected)

  assert.equal(result.replaced, 1)
  assert.equal(result.suppressed, 1)
  assert.deepEqual(result.annotations.map((a) => a.class_name),
    ['defect', 'defect', 'defect.part'])
  assert.deepEqual(result.annotations[1].bbox, [80, 80, 20, 20])
})

test('앙상블 후보는 합의와 우선 검수 수를 분리해 센다', () => {
  const anns = [
    { meta: { ensemble: { agreement: 'consensus' } } },
    { meta: { ensemble: { agreement: 'sam3_only' } } },
    { meta: { ensemble: { agreement: 'gdino_only' } } },
    { source: 'human' },
  ]
  assert.deepEqual(ensembleReviewSummary(anns), {
    consensus: 1, sam3_only: 1, gdino_only: 1, total: 3, reviewFirst: 2,
  })
})

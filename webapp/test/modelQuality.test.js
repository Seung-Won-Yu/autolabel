import test from 'node:test'
import assert from 'node:assert/strict'

import { modelImprovementPlan, modelQuality, rankedClassMetrics, readinessLabel } from '../src/modelQuality.js'

test('성능표 없는 활성 모델은 가동 중이 아니라 검증 필요로 표시한다', () => {
  const quality = modelQuality({ active: 1, map50: null, meta: { imported: true } })
  assert.equal(quality.status, 'unverified')
  assert.equal(quality.tone, 'warn')
  assert.equal(quality.usable, false)
})

test('홀드아웃과 validation 모델의 품질 단계를 구분한다', () => {
  assert.equal(modelQuality({ test_map50: 0.5, map50: 0.6 }).status, 'verified')
  assert.equal(modelQuality({ test_map50: null, map50: 0.6 }).status, 'provisional')
  assert.equal(modelQuality({ map50: 0.01 }).status, 'failed')
})

test('학습 가능과 전문 평가 가능을 같은 상태로 말하지 않는다', () => {
  assert.equal(readinessLabel({ ready_manual: true, professional_ready: false }), '실험 학습 가능')
  assert.equal(readinessLabel({ ready_manual: true, professional_ready: true }), '전문 평가 가능')
  assert.equal(readinessLabel({ ready_manual: false }), '라벨 수집 중')
})

test('클래스 성능은 취약 클래스부터 보여준다', () => {
  const rows = rankedClassMetrics({
    scratches: { test_map50: 0.619, test_instances: 11 },
    crazing: { test_map50: 0.064, test_instances: 8 },
    invalid: { test_map50: null, test_instances: 3 },
  })
  assert.deepEqual(rows.map((row) => row.name), ['crazing', 'scratches'])
  assert.equal(rows[0].test_instances, 8)
})

test('품질 하한 미달과 평가 근거 부족을 서로 다른 처방으로 만든다', () => {
  const plan = modelImprovementPlan({
    meta: {
      quality_status: 'failed',
      quality_reason: 'crazing 성능 미달',
      class_metrics: {
        scratches: { test_map50: 0.619, test_instances: 11 },
        crazing: { test_map50: 0.064, test_instances: 8 },
        pitted_surface: { test_map50: 0.695, test_instances: 2 },
      },
    },
  })
  assert.equal(plan.tone, 'bad')
  assert.equal(plan.summary, 'crazing 성능 미달')
  assert.deepEqual(plan.actions.map((action) => [action.className, action.kind]), [
    ['crazing', 'critical'],
    ['pitted_surface', 'evidence'],
  ])
  assert.equal(plan.actions[0].recommendedImages, 20)
  assert.equal(plan.actions[1].reserveImages, 4)
})

test('사용 가능한 모델도 중간 성능 클래스는 다음 개선 대상으로 안내한다', () => {
  const plan = modelImprovementPlan({
    test_map50: 0.52,
    meta: {
      quality_status: 'verified',
      class_metrics: {
        dent: { test_map50: 0.22, test_instances: 8 },
        scratch: { test_map50: 0.71, test_instances: 12 },
      },
    },
  })
  assert.equal(plan.tone, 'warn')
  assert.equal(plan.actions[0].kind, 'improve')
  assert.equal(plan.actions[0].recommendedImages, 12)
})

test('취약 클래스가 없는 검증 모델은 운영 표본 검수로 이어진다', () => {
  const plan = modelImprovementPlan({
    meta: {
      quality_status: 'verified',
      class_metrics: {
        dent: { test_map50: 0.68, test_instances: 9 },
        scratch: { test_map50: 0.71, test_instances: 12 },
      },
    },
  })
  assert.equal(plan.tone, 'ok')
  assert.equal(plan.actions.length, 0)
  assert.match(plan.nextStep, /표본 검수/)
})

test('클래스별 지표가 없는 미검증 후보는 전체 평가 재실행을 안내한다', () => {
  const plan = modelImprovementPlan({ meta: { quality_status: 'unverified' } })
  assert.equal(plan.actions[0].kind, 'overall')
  assert.match(plan.actions[0].guidance, /클래스별 test 지표/)
})

test('시험 결함 1개의 낮은 점수는 성능 실패 대신 근거 부족으로만 안내한다', () => {
  const plan = modelImprovementPlan({
    meta: {
      quality_status: 'verified',
      class_metrics: { rare_defect: { test_map50: 0.02, test_instances: 1 } },
    },
  })
  assert.equal(plan.actions.length, 1)
  assert.equal(plan.actions[0].kind, 'evidence')
  assert.equal(plan.tone, 'warn')
})

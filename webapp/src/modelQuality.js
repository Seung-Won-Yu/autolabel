export const MODEL_QUALITY = {
  verified: { label: '전문 검증 통과', tone: 'ok', usable: true },
  provisional: { label: '실험 모델', tone: 'warn', usable: true },
  failed: { label: '품질 미달', tone: 'bad', usable: false },
  unverified: { label: '검증 필요', tone: 'warn', usable: false },
}

export function modelQuality(model) {
  if (!model) return MODEL_QUALITY.unverified
  let status = model.meta?.quality_status
  if (!MODEL_QUALITY[status]) {
    const score = model.test_map50 ?? model.map50
    status = score == null ? 'unverified' : score >= 0.3
      ? (model.test_map50 != null ? 'verified' : 'provisional') : 'failed'
  }
  return { status, ...MODEL_QUALITY[status] }
}

export function readinessLabel(readiness) {
  if (readiness?.professional_ready) return '전문 평가 가능'
  if (readiness?.ready_manual) return '실험 학습 가능'
  return '라벨 수집 중'
}

export function rankedClassMetrics(classMetrics) {
  if (!classMetrics || typeof classMetrics !== 'object') return []
  return Object.entries(classMetrics)
    .filter(([, row]) => Number.isFinite(row?.test_map50))
    .map(([name, row]) => ({ name, ...row }))
    .sort((a, b) => a.test_map50 - b.test_map50 || a.name.localeCompare(b.name))
}

const CLASS_GATE_FLOOR = 0.1
const CLASS_IMPROVEMENT_TARGET = 0.3
const MIN_CLASS_EVIDENCE = 4

// 점수만 보여주면 비전문 사용자는 다음 행동을 결정할 수 없다. 최신 후보의
// 홀드아웃 결과를 "어느 클래스를, 왜, 어느 정도부터 보강할지"로 번역한다.
// 권장 장수는 성능 보장값이 아니라 한 번의 재학습 효과를 확인할 시작 묶음이다.
export function modelImprovementPlan(model) {
  if (!model) return null
  const quality = modelQuality(model)
  const meta = model.meta || {}
  const classMetrics = meta.class_metrics || model.class_metrics || {}
  const rows = Object.entries(classMetrics).map(([name, row = {}]) => ({
    name,
    test_map50: Number.isFinite(row.test_map50) ? row.test_map50 : null,
    test_instances: Number.isFinite(row.test_instances) ? row.test_instances : 0,
  }))
  const actions = []

  for (const row of rows) {
    // 결함 1개로 나온 낮은 점수는 성능 실패의 근거로 삼지 않는다. 먼저
    // 평가 표본을 늘리고, 게이트가 유효해지는 2개 이상부터 성능을 판정한다.
    if (row.test_instances < 2 || row.test_map50 == null) {
      actions.push({
        className: row.name,
        kind: 'evidence',
        score: row.test_map50,
        instances: row.test_instances,
        recommendedImages: 12,
        reserveImages: 4,
        reason: '평가 근거 부족',
        guidance: '새 사례를 모으되 최소 4장은 다음 홀드아웃용으로 따로 남겨 두세요.',
      })
    } else if (row.test_map50 < CLASS_GATE_FLOOR) {
      actions.push({
        className: row.name,
        kind: 'critical',
        score: row.test_map50,
        instances: row.test_instances,
        recommendedImages: 20,
        reason: '품질 하한 미달',
        guidance: '누락·경계 오류를 먼저 고치고 크기·조명·표면 상태가 다른 사례를 추가하세요.',
      })
    } else if (row.test_map50 != null && row.test_map50 < CLASS_IMPROVEMENT_TARGET) {
      actions.push({
        className: row.name,
        kind: 'improve',
        score: row.test_map50,
        instances: row.test_instances,
        recommendedImages: 12,
        reason: '개선 여지 큼',
        guidance: '현재 모델이 헷갈린 모양과 배경을 우선 수집해 승인 라벨로 추가하세요.',
      })
    } else if (row.test_instances < MIN_CLASS_EVIDENCE) {
      actions.push({
        className: row.name,
        kind: 'evidence',
        score: row.test_map50,
        instances: row.test_instances,
        recommendedImages: 12,
        reserveImages: 4,
        reason: '평가 근거 부족',
        guidance: '새 사례를 모으되 최소 4장은 다음 홀드아웃용으로 따로 남겨 두세요.',
      })
    }
  }

  const hasClassMetrics = rows.length > 0
  const overallTest = meta.metrics?.test_map50 ?? model.metrics?.test_map50 ?? model.test_map50
  if (!hasClassMetrics && (!Number.isFinite(overallTest) || !quality.usable)) {
    actions.push({
      className: '전체 클래스',
      kind: 'overall',
      score: Number.isFinite(overallTest) ? overallTest : null,
      instances: null,
      recommendedImages: 20,
      reason: Number.isFinite(overallTest) ? '전체 성능 보강 필요' : '클래스별 평가 없음',
      guidance: '클래스별 test 지표가 포함된 번들로 다시 평가하고 오라벨과 누락 라벨을 먼저 점검하세요.',
    })
  }

  const priority = { critical: 0, overall: 1, improve: 2, evidence: 3 }
  actions.sort((a, b) => priority[a.kind] - priority[b.kind]
    || (a.score ?? Infinity) - (b.score ?? Infinity)
    || a.className.localeCompare(b.className))

  const blocking = actions.some((action) => ['critical', 'overall'].includes(action.kind)) || !quality.usable
  const improving = actions.some((action) => action.kind === 'improve')
  const evidence = actions.some((action) => action.kind === 'evidence')
  const tone = blocking ? 'bad' : actions.length ? 'warn' : 'ok'
  const title = blocking ? '재학습 전 보강 필요'
    : improving ? '사용 가능 · 다음 라운드 개선 권장'
      : evidence ? '사용 가능 · 평가 근거 보강 권장' : '도메인 모델 사용 준비 완료'
  const summary = blocking
    ? (meta.quality_reason || model.quality_reason || '최신 후보가 품질 기준을 통과하지 못했습니다.')
    : actions.length
      ? '현재 모델은 사용할 수 있지만 아래 항목을 보강하면 다음 라운드를 더 신뢰할 수 있습니다.'
      : '홀드아웃과 클래스별 품질 기준을 통과했습니다. 실제 오토라벨 오류율을 계속 기록하세요.'

  return {
    tone, title, summary, actions,
    qualityStatus: quality.status,
    holdoutNote: '기존 test 세트는 수정하거나 학습에 섞지 마세요. 새 데이터에서 다음 평가용 표본을 별도로 확보하세요.',
    nextStep: actions.length
      ? '라벨 보강 → 승인 → Colab 재학습 → 새 번들 검증'
      : '오토라벨 실행 → 표본 검수 → 클래스별 오류율 기록',
  }
}

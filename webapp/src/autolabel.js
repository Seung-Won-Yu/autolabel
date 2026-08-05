const overlap = (a, b) => {
  const [ax, ay, aw, ah] = a
  const [bx, by, bw, bh] = b
  const iw = Math.max(0, Math.min(ax + aw, bx + bw) - Math.max(ax, bx))
  const ih = Math.max(0, Math.min(ay + ah, by + bh) - Math.max(ay, by))
  const inter = iw * ih
  if (!inter) return 0
  const aa = aw * ah
  const ba = bw * bh
  return Math.max(inter / (aa + ba - inter), inter / aa, inter / ba)
}

export function mergeAutolabelDrafts(existing, detected, overlapThreshold = 0.3) {
  // 모델 초안 재실행은 교체 동작이다. 사람이 그리거나 고친 박스는 source가
  // human이므로 남기고, 같은 클래스·같은 객체로 겹치는 새 초안은 억제한다.
  const trusted = existing.filter((a) => a.source !== 'model')
  const fresh = detected.filter((d) => !trusted.some((t) =>
    d.class_name === t.class_name && overlap(d.bbox, t.bbox) >= overlapThreshold))
  return {
    annotations: [...trusted, ...fresh],
    replaced: existing.length - trusted.length,
    suppressed: detected.length - fresh.length,
  }
}

export function ensembleReviewSummary(annotations) {
  const counts = { consensus: 0, sam3_only: 0, gdino_only: 0 }
  for (const annotation of annotations) {
    const agreement = annotation.meta?.ensemble?.agreement
    if (agreement in counts) counts[agreement] += 1
  }
  return {
    ...counts,
    total: counts.consensus + counts.sam3_only + counts.gdino_only,
    reviewFirst: counts.sam3_only + counts.gdino_only,
  }
}

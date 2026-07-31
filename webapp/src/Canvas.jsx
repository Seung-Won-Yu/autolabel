// 어노테이션 캔버스: bbox 그리기/이동/리사이즈 + SAM 클릭 마스크 + 줌/팬.
// 좌표는 항상 "이미지 픽셀" 기준으로 저장 — 뷰 변환은 Stage scale/position만 담당.
import { useEffect, useMemo, useRef, useState } from 'react'
import { Stage, Layer, Image as KImage, Rect, Text, Group, Circle } from 'react-konva'
import useImage from 'use-image'
import { classColor } from './api'
import { decodeMask, maskToBbox, maskToRLE, rleToMask, maskToCanvas } from './sam'

export default function Canvas({
  imageUrl, imageId, tool, anns, setAnns, ontology, activeClass,
  selectedId, setSelectedId, hoverId = null, onMsg = () => {}, onExemplar = null,
  suggestions = [], onAcceptSuggestion = null,
  size = { w: 900, h: 640 },
}) {
  const [img] = useImage(imageUrl, 'anonymous')
  const stageRef = useRef()
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 })
  const [draft, setDraft] = useState(null) // 그리는 중인 박스
  const [sam, setSam] = useState({ points: [], mask: null, busy: false })

  // 이미지 로드 시 화면에 맞춤 + 상태 무효화 (이미지 전환 레이스 차단)
  useEffect(() => {
    if (!img) return
    const s = Math.min(size.w / img.width, size.h / img.height, 1)
    setView({ scale: s, x: (size.w - img.width * s) / 2, y: (size.h - img.height * s) / 2 })
    setDraft(null)
    setSam({ points: [], mask: null, busy: false })
    setSelectedId(null)
  }, [img]) // eslint-disable-line

  const toImageCoords = (pos) => ({
    x: (pos.x - view.x) / view.scale,
    y: (pos.y - view.y) / view.scale,
  })

  const onWheel = (e) => {
    e.evt.preventDefault()
    const oldScale = view.scale
    const pointer = stageRef.current.getPointerPosition()
    const factor = e.evt.deltaY > 0 ? 0.9 : 1.1
    const scale = Math.min(Math.max(oldScale * factor, 0.05), 20)
    setView({
      scale,
      x: pointer.x - ((pointer.x - view.x) / oldScale) * scale,
      y: pointer.y - ((pointer.y - view.y) / oldScale) * scale,
    })
  }

  // ---------- SAM 모드 ----------
  // 프레임을 거의 다 덮는 마스크는 배경(종이·벽·하늘)을 클릭한 것이다.
  // 라벨링에서 그건 언제나 오클릭인데, 예전엔 다음 클릭이 그걸 조용히 확정해
  // 이미지 전체 크기 어노테이션이 학습 데이터에 들어갔다 (실측: 1920x1079).
  // server/tiling.py의 MAX_FRAME_COVERAGE와 같은 기준.
  const MAX_FRAME_COVERAGE = 0.85

  const commitSam = (points, mask) => {
    if (!mask || !img) return []
    const bbox = maskToBbox(mask, img.width, img.height)
    if (!bbox) return []
    if (bbox[2] >= img.width * MAX_FRAME_COVERAGE
        && bbox[3] >= img.height * MAX_FRAME_COVERAGE) {
      onMsg('배경을 클릭한 것 같습니다 — 객체 위를 클릭하세요 (확정 안 함)')
      return []
    }
    return [{
      _key: `sam-${Date.now()}`,
      class_name: activeClass,
      bbox,
      segmentation: maskToRLE(mask, img.width, img.height),
      source: 'human', confidence: null,
    }]
  }

  const runSam = async (points, committed) => {
    setSam({ points, mask: null, busy: true })
    try {
      const r = await decodeMask(imageId, points)
      if (!r) return
      setSam({ points, mask: r.mask, busy: false })
      if (committed.length) setAnns([...anns, ...committed])
      onMsg(`SAM IoU ${r.iou.toFixed(2)} — 클릭=다음 객체 · Shift=정제 · 우클릭=제외 · Enter=확정`)
    } catch (e) {
      setSam({ points: [], mask: null, busy: false })
      onMsg(`SAM 오류: ${e.message}`)
    }
  }

  const onSamClick = (e, label) => {
    const p = toImageCoords(stageRef.current.getPointerPosition())
    const pt = { x: p.x, y: p.y, label }
    if (label === 0 && !sam.points.length) return // 제외 포인트는 정제 전용
    if (label === 1 && !e.evt.shiftKey && sam.points.length) {
      // 새 객체: 현재 마스크 확정 후 새 포인트로 시작
      const committed = commitSam(sam.points, sam.mask)
      setAnns([...anns, ...committed])
      runSam([pt], [])
    } else {
      runSam([...sam.points, pt], [])
    }
  }

  // Enter 확정 / Escape 취소
  useEffect(() => {
    const h = (e) => {
      if (tool !== 'sam') return
      if (e.key === 'Enter' && sam.mask) {
        setAnns([...anns, ...commitSam(sam.points, sam.mask)])
        setSam({ points: [], mask: null, busy: false })
      } else if (e.key === 'Escape') {
        setSam({ points: [], mask: null, busy: false })
      }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  })

  // ---------- 박스 모드 ----------
  const onMouseDown = (e) => {
    if (e.evt.altKey || e.evt.button === 1) return
    if (tool === 'sam') return // SAM은 click 핸들러에서
    if (tool !== 'ex' && e.target !== e.target.getStage() && e.target.name() !== 'bg') return
    const p = toImageCoords(stageRef.current.getPointerPosition())
    setDraft({ x: p.x, y: p.y, w: 0, h: 0 })
    setSelectedId(null)
  }
  const onMouseMove = () => {
    if (!draft) return
    const p = toImageCoords(stageRef.current.getPointerPosition())
    setDraft({ ...draft, w: p.x - draft.x, h: p.y - draft.y })
  }
  const onMouseUp = () => {
    if (!draft) return
    const { x, y, w, h } = draft
    setDraft(null)
    if (Math.abs(w) < 4 || Math.abs(h) < 4) return
    const bbox = [Math.min(x, x + w), Math.min(y, y + h), Math.abs(w), Math.abs(h)].map(r1)
    if (tool === 'ex') {
      // 예시 모드: 박스는 어노테이션이 아니라 "이런 거 다 찾아줘" 쿼리
      onExemplar?.(bbox)
      return
    }
    setAnns([...anns, {
      _key: `new-${Date.now()}`,
      class_name: activeClass,
      bbox,
      source: 'human', confidence: null,
    }])
  }

  const updateBox = (key, bbox) => {
    setAnns(anns.map((a) => (a._key === key ? { ...a, bbox, source: 'human' } : a)))
  }

  const activeColor = classColor(ontology, activeClass)

  return (
    <Stage
      ref={stageRef} width={size.w} height={size.h}
      scaleX={view.scale} scaleY={view.scale} x={view.x} y={view.y}
      onWheel={onWheel} onMouseDown={onMouseDown}
      onMouseMove={onMouseMove} onMouseUp={onMouseUp}
      onClick={(e) => {
        if (tool !== 'sam' || e.evt.button !== 0) return
        if (e.target === e.target.getStage() || e.target.name() === 'bg' || e.target.name() === 'mask') {
          onSamClick(e, 1)
        }
      }}
      onContextMenu={(e) => {
        e.evt.preventDefault()
        if (tool === 'sam') onSamClick(e, 0)
      }}
      style={{ background: '#181818', cursor: 'crosshair' }}
    >
      <Layer>
        {img && <KImage image={img} name="bg" />}
        {anns.map((a) => a.segmentation?.counts && (
          <MaskAnn key={`m-${a._key}`} ann={a} color={classColor(ontology, a.class_name)} />
        ))}
        {anns.map((a, i) => (
          <BoxAnn
            key={a._key} ann={a} index={i + 1}
            color={classColor(ontology, a.class_name)}
            selected={selectedId === a._key}
            highlighted={hoverId === a._key || selectedId === a._key}
            // 하나가 강조되면 나머지는 흐리게 — 밀집 장면에서 구분
            dimmed={(hoverId || selectedId) && hoverId !== a._key && selectedId !== a._key}
            // 라벨 텍스트: 붐비면(9개+) 강조된 것만 표시
            showLabel={anns.length < 9 || hoverId === a._key || selectedId === a._key}
            onSelect={() => setSelectedId(a._key)}
            onChange={(bbox) => updateBox(a._key, bbox)}
            scale={view.scale}
            interactive={tool === 'box'} // SAM 모드에선 클릭이 박스에 먹히지 않게
          />
        ))}
        {draft && (
          <Rect
            x={Math.min(draft.x, draft.x + draft.w)} y={Math.min(draft.y, draft.y + draft.h)}
            width={Math.abs(draft.w)} height={Math.abs(draft.h)}
            stroke="#fff" strokeWidth={1.5 / view.scale} dash={[6 / view.scale, 4 / view.scale]}
          />
        )}
        {/* 누락 의심 제안 — 점선 박스, 클릭하면 라벨로 승격 */}
        {suggestions.map((s, i) => {
          const [sx, sy, sw2, sh2] = s.bbox
          const c = classColor(ontology, s.class_name)
          return (
            <Group key={`sug-${i}`}>
              <Rect
                x={sx} y={sy} width={sw2} height={sh2}
                stroke={c} strokeWidth={2.5 / view.scale}
                dash={[8 / view.scale, 5 / view.scale]}
                fill={c + '18'}
                onClick={() => onAcceptSuggestion?.(s, i)}
                onMouseEnter={(e) => (e.target.getStage().container().style.cursor = 'copy')}
                onMouseLeave={(e) => (e.target.getStage().container().style.cursor = 'crosshair')}
              />
              <Text
                x={sx} y={sy - 17 / view.scale}
                text={`+ ${s.class_name} ${s.confidence?.toFixed(2) ?? ''} (클릭하여 추가)`}
                fill={c} fontSize={12 / view.scale} fontStyle="bold" listening={false}
              />
            </Group>
          )
        })}
        {tool === 'sam' && sam.mask && img && (
          <LiveMask mask={sam.mask} W={img.width} H={img.height} color={activeColor} />
        )}
        {tool === 'sam' && sam.points.map((p, i) => (
          <Circle key={i} x={p.x} y={p.y} radius={5 / view.scale}
            fill={p.label === 1 ? '#2ecc71' : '#e74c3c'}
            stroke="#fff" strokeWidth={1.5 / view.scale} listening={false} />
        ))}
      </Layer>
    </Stage>
  )
}

function MaskAnn({ ann, color }) {
  const canvas = useMemo(() => {
    const [H, W] = ann.segmentation.size
    return maskToCanvas(rleToMask(ann.segmentation), W, H, color)
  }, [ann.segmentation, color])
  return <KImage image={canvas} name="mask" listening={false} />
}

function LiveMask({ mask, W, H, color }) {
  const canvas = useMemo(() => maskToCanvas(mask, W, H, color, 150), [mask, W, H, color])
  return <KImage image={canvas} name="mask" listening={false} />
}

function BoxAnn({
  ann, index, color, selected, highlighted = false, dimmed = false,
  showLabel = true, onSelect, onChange, scale, interactive = true,
}) {
  const [x, y, w, h] = ann.bbox
  const sw = (highlighted ? 3 : 1.5) / scale
  const label = `#${index} ${ann.class_name}${ann.confidence != null ? ` ${ann.confidence.toFixed(2)}` : ''}`
  const HANDLE = 8 / scale

  return (
    <Group listening={interactive} opacity={dimmed ? 0.25 : 1}>
      <Rect
        x={x} y={y} width={w} height={h}
        stroke={color} strokeWidth={sw}
        fill={highlighted ? color + '33' : 'transparent'}
        draggable
        onClick={onSelect} onTap={onSelect}
        onDragEnd={(e) => onChange([e.target.x(), e.target.y(), w, h].map(r1))}
        onMouseEnter={(e) => (e.target.getStage().container().style.cursor = 'move')}
        onMouseLeave={(e) => (e.target.getStage().container().style.cursor = 'crosshair')}
      />
      {showLabel && (
        <Text
          x={x} y={y - 16 / scale} text={label} fill={color}
          fontSize={13 / scale} fontStyle="bold" listening={false}
        />
      )}
      {selected && (
        <Rect
          x={x + w - HANDLE / 2} y={y + h - HANDLE / 2} width={HANDLE} height={HANDLE}
          fill="#fff" stroke={color} strokeWidth={1 / scale}
          draggable
          onDragMove={(e) => {
            const nw = Math.max(4, e.target.x() + HANDLE / 2 - x)
            const nh = Math.max(4, e.target.y() + HANDLE / 2 - y)
            onChange([x, y, nw, nh].map(r1))
          }}
        />
      )}
    </Group>
  )
}

const r1 = (v) => Math.round(v * 10) / 10

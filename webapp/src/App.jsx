import { useCallback, useEffect, useRef, useState } from 'react'
import Canvas from './Canvas'
import { api, classColor } from './api'
import { ensureEmbed, resetEmbed, loadDecoder } from './sam'

export default function App() {
  const [projects, setProjects] = useState([])
  const [project, setProject] = useState(null)
  const [images, setImages] = useState([])
  const [current, setCurrent] = useState(null)
  const [anns, setAnns] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [hoverId, setHoverId] = useState(null) // 패널 행 ↔ 캔버스 박스 상호 하이라이트
  const [activeClass, setActiveClass] = useState('')
  const [job, setJob] = useState({ status: 'idle' })
  const [toast, setToast] = useState(null)
  const [tool, setTool] = useState('box')
  const [trainInfo, setTrainInfo] = useState({ job: { status: 'idle' }, active_model: null })
  const [showHelp, setShowHelp] = useState(false)
  const [filter, setFilter] = useState('all')
  const [sortMode, setSortMode] = useState('none')
  const [lastQa, setLastQa] = useState(null)
  const [qaJob, setQaJob] = useState(null)
  const [suggest, setSuggest] = useState([]) // 현재 이미지의 누락 라벨 제안
  const dirty = useRef(false)
  const undoStack = useRef([])

  // 화면에 보이는 목록(필터·정렬 적용) — 이동(←→)도 이 순서를 따른다
  let visible = filter === 'all' ? images : images.filter((im) => im.status === filter)
  if (sortMode === 'conf') visible = [...visible].sort((a, b) => (a.min_conf ?? 2) - (b.min_conf ?? 2))
  else if (sortMode === 'qa') visible = [...visible].sort((a, b) => (b.qa_score ?? -1) - (a.qa_score ?? -1))

  const setMsg = useCallback((text) => {
    setToast(text)
  }, [])

  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 3000)
    return () => clearTimeout(t)
  }, [toast])

  useEffect(() => { loadDecoder().catch(() => {}) }, [])

  useEffect(() => {
    if (!project) return
    const poll = () => api.trainStatus(project.id).then(setTrainInfo).catch(() => {})
    poll()
    const t = setInterval(poll, 5000)
    return () => clearInterval(t)
  }, [project?.id]) // eslint-disable-line

  const refreshProjects = () => api.listProjects().then(setProjects)
  useEffect(() => { refreshProjects() }, [])

  const openProject = async (p) => {
    const full = await api.getProject(p.id)
    setProject(full)
    setActiveClass(full.ontology[0]?.name || '')
    const imgs = await api.listImages(p.id)
    setImages(imgs)
    setCurrent(null); setAnns([])
    // 첫 이미지 자동 열기 — 빈 캔버스로 시작하지 않게
    if (imgs.length) {
      const first = imgs.find((im) => im.status === 'prelabeled')
        || imgs.find((im) => im.status === 'unlabeled') || imgs[0]
      const list = await api.getAnnotations(first.id)
      setCurrent(first)
      setAnns(list.map((a) => ({ ...a, _key: `db-${a.id}` })))
      ensureEmbed(first.id).catch(() => {})
    }
  }

  const saveAnns = useCallback(async (imageId, list) => {
    await api.saveAnnotations(imageId, list.map(({ _key, id, ...a }) => a))
    dirty.current = false
    setMsg('저장됨')
  }, [setMsg])

  const openImage = useCallback(async (im) => {
    if (dirty.current && current) await saveAnns(current.id, anns)
    resetEmbed()
    setCurrent(im)
    const list = await api.getAnnotations(im.id)
    setAnns(list.map((a) => ({ ...a, _key: `db-${a.id}` })))
    setSelectedId(null)
    setSuggest([])
    dirty.current = false
    undoStack.current = []
    ensureEmbed(im.id).catch(() => {})
  }, [current, anns, saveAnns])

  // 모든 어노테이션 변경은 이 함수로 — 언두 스택 자동 축적
  const setAnnsDirty = useCallback((next) => {
    undoStack.current.push(anns)
    if (undoStack.current.length > 50) undoStack.current.shift()
    dirty.current = true
    setAnns(next)
  }, [anns])

  const undo = useCallback(() => {
    const prev = undoStack.current.pop()
    if (prev) { dirty.current = true; setAnns(prev); setSelectedId(null); setMsg('실행 취소') }
  }, [setMsg])

  const moveImage = useCallback((delta) => {
    if (!current || !visible.length) return
    const i = visible.findIndex((im) => im.id === current.id)
    const next = visible[i + delta] || (i === -1 ? visible[0] : null)
    if (next) openImage(next)
  }, [current, visible, openImage])

  // 자동 저장 — 수정 후 2초 조용하면 저장 (S 강제도 여전히 가능)
  useEffect(() => {
    if (!dirty.current || !current) return
    const t = setTimeout(() => {
      if (dirty.current) saveAnns(current.id, anns)
    }, 2000)
    return () => clearTimeout(t)
  }, [anns, current, saveAnns])

  const setStatus = useCallback(async (status) => {
    if (!current) return
    await saveAnns(current.id, anns)
    await api.setImageStatus(current.id, status)
    setImages((imgs) => imgs.map((im) => (im.id === current.id ? { ...im, status } : im)))
    setMsg(status === 'approved' ? '승인 → 다음 이미지' : '거부 → 다음 이미지')
    moveImage(1)
  }, [current, anns, saveAnns, moveImage, setMsg])

  // 전역 핫키
  useEffect(() => {
    const h = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      if ((e.metaKey || e.ctrlKey) && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); undo(); return }
      if (e.key === 'Escape' && showHelp) { setShowHelp(false); return }
      if (e.key === 'ArrowRight') moveImage(1)
      else if (e.key === 'ArrowLeft') moveImage(-1)
      else if (e.key === 'a' || e.key === 'A') setStatus('approved')
      else if (e.key === 'x' || e.key === 'X') setStatus('rejected')
      else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedId) { setAnnsDirty(anns.filter((a) => a._key !== selectedId)); setSelectedId(null) }
      } else if ((e.key === 's' || e.key === 'S') && current) saveAnns(current.id, anns)
      else if (e.key === 'b' || e.key === 'B') setTool('box')
      else if (e.key === 'm' || e.key === 'M') setTool('sam')
      else if (e.key === 'e' || e.key === 'E') setTool('ex')
      else if (e.key === '?') setShowHelp((v) => !v)
      else if (/^[1-9]$/.test(e.key)) {
        const cls = project?.ontology[+e.key - 1]?.name
        if (!cls) return
        setActiveClass(cls)
        if (selectedId) {
          setAnnsDirty(anns.map((a) =>
            a._key === selectedId ? { ...a, class_name: cls, source: 'human' } : a))
        }
      }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  })

  // 백그라운드 심판 잡 폴링
  useEffect(() => {
    if (qaJob?.status !== 'running' || !project) return
    const t = setInterval(async () => {
      const s = await api.qaStatus(project.id)
      setQaJob(s)
      if (s.status !== 'running') {
        clearInterval(t)
        setImages(await api.listImages(project.id))
        if (s.result) { setLastQa(s.result); setMsg(qaSummary(s.result)) }
        else setMsg(`심판 실패: ${s.error}`)
      }
    }, 2000)
    return () => clearInterval(t)
  }, [qaJob?.status, project?.id]) // eslint-disable-line

  // 드래그앤드롭 업로드 — 창 어디에 놓아도 됨
  useEffect(() => {
    if (!project) return
    const over = (e) => e.preventDefault()
    const drop = async (e) => {
      e.preventDefault()
      const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith('image/'))
      if (!files.length) return
      setMsg(`${files.length}장 업로드 중…`)
      await api.uploadImages(project.id, files)
      setImages(await api.listImages(project.id))
      setMsg(`${files.length}장 업로드 완료`)
    }
    window.addEventListener('dragover', over)
    window.addEventListener('drop', drop)
    return () => { window.removeEventListener('dragover', over); window.removeEventListener('drop', drop) }
  }, [project?.id, setMsg]) // eslint-disable-line

  // 배치 잡 폴링
  useEffect(() => {
    if (job.status !== 'running' || !project) return
    const t = setInterval(async () => {
      const s = await api.autolabelStatus(project.id)
      setJob(s)
      if (s.status !== 'running') {
        clearInterval(t)
        setImages(await api.listImages(project.id))
        if (current) {
          const list = await api.getAnnotations(current.id)
          setAnns(list.map((a) => ({ ...a, _key: `db-${a.id}` })))
          dirty.current = false
        }
        setMsg(s.status === 'failed' ? `배치 실패: ${s.error}` : `배치 오토라벨 완료: ${s.done}/${s.total}장`)
      }
    }, 1500)
    return () => clearInterval(t)
  }, [job.status, project]) // eslint-disable-line

  if (!project) {
    return (
      <div className="page">
        <h2>오토라벨</h2>
        <p className="hint">프로젝트를 선택하거나 새로 만드세요. 흐름: 클래스 정의 → 이미지 업로드 → 오토라벨 → 리뷰 → (자동) 전용 모델 학습</p>
        <ProjectPicker projects={projects} onOpen={openProject} onCreated={refreshProjects} />
      </div>
    )
  }

  const approved = images.filter((im) => im.status === 'approved').length

  return (
    <div className="app">
      <header className="header">
        <button onClick={() => setProject(null)}>← 프로젝트</button>
        <b>{project.name}</b>
        <div className="progress" title={`승인 ${approved} / 전체 ${images.length}`}>
          <div className="progress-fill" style={{ width: images.length ? `${(approved / images.length) * 100}%` : 0 }} />
          <span>{approved}/{images.length} 승인</span>
        </div>
        <span className="spacer" />
        {trainInfo.active_model && (
          <span className="chip" title="오토라벨이 이 전용 모델을 사용 중">
            전용 모델 mAP50 {trainInfo.active_model.map50?.toFixed(2)}
          </span>
        )}
        <button onClick={() => setShowHelp(true)}>단축키 (?)</button>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <NextStep
            project={project} images={images} trainInfo={trainInfo} job={job}
            onAutolabel={async () => {
              if (!project.ontology.length) return setMsg('클래스를 먼저 정의하세요')
              setJob(await api.autolabelBatch(project.id, { masks: false }))
            }}
          />
          <OntologyEditor project={project} setProject={setProject} />
          <UploadBox project={project} onUploaded={async () => setImages(await api.listImages(project.id))} />
          <LinkImport project={project} setProject={setProject} onMsg={setMsg}
            onDone={async () => setImages(await api.listImages(project.id))} />
          <div className="card">
            <div className="panel-title">일괄 작업</div>
            <div className="row">
              <button className="primary" disabled={job.status === 'running'}
                title="전 이미지에 모델이 라벨 초안을 생성합니다 (덮어쓰지 않고 모델 라벨만 갱신)"
                onClick={async () => {
                  if (!project.ontology.length) return setMsg('클래스를 먼저 정의하세요')
                  setJob(await api.autolabelBatch(project.id, { masks: false }))
                }}>
                {job.status === 'running' ? `오토라벨 ${job.done ?? 0}/${job.total}` : '▶ 전체 오토라벨'}
              </button>
              <button title="전용 모델과 저장된 라벨을 대조해 의심 라벨을 찾고, 클래스별 권장 임계값을 계산합니다 (전용 모델 필요). 이미지가 많으면 백그라운드로 실행됩니다."
                disabled={qaJob?.status === 'running'}
                onClick={async () => {
                  const big = images.length > 200
                  setMsg(big ? '라벨 심판 시작 (백그라운드)…' : 'QA 분석 중…')
                  const r = await api.runQa(project.id, big)
                  if (r.error) return setMsg(r.error)
                  if (big) return setQaJob(r)
                  setImages(await api.listImages(project.id))
                  setLastQa(r)
                  setMsg(qaSummary(r))
                }}>
                {qaJob?.status === 'running' ? `심판 중 ${qaJob.done}/${qaJob.total}` : 'QA 분석'}
              </button>
            </div>
            {lastQa?.recommended_thresholds && Object.values(lastQa.recommended_thresholds).some((v) => v.tau != null) && (
              <div className="row">
                <button className="ok" title="정밀도 95% 기준으로 계산된 클래스별 임계값을 온톨로지에 반영합니다"
                  onClick={async () => {
                    const next = project.ontology.map((c) => {
                      const t = lastQa.recommended_thresholds[c.name]
                      return t?.tau != null ? { ...c, threshold: t.tau } : c
                    })
                    await api.saveOntology(project.id, next)
                    setProject({ ...project, ontology: next })
                    setMsg('권장 임계값 적용됨 — 다음 오토라벨부터 반영')
                  }}>✓ 권장 임계값 적용</button>
              </div>
            )}
            <div className="row">
              <button className="ok" title="모든 박스가 임계값 이상인 리뷰 대기 이미지를 한 번에 승인합니다"
                onClick={async () => {
                  const dry = await api.autoApprove(project.id, { min_conf: 0.7, dry_run: true })
                  if (!dry.approved) return setMsg(`자동 승인 대상 없음 (대기 ${dry.pending}장)`)
                  if (!confirm(`리뷰 대기 ${dry.pending}장 중 ${dry.approved}장이 고신뢰(≥0.7)입니다. 승인할까요?\n(저신뢰 ${dry.skipped_low_confidence}장은 남겨둡니다)`)) return
                  const r = await api.autoApprove(project.id, { min_conf: 0.7 })
                  setImages(await api.listImages(project.id))
                  setMsg(`${r.approved}장 자동 승인 (커버리지 ${(r.coverage * 100).toFixed(0)}%) — 나머지만 리뷰하세요`)
                }}>⚡ 고신뢰 자동 승인</button>
              <button title="모델이 헷갈리는 이미지를 먼저 보여줍니다 (라벨 예산 최적화)"
                onClick={async () => {
                  const r = await api.nextToLabel(project.id, 1)
                  if (!r.recommended.length) return setMsg('추천할 이미지 없음')
                  const top = r.recommended[0]
                  const im = images.find((x) => x.id === top.image_id)
                  if (im) { openImage(im); setMsg(`추천: ${top.file_name} (라벨 가치 ${top.score})`) }
                }}>🎯 다음 라벨 추천</button>
            </div>
            <div className="hint">내보내기 (zip — 바로 학습 가능한 구조)</div>
            <div className="row">
              <a href={api.exportUrl(project.id, 'coco')}>
                <button title="images/ + annotations.json (마스크 포함)">COCO.zip</button></a>
              <a href={api.exportUrl(project.id, 'yolo')}>
                <button title="images/ + labels/*.txt + data.yaml — ultralytics로 바로 학습 가능">YOLO.zip</button></a>
              {trainInfo.active_model && (
                <a href={api.modelUrl(project.id)}>
                  <button title="현재 활성 전용 모델 가중치 (.pt) — 다른 곳에서 바로 추론 가능">모델 .pt</button></a>
              )}
            </div>
          </div>
          <TrainPanel trainInfo={trainInfo} approved={approved} pid={project.id} onMsg={setMsg}
            onTrigger={async () => {
            setTrainInfo({ ...trainInfo, job: await api.triggerTrain(project.id) })
          }} />
          <ImageList
            visible={visible} current={current} onOpen={openImage}
            filter={filter} setFilter={setFilter}
            sortMode={sortMode} setSortMode={setSortMode}
            onDelete={async (im) => {
              if (!confirm(`${im.file_name} 삭제? 라벨도 함께 지워집니다.`)) return
              await api.deleteImage(im.id)
              const imgs = await api.listImages(project.id)
              setImages(imgs)
              if (current?.id === im.id) { setCurrent(null); setAnns([]) }
              setMsg('이미지 삭제됨')
            }}
          />
        </aside>

        <main className="main">
          <div className="toolbar row">
            {project.ontology.map((c, i) => (
              <button key={c.name}
                className={`cls ${activeClass === c.name ? 'active' : ''}`}
                style={{ '--c': classColor(project.ontology, c.name) }}
                onClick={() => {
                  setActiveClass(c.name)
                  if (selectedId) {
                    setAnnsDirty(anns.map((a) =>
                      a._key === selectedId ? { ...a, class_name: c.name, source: 'human' } : a))
                  }
                }}>
                <i />{i + 1}. {c.name}
              </button>
            ))}
            <span className="spacer" />
            <div className="seg">
              <button className={tool === 'box' ? 'active' : ''} onClick={() => setTool('box')} title="드래그로 박스 (B)">□ 박스</button>
              <button className={tool === 'sam' ? 'active' : ''} onClick={() => setTool('sam')} title="클릭으로 마스크 (M)">✦ SAM</button>
              <button className={tool === 'ex' ? 'active' : ''} onClick={() => setTool('ex')} title="예시 박스 → 유사 객체 전부 (E)">≡ 예시</button>
            </div>
            <button onClick={async () => {
              if (!current) return
              setMsg('오토라벨 중…')
              const r = await api.autolabelOne(current.id, project.ontology)
              setAnnsDirty([...anns, ...r.detections.map((d, i) => ({
                ...d, _key: `auto-${Date.now()}-${i}`, source: 'model' }))])
              setMsg(`오토라벨 ${r.detections.length}개 (${r.engine.split('(')[0]})`)
            }}>이 이미지 오토라벨</button>
            {trainInfo.active_model && (
              <button title="전용 모델이 찾았는데 라벨에 없는 박스를 점선으로 보여줍니다 (누락 라벨 찾기)"
                onClick={async () => {
                  if (!current) return
                  setMsg('모델 제안 확인 중…')
                  const r = await api.suggestions(current.id)
                  setSuggest(r.missing_labels)
                  setMsg(r.missing_labels.length
                    ? `누락 의심 ${r.missing_labels.length}개 — 점선 박스 클릭하면 라벨로 추가됩니다`
                    : '누락 의심 없음 — 라벨이 모델과 일치합니다')
                }}>🔍 누락 찾기</button>
            )}
            <button className="ok" onClick={() => setStatus('approved')} title="승인 후 다음 (A)">✓ 승인</button>
            <button className="bad" onClick={() => setStatus('rejected')} title="거부 후 다음 (X)">✗ 거부</button>
          </div>

          <div className="toolhint">
            {tool === 'box' && <>드래그로 박스를 그립니다 · 박스 클릭=선택 후 이동/크기조절 · <kbd>1~9</kbd>로 클래스 변경 · <kbd>Del</kbd> 삭제</>}
            {tool === 'sam' && <>객체를 클릭하면 마스크가 생깁니다 · 클릭=다음 객체(이전 확정) · <kbd>Shift</kbd>+클릭=현재 객체 넓히기 · 우클릭=빼기 · <kbd>Enter</kbd> 확정</>}
            {tool === 'ex' && <>찾고 싶은 객체 하나에 박스를 그리면, 같은 이미지에서 <b>닮은 것을 모두</b> 찾아줍니다 (텍스트로 부르기 어려운 객체용)</>}
          </div>
          {current ? (
            <div className="workspace">
              <Canvas
                imageUrl={api.imageUrl(current.id)}
                imageId={current.id} tool={tool} onMsg={setMsg}
                anns={anns} setAnns={setAnnsDirty}
                ontology={project.ontology}
                activeClass={activeClass}
                selectedId={selectedId} setSelectedId={setSelectedId}
                hoverId={hoverId}
                suggestions={suggest}
                onAcceptSuggestion={(s, i) => {
                  setAnnsDirty([...anns, {
                    ...s, _key: `sug-${Date.now()}-${i}`, source: 'model',
                    meta: { applied_from: 'suggestion' },
                  }])
                  setSuggest(suggest.filter((_, j) => j !== i))
                  setMsg('제안 수락 — 라벨로 추가됨')
                }}
                onExemplar={async (bbox) => {
                  setMsg('예시로 유사 객체 검색 중…')
                  const r = await api.exemplar(current.id, bbox, activeClass)
                  setAnnsDirty([...anns, ...r.detections.map((d, i) => ({
                    ...d, _key: `ex-${Date.now()}-${i}`, source: 'model', meta: { engine: 'exemplar' } }))])
                  setMsg(`예시 매칭 ${r.detections.length}개 — 틀린 것만 정리하세요`)
                }}
                size={{ w: window.innerWidth - 560, h: window.innerHeight - 120 }}
              />
              <AnnPanel
                anns={anns} ontology={project.ontology}
                selectedId={selectedId} setSelectedId={setSelectedId}
                hoverId={hoverId} setHoverId={setHoverId}
                onDelete={(key) => { setAnnsDirty(anns.filter((a) => a._key !== key)); if (selectedId === key) setSelectedId(null) }}
                onClass={(key, cls) => setAnnsDirty(anns.map((a) =>
                  a._key === key ? { ...a, class_name: cls, source: 'human' } : a))}
              />
            </div>
          ) : (
            <div className="empty">
              {images.length === 0
                ? <div>이미지가 없습니다.<br /><small>좌측에서 이미지를 업로드하세요 → 전체 오토라벨 → 리뷰(A/X)</small></div>
                : <div>좌측에서 이미지를 선택하세요.<br /><small>드래그=박스 · M=SAM 클릭 · E=예시 찾기 · A/X=승인/거부 · ?=단축키</small></div>}
            </div>
          )}
        </main>
      </div>

      {toast && <div className="toast">{toast}</div>}
      {showHelp && <HelpOverlay onClose={() => setShowHelp(false)} />}
    </div>
  )
}

const MIN_APPROVED = 8

// 심판 결과 한 줄 요약 — 라벨 오류율과 다음 행동
function qaSummary(r) {
  const rate = (r.estimated_label_error_rate * 100).toFixed(1)
  const taus = Object.entries(r.recommended_thresholds || {})
    .filter(([, v]) => v.tau != null).map(([c, v]) => `${c}≥${v.tau}`).join(' ')
  return `심판 완료 · 라벨 ${r.labels_checked}개 검사 · 추정 오류율 ${rate}% ` +
    `(불일치 ${r.breakdown.class_mismatch}, 누락의심 ${r.breakdown.possible_missing_label}) ` +
    `→ "의심" 정렬로 상위부터 수정하세요` + (taus ? ` · 권장 임계값 ${taus}` : '')
}

// 현재 상태에서 "지금 해야 할 일" 하나만 크게 안내 — 순서를 외우지 않게
function NextStep({ project, images, trainInfo, job, onAutolabel }) {
  const approved = images.filter((im) => im.status === 'approved').length
  const labeled = images.filter((im) => im.ann_count > 0).length
  const pending = images.filter((im) => im.status === 'prelabeled').length
  const model = trainInfo.active_model
  const training = trainInfo.job?.status === 'running'

  let step
  if (!project.ontology.length || project.ontology.some((c) => !c.name))
    step = { n: 1, title: '클래스를 정의하세요', desc: '찾을 객체 이름 + 검출 프롬프트(영문). 예: helmet / safety helmet' }
  else if (!images.length)
    step = { n: 2, title: '이미지를 업로드하세요', desc: '여러 장 한번에 선택 가능' }
  else if (job.status === 'running')
    step = { n: 3, title: `오토라벨 중… ${job.done ?? 0}/${job.total}`, desc: '완료되면 리뷰 대기로 넘어갑니다' }
  else if (!labeled)
    step = { n: 3, title: '전체 오토라벨을 실행하세요', desc: '모델이 전 이미지에 라벨 초안을 깝니다', action: { label: '▶ 전체 오토라벨 실행', fn: onAutolabel } }
  else if (pending)
    step = { n: 4, title: `리뷰 ${pending}장 남음`, desc: '맞으면 A(승인), 틀리면 고친 뒤 A. 필요 없으면 X(거부)' }
  else if (training)
    step = { n: 5, title: '전용 모델 학습 중…', desc: `${trainInfo.job.phase} — 끝나면 오토라벨이 더 정확해집니다` }
  else if (approved < MIN_APPROVED)
    step = { n: 5, title: `전용 모델까지 승인 ${approved}/${MIN_APPROVED}장`, desc: `${MIN_APPROVED - approved}장 더 승인하면 자동으로 학습이 시작됩니다` }
  else if (!model)
    step = { n: 5, title: '학습 준비 완료', desc: '"지금 학습"을 누르거나 승인을 더 쌓으세요' }
  else
    step = { n: 6, title: `전용 모델 가동 중 (mAP50 ${model.map50?.toFixed(2)})`, desc: '이미지를 더 넣고 오토라벨 → 리뷰를 반복하면 정확도가 계속 올라갑니다' }

  return (
    <div className="nextstep">
      <div className="ns-head">지금 할 일 · {step.n}단계</div>
      <div className="ns-title">{step.title}</div>
      <div className="ns-desc">{step.desc}</div>
      {step.action && (
        <button className="primary" style={{ marginTop: 8, width: '100%' }}
          onClick={step.action.fn}>{step.action.label}</button>
      )}
    </div>
  )
}

function AnnPanel({ anns, ontology, selectedId, setSelectedId, hoverId, setHoverId, onDelete, onClass }) {
  const listRef = useRef()
  // 캔버스에서 박스를 클릭하면 패널의 해당 행으로 스크롤
  useEffect(() => {
    if (!selectedId || !listRef.current) return
    listRef.current.querySelector(`[data-key="${selectedId}"]`)
      ?.scrollIntoView({ block: 'nearest' })
  }, [selectedId])

  return (
    <div className="annpanel" ref={listRef}>
      <div className="panel-title">어노테이션 ({anns.length})</div>
      <div className="hint">행에 마우스를 올리면 캔버스에서 해당 박스만 밝게 표시됩니다</div>
      {anns.length === 0 && <div className="hint">아직 없음 — 오토라벨 또는 드래그로 시작</div>}
      {anns.map((a, i) => (
        <div key={a._key} data-key={a._key}
          className={`annrow ${selectedId === a._key ? 'active' : ''} ${hoverId === a._key ? 'hovered' : ''}`}
          onClick={() => setSelectedId(a._key)}
          onMouseEnter={() => setHoverId(a._key)}
          onMouseLeave={() => setHoverId(null)}>
          <span className="annidx">{i + 1}</span>
          <i style={{ background: classColor(ontology, a.class_name) }} />
          <select value={a.class_name} onClick={(e) => e.stopPropagation()}
            onChange={(e) => onClass(a._key, e.target.value)}>
            {ontology.map((c) => <option key={c.name}>{c.name}</option>)}
          </select>
          <small>{a.confidence != null ? a.confidence.toFixed(2) : ''} {a.source === 'model' ? '🤖' : '✍️'}{a.segmentation ? ' ▦' : ''}</small>
          <button className="x" onClick={(e) => { e.stopPropagation(); onDelete(a._key) }}>×</button>
        </div>
      ))}
    </div>
  )
}

function HelpOverlay({ onClose }) {
  const keys = [
    ['A / X', '승인 / 거부 후 다음 이미지'], ['← →', '이미지 이동'],
    ['B / M / E', '박스 / SAM 클릭 / 예시 찾기'], ['1~9', '클래스 선택 (박스 선택 중이면 재할당)'],
    ['Del', '선택 박스 삭제'], ['Cmd+Z', '실행 취소'], ['S', '저장'],
    ['SAM: 클릭', '새 객체 (이전 자동 확정)'], ['SAM: Shift+클릭', '현재 객체 정제(포함)'],
    ['SAM: 우클릭', '제외 포인트'], ['SAM: Enter / Esc', '확정 / 취소'],
    ['휠 / Alt+드래그', '줌 / 팬'],
  ]
  return (
    <div className="overlay" onClick={onClose}>
      <div className="help" onClick={(e) => e.stopPropagation()}>
        <div className="panel-title">단축키</div>
        <table><tbody>
          {keys.map(([k, v]) => <tr key={k}><td><kbd>{k}</kbd></td><td>{v}</td></tr>)}
        </tbody></table>
        <button onClick={onClose}>닫기 (Esc)</button>
      </div>
    </div>
  )
}

function ProjectPicker({ projects, onOpen, onCreated }) {
  const [name, setName] = useState('')
  return (
    <div>
      <div className="row">
        <input placeholder="새 프로젝트 이름" value={name} onChange={(e) => setName(e.target.value)}
          onKeyDown={async (e) => {
            if (e.key === 'Enter' && name.trim()) { await api.createProject(name.trim(), []); setName(''); onCreated() }
          }} />
        <button className="primary" onClick={async () => {
          if (!name.trim()) return
          await api.createProject(name.trim(), [])
          setName(''); onCreated()
        }}>생성</button>
      </div>
      <ul className="plist">
        {projects.map((p) => (
          <li key={p.id} onClick={() => onOpen(p)}>
            <b>{p.name}</b> <small>{p.image_count ?? 0}장</small>
            <button className="x rowdel" title="프로젝트 삭제"
              onClick={async (e) => {
                e.stopPropagation()
                if (!confirm(`프로젝트 "${p.name}" 삭제? 이미지·라벨·모델 기록이 모두 지워집니다.`)) return
                await api.deleteProject(p.id)
                onCreated()
              }}>×</button>
          </li>
        ))}
      </ul>
    </div>
  )
}

function OntologyEditor({ project, setProject }) {
  const [rows, setRows] = useState(project.ontology)
  const save = async (next) => {
    setRows(next)
    await api.saveOntology(project.id, next)
    setProject({ ...project, ontology: next })
  }
  return (
    <details open>
      <summary>클래스 ({rows.length})</summary>
      <div className="hint">클래스명 · 검출 프롬프트(영문 권장) · 임계값</div>
      {rows.map((c, i) => (
        <div className="row" key={i}>
          <i className="dot" style={{ background: classColor(rows, c.name) }} />
          <input style={{ width: 66 }} value={c.name} placeholder="클래스"
            onChange={(e) => save(rows.map((r, j) => (j === i ? { ...r, name: e.target.value } : r)))} />
          <input style={{ flex: 1, minWidth: 60 }} value={c.prompt} placeholder="프롬프트"
            onChange={(e) => save(rows.map((r, j) => (j === i ? { ...r, prompt: e.target.value } : r)))} />
          <input style={{ width: 44 }} type="number" step="0.05" min="0" max="1" value={c.threshold}
            onChange={(e) => save(rows.map((r, j) => (j === i ? { ...r, threshold: +e.target.value } : r)))} />
          <button className="x" onClick={() => save(rows.filter((_, j) => j !== i))}>×</button>
        </div>
      ))}
      <button onClick={() => save([...rows, { name: '', prompt: '', threshold: 0.35 }])}>+ 클래스</button>
    </details>
  )
}

function UploadBox({ project, onUploaded }) {
  const [busy, setBusy] = useState(false)
  return (
    <label className="upload">
      {busy ? '업로드 중…' : '⬆ 이미지 업로드 (클릭 또는 다중 선택)'}
      <input type="file" multiple accept="image/*" hidden
        onChange={async (e) => {
          if (!e.target.files.length) return
          setBusy(true)
          await api.uploadImages(project.id, [...e.target.files])
          e.target.value = ''
          setBusy(false)
          onUploaded()
        }} />
    </label>
  )
}

// 대용량 데이터셋 연결 — 복사 없이 폴더 참조 + 기존 라벨 임포트
function LinkImport({ project, setProject, onMsg, onDone }) {
  const [open, setOpen] = useState(false)
  const [imagesDir, setImagesDir] = useState('')
  const [labelsDir, setLabelsDir] = useState('')
  const [cocoJson, setCocoJson] = useState('')
  const [limit, setLimit] = useState('')
  const [requireClass, setRequireClass] = useState('')
  const [info, setInfo] = useState(null)
  const [job, setJob] = useState(null)

  useEffect(() => {
    if (job?.status !== 'running') return
    const t = setInterval(async () => {
      const s = await api.importStatus(project.id)
      setJob(s)
      if (s.status !== 'running') {
        clearInterval(t)
        onDone()
        onMsg(s.status === 'failed' ? `임포트 실패: ${s.error}` : `임포트 완료: ${s.done}장 연결됨`)
      }
    }, 1000)
    return () => clearInterval(t)
  }, [job?.status]) // eslint-disable-line

  return (
    <details className="card" open={open} onToggle={(e) => setOpen(e.target.open)}>
      <summary>기존 데이터셋 연결 (복사 없음)</summary>
      <div className="hint">이미 라벨이 있는 폴더를 그대로 연결합니다. 이미지는 복사되지 않아 디스크를 쓰지 않습니다.</div>
      <input placeholder="이미지 폴더 경로 (필수)" value={imagesDir}
        onChange={(e) => setImagesDir(e.target.value)} style={{ width: '100%' }} />
      <input placeholder="YOLO 라벨 폴더 (선택)" value={labelsDir}
        onChange={(e) => setLabelsDir(e.target.value)} style={{ width: '100%', marginTop: 4 }} />
      <input placeholder="또는 COCO json 경로 (선택)" value={cocoJson}
        onChange={(e) => setCocoJson(e.target.value)} style={{ width: '100%', marginTop: 4 }} />
      <div className="row">
        <input placeholder="최대 장수" value={limit} onChange={(e) => setLimit(e.target.value)} style={{ width: 80 }} />
        <input placeholder="이 클래스 포함만" value={requireClass}
          onChange={(e) => setRequireClass(e.target.value)} style={{ flex: 1, minWidth: 80 }} />
      </div>
      <div className="row">
        <button onClick={async () => {
          const r = await api.importPreview({
            images_dir: imagesDir, labels_dir: labelsDir || undefined,
            coco_json: cocoJson || undefined,
          })
          setInfo(r)
          if (r.error) onMsg(r.error)
        }}>미리보기</button>
        <button className="primary" disabled={!imagesDir || job?.status === 'running'}
          onClick={async () => {
            const body = {
              images_dir: imagesDir, labels_dir: labelsDir || undefined,
              coco_json: cocoJson || undefined,
              class_names: project.ontology.map((c) => c.name),
              limit: limit ? +limit : undefined,
              require_class: requireClass || undefined,
            }
            setJob(await api.importDataset(project.id, body))
            onMsg('임포트 시작 — 진행률은 아래에 표시됩니다')
          }}>
          {job?.status === 'running' ? `연결 중 ${job.done}/${job.total}` : '연결 임포트'}
        </button>
      </div>
      {info && !info.error && (
        <div className="hint">
          이미지 <b>{info.images}</b>장 · 형식 <b>{info.format}</b>
          {info.classes && <> · 클래스 {info.classes.join(', ')}</>}
          {info.class_ids && <> · 클래스 id {info.class_ids.slice(0, 12).join(',')}{info.class_ids.length > 12 ? '…' : ''}</>}
          {info.format === 'yolo' && <><br />YOLO 라벨은 <b>왼쪽 클래스 목록 순서</b>대로 id가 매핑됩니다 (0번=첫 클래스)</>}
        </div>
      )}
      <ModelImport project={project} onMsg={onMsg} />
    </details>
  )
}

// 외부에서 학습한 .pt를 전용 모델로 등록
function ModelImport({ project, onMsg }) {
  const [path, setPath] = useState('')
  return (
    <div style={{ marginTop: 10, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
      <div className="hint">외부 학습 모델 등록 (.pt) — 클래스명은 자동 추출</div>
      <div className="row">
        <input placeholder="/path/to/best.pt" value={path}
          onChange={(e) => setPath(e.target.value)} style={{ flex: 1, minWidth: 100 }} />
        <button disabled={!path} onClick={async () => {
          const r = await api.importModel(project.id, { path })
          onMsg(r.names ? `모델 등록됨 · 클래스: ${r.names.join(', ')}` : '등록 실패')
        }}>등록</button>
      </div>
    </div>
  )
}

function TrainPanel({ trainInfo, onTrigger, approved = 0, pid, onMsg = () => {} }) {
  const { job, active_model } = trainInfo
  const running = job.status === 'running'
  const pct = Math.min(100, (approved / MIN_APPROVED) * 100)
  return (
    <div className="card">
      <div className="panel-title">전용 모델 (내 데이터로 학습)</div>
      <div className="hint">
        {active_model
          ? <>지금 이 모델이 오토라벨 담당 · mAP50 <b>{active_model.map50?.toFixed(3)}</b> · 승인 {active_model.train_images}장으로 학습</>
          : <>승인 라벨이 <b>{MIN_APPROVED}장</b> 모이면 자동으로 학습이 시작됩니다 (현재 {approved}장)</>}
      </div>
      {!active_model && (
        <div className="progress mini"><div className="progress-fill" style={{ width: `${pct}%` }} />
          <span>{approved}/{MIN_APPROVED}</span></div>
      )}
      <div className="row">
        <button disabled={running} onClick={onTrigger}
          title="승인 라벨로 지금 즉시 학습합니다 (조건 미달이어도 강제 실행)">
          {running ? `학습 중 (${job.phase || '…'})` : '지금 학습'}
        </button>
        {job.status === 'failed' && <small className="bad-text">실패: {job.error}</small>}
        {job.status === 'completed' && (
          <small className={job.promoted ? 'ok-text' : 'warn-text'}>
            {job.promoted ? `승격 (mAP50 ${job.map50})` : `게이트 탈락 (${job.map50})`}
          </small>
        )}
      </div>
      <ModelHistory pid={pid} refreshKey={job.status} onMsg={onMsg} />
    </div>
  )
}

// 학습 라운드별 성능 추이 + 롤백
function ModelHistory({ pid, refreshKey, onMsg }) {
  const [models, setModels] = useState([])
  const [open, setOpen] = useState(false)
  useEffect(() => {
    if (open) api.listModels(pid).then(setModels).catch(() => {})
  }, [open, pid, refreshKey])
  if (!open) {
    return <button style={{ marginTop: 6, fontSize: 12, padding: '2px 8px' }}
      onClick={() => setOpen(true)}>학습 이력 보기</button>
  }
  const best = Math.max(...models.map((m) => m.map50 || 0), 0.001)
  return (
    <div style={{ marginTop: 8 }}>
      <div className="panel-title">학습 이력 ({models.length})</div>
      {models.map((m) => (
        <div key={m.id} className="row" style={{ fontSize: 12, gap: 8 }}>
          <div style={{ width: 60, height: 8, background: 'var(--bg3)', borderRadius: 4 }}>
            <div style={{ width: `${((m.map50 || 0) / best) * 100}%`, height: '100%',
              background: m.active ? 'var(--ok)' : '#4a5560', borderRadius: 4 }} />
          </div>
          <span style={{ minWidth: 44 }}>{m.map50?.toFixed(3) ?? '—'}</span>
          <span className="hint" style={{ margin: 0 }}>{m.train_images}장</span>
          {m.meta?.imported && <span className="hint" style={{ margin: 0 }}>임포트</span>}
          {m.active
            ? <span className="ok-text">● 사용 중</span>
            : <button style={{ padding: '1px 6px', fontSize: 11 }}
                onClick={async () => {
                  await api.activateModel(pid, m.id)
                  setModels(await api.listModels(pid))
                  onMsg(`모델 #${m.id}로 전환 (mAP50 ${m.map50?.toFixed(3)})`)
                }}>사용</button>}
        </div>
      ))}
      <button style={{ marginTop: 4, fontSize: 12, padding: '2px 8px' }}
        onClick={() => setOpen(false)}>닫기</button>
    </div>
  )
}

function ImageList({ visible, current, onOpen, filter, setFilter, sortMode, setSortMode, onDelete }) {
  const badge = { unlabeled: '·', prelabeled: '◐', approved: '✓', rejected: '✗' }
  return (
    <div className="imagelist">
      <div className="row filters">
        {['all', 'prelabeled', 'approved', 'unlabeled'].map((f) => (
          <button key={f} className={filter === f ? 'active' : ''} onClick={() => setFilter(f)}>
            {{ all: '전체', prelabeled: '리뷰 대기', approved: '승인', unlabeled: '미라벨' }[f]}
          </button>
        ))}
        <button className={sortMode === 'conf' ? 'active' : ''}
          onClick={() => setSortMode(sortMode === 'conf' ? 'none' : 'conf')} title="신뢰도 낮은 이미지 먼저">불확실</button>
        <button className={sortMode === 'qa' ? 'active' : ''}
          onClick={() => setSortMode(sortMode === 'qa' ? 'none' : 'qa')} title="모델과 라벨이 싸우는 이미지 먼저 (QA 분석 후)">의심</button>
      </div>
      <div className="hint">←→ 이동도 이 목록 순서를 따릅니다</div>
      <ul className="ilist">
        {visible.map((im) => (
          <li key={im.id} className={current?.id === im.id ? 'active' : ''} onClick={() => onOpen(im)}>
            <img src={api.imageUrl(im.id)} loading="lazy" alt="" />
            <div className="meta">
              <div className="name">{im.file_name}</div>
              <small>
                <span className={`badge ${im.status}`}>{badge[im.status] || '·'}</span>
                {im.ann_count > 0 ? ` ${im.ann_count}개` : ' 라벨 없음'}
                {im.min_conf != null ? ` · conf ${im.min_conf.toFixed(2)}` : ''}
                {im.qa_score != null ? ` · QA ${im.qa_score}` : ''}
              </small>
            </div>
            <button className="x rowdel" title="이미지 삭제"
              onClick={(e) => { e.stopPropagation(); onDelete(im) }}>×</button>
          </li>
        ))}
      </ul>
    </div>
  )
}

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
  const [activeClass, setActiveClass] = useState('')
  const [job, setJob] = useState({ status: 'idle' })
  const [toast, setToast] = useState(null)
  const [tool, setTool] = useState('box')
  const [trainInfo, setTrainInfo] = useState({ job: { status: 'idle' }, active_model: null })
  const [showHelp, setShowHelp] = useState(false)
  const dirty = useRef(false)
  const undoStack = useRef([])

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
    if (!current || !images.length) return
    const i = images.findIndex((im) => im.id === current.id)
    const next = images[i + delta]
    if (next) openImage(next)
  }, [current, images, openImage])

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
              <button title="전용 모델과 저장된 라벨을 대조해 의심 라벨을 찾고, 클래스별 권장 임계값을 계산합니다 (전용 모델 필요)"
                onClick={async () => {
                  setMsg('QA 분석 중…')
                  const r = await api.runQa(project.id)
                  if (r.error) return setMsg(r.error)
                  setImages(await api.listImages(project.id))
                  const taus = Object.entries(r.recommended_thresholds || {})
                    .filter(([, v]) => v.tau != null).map(([c, v]) => `${c}≥${v.tau}`).join(' ')
                  setMsg(`QA 완료 — 이미지 목록에서 "의심" 정렬 사용 가능. 권장 임계값: ${taus || '표본 부족'}`)
                }}>QA 분석</button>
            </div>
            <div className="hint">라벨 내보내기</div>
            <div className="row">
              <a href={api.exportUrl(project.id, 'coco')} download={`${project.name}_coco.json`}>
                <button title="COCO 형식 JSON (마스크 포함)">COCO</button></a>
              <a href={api.exportUrl(project.id, 'yolo')} download={`${project.name}_yolo.json`}>
                <button title="YOLO 형식 (파일별 txt를 JSON으로 묶음)">YOLO</button></a>
            </div>
          </div>
          <TrainPanel trainInfo={trainInfo} approved={approved} onTrigger={async () => {
            setTrainInfo({ ...trainInfo, job: await api.triggerTrain(project.id) })
          }} />
          <ImageList images={images} current={current} onOpen={openImage} />
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

function AnnPanel({ anns, ontology, selectedId, setSelectedId, onDelete, onClass }) {
  return (
    <div className="annpanel">
      <div className="panel-title">어노테이션 ({anns.length})</div>
      {anns.length === 0 && <div className="hint">아직 없음 — 오토라벨 또는 드래그로 시작</div>}
      {anns.map((a) => (
        <div key={a._key}
          className={`annrow ${selectedId === a._key ? 'active' : ''}`}
          onClick={() => setSelectedId(a._key)}>
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

function TrainPanel({ trainInfo, onTrigger, approved = 0 }) {
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
    </div>
  )
}

function ImageList({ images, current, onOpen }) {
  const [filter, setFilter] = useState('all')
  const [sortMode, setSortMode] = useState('none')

  let list = filter === 'all' ? images : images.filter((im) => im.status === filter)
  if (sortMode === 'conf') list = [...list].sort((a, b) => (a.min_conf ?? 2) - (b.min_conf ?? 2))
  else if (sortMode === 'qa') list = [...list].sort((a, b) => (b.qa_score ?? -1) - (a.qa_score ?? -1))

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
      <ul className="ilist">
        {list.map((im) => (
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
          </li>
        ))}
      </ul>
    </div>
  )
}

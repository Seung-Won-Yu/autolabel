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
  const [sampling, setSampling] = useState(null) // 진행 중인 통계 검수 계획
  // 설정·도구는 라벨링을 시작하면 접는다 — 매번 쓰는 건 이미지 목록이다
  const [setupOpen, setSetupOpen] = useState(true)
  // 캔버스에서 숨긴 클래스 — 밀집 장면에서 한 종류만 보며 고칠 수 있게
  const [hidden, setHidden] = useState(new Set())
  // 저장 실패 상태 — 헤더에 계속 띄우고 스스로 재시도한다
  const [saveError, setSaveError] = useState(false)
  const dirty = useRef(false)
  const undoStack = useRef([])
  // 캔버스 크기는 컨테이너를 실측한다. 예전엔 window.innerWidth에서 상수를 뺀
  // 값이라 실제 여백과 어긋났고(가로는 남고 세로는 넘침), 창 크기를 바꿔도
  // 갱신되지 않았다.
  const canvasHost = useRef(null)
  const [canvasSize, setCanvasSize] = useState({ w: 900, h: 640 })
  // 승인/거부는 비동기라 연타하면 다음 호출이 낡은 클로저의 current를 읽어
  // 같은 이미지를 두 번 처리하고 나머지를 건너뛴다 (실측: a 5연타 → 2장만
  // 승인). 최신 값을 ref로 들고, 작업을 프라미스 체인으로 직렬화한다.
  const currentRef = useRef(null)
  const annsRef = useRef([])
  const openSeq = useRef(0) // 이미지 열기 순번 — 낡은 fetch가 새 화면을 덮지 않게
  const visibleRef = useRef([])
  const imagesRef = useRef([])
  const queue = useRef(Promise.resolve())

  // 화면에 보이는 목록(필터·정렬 적용) — 이동(←→)도 이 순서를 따른다
  let visible = filter === 'all' ? images
    : filter === 'flagged' ? images.filter((im) => (im.vlm_flags ?? 0) > 0)
    : images.filter((im) => im.status === filter)
  if (sortMode === 'conf') visible = [...visible].sort((a, b) => (a.min_conf ?? 2) - (b.min_conf ?? 2))
  else if (sortMode === 'qa') visible = [...visible].sort((a, b) => (b.qa_score ?? -1) - (a.qa_score ?? -1))
  visibleRef.current = visible
  imagesRef.current = images
  currentRef.current = current
  annsRef.current = anns

  // 읽고 판단해야 하는 안내(배치 진단·심판 결과)는 3초 만에 사라지면 안 된다.
  // sticky는 사용자가 닫을 때까지 남는다.
  const setMsg = useCallback((text, sticky = false) => {
    setToast(text ? { text, sticky } : null)
  }, [])

  useEffect(() => {
    if (!toast || toast.sticky) return
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
    // 빈 프로젝트는 설정부터 해야 하고, 이미 이미지가 있으면 바로 라벨링이다
    setSetupOpen(imgs.length === 0)
    // 새로고침·재진입 시 서버에서 아직 도는 배치를 복원한다 — 안 하면 진행
    // 표시가 사라지고, 재클릭은 409로 무음 실패한다. await 금지: 여기서 열기
    // 흐름을 늦추면 사용자가 이미 이동한 뒤 아래 setCurrent가 위치를 되감는다
    api.autolabelStatus(p.id)
      .then((s) => { if (s?.status === 'running') setJob(s) })
      .catch(() => {})
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
    // id는 유지해서 보낸다 — 서버가 백그라운드로 채운 meta.vlm(유료 판정)을
    // 이 사본에 없어도 id 기준으로 보존 병합할 수 있게
    await api.saveAnnotations(imageId, list.map(({ _key, ...a }) => a))
    // 저장 비행 중 새 편집이 있었으면(annsRef가 이미 다른 참조) dirty를 지우지
    // 않는다 — 지우면 그 편집의 자동저장 타이머와 이탈 경고가 전부 조용히
    // 건너뛰어, "저장됨" 토스트를 보고 닫은 탭에서 마지막 편집이 사라진다
    if (annsRef.current === list) dirty.current = false
    // 사이드바 메타(개수·최저 conf)도 즉시 맞춘다 — 박스를 지웠는데 옛 숫자가
    // 남아 있으면 검수 우선순위 판단이 틀어진다
    const confs = list.map((a) => a.confidence).filter((c) => c != null)
    setImages((imgs) => imgs.map((im) => (im.id === imageId
      ? { ...im, ann_count: list.length, min_conf: confs.length ? Math.min(...confs) : null }
      : im)))
    setMsg('저장됨')
  }, [setMsg])

  const openImage = useCallback(async (im) => {
    // 열기 순번 — 연속 이동(화살표 연타, 이동 직후 undo)에서 먼저 시작한
    // 열기의 fetch가 늦게 도착해 나중 이미지의 어노테이션을 덮는 레이스 차단
    const seq = ++openSeq.current
    const cur = currentRef.current
    if (dirty.current && cur) await saveAnns(cur.id, annsRef.current)
    // 저장을 기다리는 사이 더 최신 열기가 시작됐으면 여기서 멈춘다 — 계속
    // 진행하면 currentRef를 낡은 이미지로 되돌려 화면과 상태가 어긋난다
    if (seq !== openSeq.current) return false
    resetEmbed()
    // ref를 즉시 갱신한다 — 리렌더를 기다리면 연속 처리에서 다음 호출이
    // 아직 이전 이미지를 현재로 보고 같은 것을 또 처리한다
    currentRef.current = im
    setCurrent(im)
    const list = await api.getAnnotations(im.id)
    if (seq !== openSeq.current) return false // 더 최신 열기가 진행 중 — 낡은 응답 폐기
    const mapped = list.map((a) => ({ ...a, _key: `db-${a.id}` }))
    annsRef.current = mapped
    setAnns(mapped)
    setSelectedId(null)
    setSuggest([])
    dirty.current = false
    // 이력은 지우지 않는다 — 이미지를 넘긴 뒤에도 되돌릴 수 있어야 한다
    ensureEmbed(im.id).catch(() => {})
    return true // 이 열기가 최신으로 완료됨 (경합 시 false)
  }, [saveAnns])

  // 이미지가 처음 생기면 자동으로 연다. 프로젝트를 열 때만 열면, 임포트나
  // 업로드로 이미지를 넣은 직후 "좌측에서 이미지를 선택하세요"에 멈춰 있고
  // 썸네일 목록은 좌측 패널 아래라 화면 밖이다 — 리뷰할 게 눈앞에 없다.
  useEffect(() => {
    if (current || !images.length) return
    const first = images.find((im) => im.status === 'prelabeled')
      || images.find((im) => im.status === 'unlabeled') || images[0]
    openImage(first)
  }, [images, current, openImage])

  // 캔버스 컨테이너 크기 추적 — 창 조절·패널 접기에도 캔버스가 따라간다
  useEffect(() => {
    const el = canvasHost.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => {
      const { width, height } = e.contentRect
      if (width > 0 && height > 0) setCanvasSize({ w: Math.round(width), h: Math.round(height) })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [current?.id])

  // 되돌리기 이력은 이미지별이 아니라 프로젝트 단위 액션 로그다.
  // 예전엔 이미지를 넘기면 초기화돼서, A로 빠르게 리뷰하다 오승인하면
  // 복구할 방법이 아예 없었다 (되돌릴 수 없는 유일한 파괴적 동작).
  const pushHistory = useCallback((entry) => {
    undoStack.current.push(entry)
    if (undoStack.current.length > 100) undoStack.current.shift()
  }, [])

  const setAnnsDirty = useCallback((next) => {
    const cur = currentRef.current
    if (cur) pushHistory({ kind: 'anns', imageId: cur.id, before: annsRef.current })
    dirty.current = true
    annsRef.current = next
    setAnns(next)
  }, [pushHistory])

  const undoOnce = useCallback(async () => {
    const entry = undoStack.current.pop()
    if (!entry) return setMsg('되돌릴 작업이 없습니다')
    const target = imagesRef.current.find((im) => im.id === entry.imageId)
    const label = target?.file_name ? ` (${target.file_name})` : ''

    if (entry.kind === 'status') {
      await api.setImageStatus(entry.imageId, entry.before)
      setImages((imgs) => imgs.map((im) => (
        im.id === entry.imageId ? { ...im, status: entry.before } : im)))
      if (target && currentRef.current?.id !== entry.imageId) await openImage(target)
      return setMsg(`${entry.was === 'approved' ? '승인' : '거부'} 취소${label}`)
    }
    // 어노테이션 되돌리기 — 다른 이미지 것이면 그 이미지로 이동한 뒤 적용한다
    if (currentRef.current?.id !== entry.imageId) {
      if (!target) return setMsg('되돌릴 이미지를 찾을 수 없습니다')
      await openImage(target)
      // 이동(화살표)과 경합해 다른 이미지가 열렸으면 여기서 적용하면 안 된다 —
      // 엉뚱한 이미지에 이전 어노테이션이 저장된다. 되밀어 넣고 중단.
      if (currentRef.current?.id !== entry.imageId) {
        undoStack.current.push(entry)
        return setMsg('이미지 이동과 겹쳐 되돌리기를 중단했습니다 — 다시 Cmd+Z')
      }
    }
    dirty.current = true
    annsRef.current = entry.before
    setAnns(entry.before)
    setSelectedId(null)
    await saveAnns(entry.imageId, entry.before)
    setMsg(`실행 취소${label}`)
  }, [openImage, saveAnns, setMsg])

  // 되돌리기도 승인과 같은 큐에 태운다 — 처리 중에 끼어들면 이력과 실제
  // 상태가 어긋난다
  const undo = useCallback(() => {
    queue.current = queue.current.then(undoOnce).catch(
      (e) => setMsg(`되돌리기 실패: ${e.message}`))
    return queue.current
  }, [undoOnce, setMsg])

  // ref로 최신 목록·현재 이미지를 읽는다 — 연타 중 낡은 클로저를 보지 않게
  const moveImage = useCallback((delta) => {
    const list = visibleRef.current
    const cur = currentRef.current
    if (!cur || !list.length) return Promise.resolve()
    const i = list.findIndex((im) => im.id === cur.id)
    const next = list[i + delta] || (i === -1 ? list[0] : null)
    return next ? openImage(next) : Promise.resolve()
  }, [openImage])

  // 자동 저장 — 수정 후 2초 조용하면 저장 (S 강제도 여전히 가능).
  // 실패를 반드시 알린다. 예전엔 catch가 없어 서버가 죽으면 콘솔에
  // "Failed to fetch"만 남고 화면은 조용했다 — 계속 작업하다 전부 잃는다.
  useEffect(() => {
    if (!dirty.current || !current) return
    const t = setTimeout(async () => {
      if (!dirty.current) return
      try {
        await saveAnns(current.id, anns)
        setSaveError(false)
      } catch (e) {
        setSaveError(true)
        // dirty는 유지 — 다음 편집이나 재시도에서 다시 저장을 시도한다
        setMsg(`저장 실패: ${e.message} · 서버 연결을 확인하세요 (5초마다 재시도)`, true)
      }
    }, 2000)
    return () => clearTimeout(t)
  }, [anns, current, saveAnns, setMsg])

  // 저장이 실패한 동안은 스스로 재시도한다 — 사용자가 다시 편집하지 않아도
  // 서버가 돌아오면 저절로 복구되어야 한다
  useEffect(() => {
    if (!saveError) return
    const t = setInterval(async () => {
      const cur = currentRef.current
      if (!dirty.current || !cur) return setSaveError(false)
      try {
        await saveAnns(cur.id, annsRef.current)
        setSaveError(false)
        setMsg('저장 복구됨')
      } catch { /* 다음 주기에 다시 */ }
    }, 5000)
    return () => clearInterval(t)
  }, [saveError, saveAnns, setMsg])

  // 저장 안 된 변경이 있으면 이탈을 막는다
  useEffect(() => {
    const h = (e) => {
      if (!dirty.current) return
      e.preventDefault()
      e.returnValue = ''
    }
    window.addEventListener('beforeunload', h)
    return () => window.removeEventListener('beforeunload', h)
  }, [])

  // 연타해도 한 건도 잃지 않게 프라미스 체인으로 직렬화한다.
  // 예전엔 각 호출이 낡은 클로저의 current를 읽어 같은 이미지를 두 번
  // 처리하고 나머지를 건너뛰었다 (실측: a 5연타 → 2장만 승인).
  const setStatus = useCallback((status) => {
    queue.current = queue.current.then(async () => {
      const cur = currentRef.current
      if (!cur) return
      await saveAnns(cur.id, annsRef.current)
      // 되돌릴 수 있게 이전 상태를 남긴다 — 오승인 복구의 유일한 경로
      pushHistory({ kind: 'status', imageId: cur.id, before: cur.status, was: status })
      await api.setImageStatus(cur.id, status)
      setImages((imgs) => imgs.map((im) => (im.id === cur.id ? { ...im, status } : im)))
      setMsg(`${status === 'approved' ? '승인' : '거부'} → 다음 이미지 · Cmd+Z로 취소`)
      await moveImage(1)
      // 리스트 끝이면 앞쪽에 남은 리뷰 대기로 순환한다 — 여기서 멈추면 A 연타
      // 흐름이 죽는다 (실측: 15/15에서 승인 후 A 13연타 전부 무시)
      if (currentRef.current?.id === cur.id) {
        const pending = visibleRef.current.find(
          (im) => im.id !== cur.id && (im.status === 'prelabeled' || im.status === 'unlabeled'))
        if (pending) await openImage(pending)
      }
    }).catch((e) => setMsg(`상태 변경 실패: ${e.message}`))
    return queue.current
  }, [saveAnns, moveImage, openImage, setMsg, pushHistory])

  // 전역 핫키
  useEffect(() => {
    const h = (e) => {
      // SELECT도 제외한다 — 클래스 드롭다운을 열고 화살표로 고르는 중이면
      // 그 입력이지 박스 미세조정이 아니다
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return
      if ((e.metaKey || e.ctrlKey) && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); undo(); return }
      if (e.key === 'Escape') {
        if (showHelp) setShowHelp(false)
        else if (selectedId) setSelectedId(null)  // 선택 해제 = 화살표를 이미지 이동으로 되돌림
        return
      }
      // 박스를 선택한 상태의 화살표는 박스 미세조정이다 (Shift로 10px).
      // 마우스로는 1px을 맞출 수 없어 박스가 늘 어긋난다.
      const NUDGE = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }
      if (selectedId && NUDGE[e.key]) {
        e.preventDefault()
        const [dx, dy] = NUDGE[e.key]
        const step = e.shiftKey ? 10 : 1
        setAnnsDirty(anns.map((a) => (a._key === selectedId
          ? { ...a, bbox: [a.bbox[0] + dx * step, a.bbox[1] + dy * step, a.bbox[2], a.bbox[3]],
            source: 'human' }
          : a)))
        return
      }
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
        if (s.result) { setLastQa(s.result); setMsg(qaSummary(s.result), true) }
        else setMsg(`심판 실패: ${s.error || '진행 상황을 잃었습니다 (서버 재시작?) — 다시 실행하세요'}`, true)
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
      const r = await api.uploadImages(project.id, files)
      setImages(await api.listImages(project.id))
      // 요청 개수가 아니라 서버가 실제 저장한 개수로 말한다 — 손상 파일이
      // 섞이면 서버는 건너뛰고 200을 주므로 응답을 봐야 거짓 보고가 안 된다
      if (r.failed?.length) {
        setMsg(`${r.saved.length}/${files.length}장 업로드 — 실패: ${r.failed.join(', ')}`, true)
      } else setMsg(`${r.saved.length}장 업로드 완료`)
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
        // 배치가 도는 동안 사용자는 이미지를 옮겨 다닌다 — state의 current는
        // 폴링 시작 시점 클로저라 낡았다. 완료 시점의 현재 이미지(ref)를 쓰고,
        // 편집 중(dirty)이면 서버 갱신으로 미저장 편집을 덮지 않는다.
        const cur = currentRef.current
        if (cur && !dirty.current) {
          const list = await api.getAnnotations(cur.id)
          if (currentRef.current?.id === cur.id && !dirty.current) {
            const mapped = list.map((a) => ({ ...a, _key: `db-${a.id}` }))
            annsRef.current = mapped
            setAnns(mapped)
          }
        }
        // 완료가 아닌 종료를 완료로 말하지 않는다. 예전엔 서버 재시작으로
        // 잡 기록이 사라지면 "완료: undefined/undefined장"을 띄웠다 — 절반만
        // 라벨된 데이터를 두고 사용자는 끝난 줄 안다.
        const bad = s.status === 'failed' || s.status === 'interrupted' || s.status === 'idle'
        const text = s.status === 'failed' ? `배치 실패: ${s.error}`
          : s.status === 'interrupted' ? `배치가 중단됐습니다 (${s.done ?? 0}/${s.total ?? '?'}장 처리) — ${s.error || '다시 실행하세요'}`
            : s.status === 'idle' ? '배치 진행 상황을 잃었습니다 (서버 재시작?) — 다시 실행하세요'
              : s.advice || `배치 오토라벨 완료: ${s.done}/${s.total}장`
        // 읽고 판단해야 하는 안내는 사라지지 않게 둔다
        setMsg(text, bad || (s.verdict && s.verdict !== 'good'))
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
        {saveError && (
          <span className="chip danger" title="서버에 저장하지 못했습니다. 5초마다 재시도합니다 — 창을 닫으면 변경이 사라집니다">
            ⚠ 저장 안 됨 — 재시도 중
          </span>
        )}
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
              try {
                const j = await api.autolabelBatch(project.id, { masks: false })
                setJob(j)
                // 대상 0장이면 폴링이 돌지 않아 안내가 사라진다 — 즉시 알린다
                if (j.status !== 'running') setMsg(j.advice || '실행할 이미지가 없습니다')
              } catch (e) { setMsg(`배치 시작 실패: ${e.message}`) }
            }}
          />
          {sampling && (
            <SamplingPanel plan={sampling} onClose={() => setSampling(null)}
              onSubmit={async (defects) => {
                const r = await api.acceptanceResult(project.id, {
                  sample_size: sampling.sample_size, defects,
                  max_defects: sampling.max_defects,
                  target_error_rate: sampling.target_error_rate,
                  confidence: sampling.confidence,
                })
                setImages(await api.listImages(project.id))
                setMsg(r.message + (r.approved_images ? ` · ${r.approved_images}장 승인됨` : ''))
                if (r.accepted) setSampling(null)
              }} />
          )}

          {/* 설정·도구는 접힌다. 예전엔 이게 전부 펼쳐진 채 이미지 목록 위에
              쌓여 있어서 좌측 패널이 10화면 높이가 됐고, 정작 라벨링에 매번
              쓰는 이미지 목록은 화면 밖에서 시작했다. */}
          <button className="setup-toggle" onClick={() => setSetupOpen(!setupOpen)}>
            <span>{setupOpen ? '▾' : '▸'} 설정 · 도구</span>
            <small>클래스 · 업로드 · 오토라벨 · 익스포트 · 학습</small>
          </button>
          {setupOpen && (
            <div className="setup-body">
          <OntologyEditor project={project} setProject={setProject} />
          <UploadBox project={project} onMsg={setMsg}
            onUploaded={async () => setImages(await api.listImages(project.id))} />
          <LinkImport project={project} setProject={setProject} onMsg={setMsg}
            onDone={async () => setImages(await api.listImages(project.id))} />
          <PromptLab project={project} setProject={setProject} onMsg={setMsg}
            hasImages={images.length > 0} />
          <VlmJudge project={project} setProject={setProject} onMsg={setMsg}
            onDone={async () => {
              setImages(await api.listImages(project.id))
              // 판정 결과(meta.vlm)를 현재 이미지 패널에 반영 — 편집 중이면 건드리지 않는다
              const cur = currentRef.current
              if (cur && !dirty.current) {
                const list = await api.getAnnotations(cur.id)
                if (currentRef.current?.id === cur.id && !dirty.current) {
                  const mapped = list.map((a) => ({ ...a, _key: `db-${a.id}` }))
                  annsRef.current = mapped
                  setAnns(mapped)
                }
              }
            }} />
          <div className="card">
            <div className="panel-title">일괄 작업</div>
            <div className="row">
              <button className="primary" disabled={job.status === 'running'}
                title="전 이미지에 모델이 라벨 초안을 생성합니다 (덮어쓰지 않고 모델 라벨만 갱신)"
                onClick={async () => {
                  if (!project.ontology.length) return setMsg('클래스를 먼저 정의하세요')
                  try {
                    const j = await api.autolabelBatch(project.id, { masks: false })
                    setJob(j)
                    if (j.status !== 'running') setMsg(j.advice || '실행할 이미지가 없습니다')
                  } catch (e) { setMsg(`배치 시작 실패: ${e.message}`) }
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
                  setMsg(qaSummary(r), true)
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
              <button title="전수 검사 대신 통계적으로 필요한 만큼만 검사해 배치를 승인합니다"
                onClick={async () => {
                  const p = await api.acceptancePlan(project.id)
                  if (!p.sample_size) return setMsg('리뷰 대기 이미지 없음')
                  setSampling(p)
                  setFilter('prelabeled')
                  setMsg(`검수 계획: ${p.lot_size}장 중 ${p.sample_size}장만 검사 ` +
                    `(불량 ${p.max_defects}개까지 허용, 검수 ${(p.saving * 100).toFixed(0)}% 절감). ` +
                    `표본을 확인한 뒤 결과를 입력하세요`)
                }}>📊 배치 검수</button>
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
              <a href={`/api/projects/${project.id}/colab-notebook`}>
                <button title="로컬로 감당 안 되는 대규모 학습용 — Colab 노트북을 받아 GPU에서 학습 후 .pt를 다시 등록">
                  ☁ Colab 학습
                </button></a>
            </div>
          </div>
          <TrainPanel trainInfo={trainInfo} approved={approved} pid={project.id} onMsg={setMsg}
            onTrigger={async () => {
            setTrainInfo({ ...trainInfo, job: await api.triggerTrain(project.id) })
          }} />
            </div>
          )}
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
            onBulk={async (ids, status) => {
              if (!ids.length) return
              const r = await api.bulkStatus(ids, status)
              setImages(await api.listImages(project.id))
              // 백엔드는 started가 아니라 status를 준다 (scheduled=디바운스 예약)
              setMsg(`${r.count}장 ${status === 'approved' ? '승인' : '거부'}`
                + (['scheduled', 'running'].includes(r.train?.status) ? ' · 전용 모델 학습 예약' : ''))
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
              // 추론은 수 초 걸리고 그 사이 →로 이미지를 넘기는 게 리뷰의 기본
              // 동선이다. 응답이 늦게 오면 클릭 시점 이미지가 아닌 지금 이미지에
              // 검출이 합쳐져 자동저장이 엉뚱한 라벨을 서버에 덮어쓴다 — 요청
              // 이미지가 그대로일 때만 반영한다.
              const iid = current.id
              setMsg('오토라벨 중…')
              const r = await api.autolabelOne(iid, project.ontology)
              if (currentRef.current?.id !== iid) {
                return setMsg('이미지를 이동해 오토라벨 결과를 버렸습니다 — 그 이미지에서 다시 실행하세요', true)
              }
              setAnnsDirty([...annsRef.current, ...r.detections.map((d, i) => ({
                ...d, _key: `auto-${Date.now()}-${i}`, source: 'model' }))])
              setMsg(`오토라벨 ${r.detections.length}개 (${r.engine.split('(')[0]})`)
            }}>이 이미지 오토라벨</button>
            {trainInfo.active_model && (
              <button title="전용 모델이 찾았는데 라벨에 없는 박스를 점선으로 보여줍니다 (누락 라벨 찾기)"
                onClick={async () => {
                  if (!current) return
                  const iid = current.id
                  setMsg('모델 제안 확인 중…')
                  const r = await api.suggestions(iid)
                  // 응답 대기 중 이미지를 옮겼으면 버린다 — 다른 이미지 위에
                  // 엉뚱한 점선 제안이 뜬다
                  if (currentRef.current?.id !== iid) return
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
              <div className="canvas-host" ref={canvasHost}>
              <Canvas
                imageUrl={api.imageUrl(current.id)}
                imageId={current.id} tool={tool} onMsg={setMsg}
                anns={anns} setAnns={setAnnsDirty} hidden={hidden}
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
                  const iid = current.id
                  setMsg('예시로 유사 객체 검색 중…')
                  const r = await api.exemplar(iid, bbox, activeClass)
                  // 단건 오토라벨과 같은 레이스 — 요청 이미지가 그대로일 때만 반영
                  if (currentRef.current?.id !== iid) {
                    return setMsg('이미지를 이동해 예시 매칭 결과를 버렸습니다 — 그 이미지에서 다시 실행하세요', true)
                  }
                  setAnnsDirty([...annsRef.current, ...r.detections.map((d, i) => ({
                    ...d, _key: `ex-${Date.now()}-${i}`, source: 'model', meta: { engine: 'exemplar' } }))])
                  setMsg(`예시 매칭 ${r.detections.length}개 — 틀린 것만 정리하세요`)
                }}
                size={canvasSize}
              />
              </div>
              <AnnPanel
                anns={anns} ontology={project.ontology}
                selectedId={selectedId} setSelectedId={setSelectedId}
                hoverId={hoverId} setHoverId={setHoverId}
                hidden={hidden} setHidden={setHidden}
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

      {toast && (
        <div className={`toast${toast.sticky ? ' sticky' : ''}`}>
          <span>{toast.text}</span>
          {toast.sticky && <button className="x" title="닫기" onClick={() => setToast(null)}>×</button>}
        </div>
      )}
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
  // 박스 헐거움은 객체 자체는 라벨돼 있어 오류율에 안 들어간다 — 따로 보여준다
  const loose = r.breakdown.loose_box ? `, 박스어긋남 ${r.breakdown.loose_box}` : ''
  return `심판 완료 · 라벨 ${r.labels_checked}개 검사 · 추정 오류율 ${rate}% ` +
    `(불일치 ${r.breakdown.class_mismatch}, 누락의심 ${r.breakdown.possible_missing_label}${loose}) ` +
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

function AnnPanel({ anns, ontology, selectedId, setSelectedId, hoverId, setHoverId,
  onDelete, onClass, hidden, setHidden }) {
  const listRef = useRef()
  // 클래스별 개수 — 23개가 드롭다운 23줄로만 늘어서면 무엇이 몇 개인지 안 보인다
  const counts = anns.reduce((m, a) => ({ ...m, [a.class_name]: (m[a.class_name] || 0) + 1 }), {})
  const toggle = (cls) => {
    const next = new Set(hidden)
    if (next.has(cls)) next.delete(cls)
    else next.add(cls)
    setHidden(next)
  }
  const shown = anns.filter((a) => !hidden.has(a.class_name))
  // 캔버스에서 박스를 클릭하면 패널의 해당 행으로 스크롤
  useEffect(() => {
    if (!selectedId || !listRef.current) return
    listRef.current.querySelector(`[data-key="${selectedId}"]`)
      ?.scrollIntoView({ block: 'nearest' })
  }, [selectedId])

  return (
    <div className="annpanel" ref={listRef}>
      <div className="panel-title">어노테이션 ({anns.length})</div>
      {Object.keys(counts).length > 0 && (
        <>
          <div className="clschips">
            {Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([cls, n]) => (
              <button key={cls} className={`clschip${hidden.has(cls) ? ' off' : ''}`}
                title={hidden.has(cls) ? '캔버스에 다시 표시' : '캔버스에서 숨기기'}
                onClick={() => toggle(cls)}>
                <i style={{ background: classColor(ontology, cls) }} />
                {cls} <b>{n}</b>
              </button>
            ))}
          </div>
          <div className="hint">
            클래스를 눌러 캔버스에서 숨기거나 다시 표시합니다 · 행에 마우스를 올리면 해당 박스만 밝게
          </div>
        </>
      )}
      {anns.length === 0 && <div className="hint">아직 없음 — 오토라벨 또는 드래그로 시작</div>}
      {shown.map((a, i) => (
        <div key={a._key} data-key={a._key}
          className={`annrow ${selectedId === a._key ? 'active' : ''} ${hoverId === a._key ? 'hovered' : ''}`}
          // 행을 누르면 클래스 드롭다운에 포커스가 남아 화살표를 그쪽이 먹는다.
          // 박스를 고른 뒤 바로 화살표로 미세조정할 수 있어야 한다.
          onClick={(e) => {
            setSelectedId(a._key)
            if (e.target.tagName === 'SELECT') e.target.blur()
          }}
          onMouseEnter={() => setHoverId(a._key)}
          onMouseLeave={() => setHoverId(null)}>
          <span className="annidx">{i + 1}</span>
          <i style={{ background: classColor(ontology, a.class_name) }} />
          <select value={a.class_name} onClick={(e) => e.stopPropagation()}
            onChange={(e) => onClass(a._key, e.target.value)}>
            {ontology.map((c) => <option key={c.name}>{c.name}</option>)}
          </select>
          <small>{a.confidence != null ? a.confidence.toFixed(2) : ''} {a.source === 'model' ? '🤖' : '✍️'}{a.segmentation ? ' ▦' : ''}</small>
          {a.meta?.vlm && (
            <span className={`vchip ${a.meta.vlm.verdict}`}
              title={`문맥 심판: ${a.meta.vlm.reason}`}>
              {a.meta.vlm.verdict === 'pass' ? '✓' : a.meta.vlm.verdict === 'fail' ? '✗' : '?'}
            </span>
          )}
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
    ['박스 선택 중 ← →↑↓', '박스 1px 이동 (Shift로 10px)'], ['Esc', '선택 해제'],
    ['SAM: 클릭', '새 객체 (이전 자동 확정)'], ['SAM: Shift+클릭', '현재 객체 정제(포함)'],
    ['SAM: 우클릭', '제외 포인트'], ['SAM: Enter / Esc', '확정 / 취소'],
    ['휠 / Alt+드래그', '줌 / 팬'], ['+ / − / 0', '확대 / 축소 / 화면에 맞춤'],
    ['목록: Shift+클릭', '범위 선택'], ['목록: ⌘/Ctrl+클릭', '하나씩 추가·해제 → 일괄 승인'],
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
            // 만들었으면 바로 들어간다 — 목록에서 방금 만든 걸 또 찾아 누르게 하지 않기
            if (e.key === 'Enter' && name.trim()) { const p = await api.createProject(name.trim(), []); setName(''); onOpen(p) }
          }} />
        <button className="primary" onClick={async () => {
          if (!name.trim()) return
          const p = await api.createProject(name.trim(), [])
          setName(''); onOpen(p)
        }}>생성</button>
      </div>
      <ul className="plist">
        {projects.map((p) => (
          <li key={p.id} role="button" tabIndex={0} onClick={() => onOpen(p)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen(p) } }}>
            <b>{p.name}</b> <small>{p.image_count ?? 0}장
              {p.approved_count > 0 ? ` · ${p.approved_count}장 승인` : ''}</small>
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
  // 임계값 적용·프롬프트 실험 등 에디터 밖에서 온톨로지가 바뀌면 로컬
  // 스냅샷도 따라간다 — 안 하면 다음 키 입력의 save가 이전 스냅샷 전체를
  // PUT해 그 변경을 조용히 되돌린다
  useEffect(() => { setRows(project.ontology) }, [project.ontology])
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
          <input style={{ width: 58 }} type="number" step="0.05" min="0" max="1" value={c.threshold}
            onChange={(e) => save(rows.map((r, j) => (j === i ? { ...r, threshold: +e.target.value } : r)))} />
          <button className="x" onClick={() => save(rows.filter((_, j) => j !== i))}>×</button>
        </div>
      ))}
      <button onClick={() => save([...rows, { name: '', prompt: '', threshold: 0.35 }])}>+ 클래스</button>
    </details>
  )
}

function UploadBox({ project, onUploaded, onMsg }) {
  const [busy, setBusy] = useState(false)
  return (
    <label className="upload">
      {busy ? '업로드 중…' : '⬆ 이미지 업로드 (클릭 또는 다중 선택)'}
      <input type="file" multiple accept="image/*" hidden
        onChange={async (e) => {
          if (!e.target.files.length) return
          setBusy(true)
          const files = [...e.target.files]
          const r = await api.uploadImages(project.id, files)
          e.target.value = ''
          setBusy(false)
          // 부분 실패를 조용히 넘기지 않는다 (드래그앤드롭 경로와 같은 규칙)
          if (r.failed?.length) {
            onMsg?.(`${r.saved.length}/${files.length}장 업로드 — 실패: ${r.failed.join(', ')}`, true)
          }
          onUploaded()
        }} />
    </label>
  )
}

// 프롬프트 실험 — 후보를 표본 몇 장에 돌려 비교하고 이긴 것을 클래스에 적용.
// 제로샷 품질은 프롬프트가 좌우하는데, 예전엔 프롬프트를 바꿔 전체를 다시
// 돌려보는 것 말고 비교할 방법이 없었다.
function PromptLab({ project, setProject, onMsg, hasImages }) {
  const [open, setOpen] = useState(false)
  const [cls, setCls] = useState(project.ontology[0]?.name || '')
  const [text, setText] = useState('')
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)

  const target = project.ontology.find((c) => c.name === cls) || project.ontology[0]

  const run = async () => {
    const prompts = text.split('\n').map((s) => s.trim()).filter(Boolean)
    if (!prompts.length) return onMsg('후보 프롬프트를 한 줄에 하나씩 적어주세요')
    setBusy(true)
    try {
      const r = await api.promptLab(project.id, { class_name: cls, prompts, n_images: 5 })
      setRes(r)
      onMsg(`표본 ${r.sampled_images}장 비교 완료 — 1위 "${r.best}"`)
    } catch (e) { onMsg(`프롬프트 실험 실패: ${e.message}`) } finally { setBusy(false) }
  }

  const apply = async (prompt) => {
    const onto = project.ontology.map((c) => (c.name === cls ? { ...c, prompt } : c))
    await api.saveOntology(project.id, onto)
    setProject({ ...project, ontology: onto })
    onMsg(`"${prompt}" 적용됨 — 전체 오토라벨을 다시 실행하세요`)
  }

  if (!project.ontology.length) return null
  return (
    <div className="card">
      <div className="panel-title" style={{ cursor: 'pointer' }} onClick={() => {
        setOpen(!open)
        if (!open && !text) {
          // 현재 프롬프트를 첫 줄에 두어 "지금보다 나은가"를 바로 볼 수 있게
          const cur = target?.prompt || target?.name || ''
          setText([cur, target?.name, `a photo of ${target?.name}`]
            .filter((v, i, a) => v && a.indexOf(v) === i).join('\n'))
        }
      }}>
        {open ? '▾' : '▶'} 프롬프트 실험 (어떤 표현이 잘 잡히나)
      </div>
      {open && (
        <>
          <p className="hint">
            후보를 표본 5장에 돌려 비교합니다. 검출된 장수가 많고 확신도가 높은 쪽이 낫습니다.
            장당 박스가 지나치게 많으면 과검출이니 주의하세요.
          </p>
          {project.ontology.length > 1 && (
            <select value={cls} onChange={(e) => { setCls(e.target.value); setRes(null) }}>
              {project.ontology.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          )}
          <textarea rows={4} value={text} onChange={(e) => setText(e.target.value)}
            placeholder={'후보를 한 줄에 하나씩\nhandwritten signature\nautograph'}
            style={{ width: '100%', marginTop: 6 }} />
          <div className="row">
            <button className="primary" disabled={busy || !hasImages} onClick={run}>
              {busy ? '비교 중…' : '비교 실행'}
            </button>
            {!hasImages && <span className="hint">이미지를 먼저 넣어주세요</span>}
          </div>
          {res && (
            <table className="lab">
              <thead><tr><th>프롬프트</th><th>검출 장수</th><th>평균 conf</th><th>장당</th><th /></tr></thead>
              <tbody>
                {res.results.map((r) => (
                  <tr key={r.prompt} className={r.prompt === res.best ? 'win' : undefined}>
                    <td>{r.prompt}</td>
                    <td>{r.images_with_detection}/{res.sampled_images}</td>
                    <td>{r.avg_confidence}</td>
                    <td>{r.per_image}</td>
                    <td><button onClick={() => apply(r.prompt)}
                      disabled={r.prompt === target?.prompt}>적용</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  )
}

// VLM 문맥 심판 — 외형이 아니라 "기준"으로 판정해야 하는 라벨용.
// 검출 모델은 어디 있는지만 안다. "이 차가 사고 차량인가" 같은 문맥 판정은
// 기준 텍스트를 읽고 이미지를 보는 VLM이 예비 판정하고, 사람은 위반·불확실만
// 확인한다 — 리뷰가 전수 판독에서 예외 확인으로 바뀐다.
function VlmJudge({ project, setProject, onMsg, onDone }) {
  const [rubric, setRubric] = useState(project.rubric || '')
  const [saved, setSaved] = useState(true)
  const [job, setJob] = useState(null)

  useEffect(() => { setRubric(project.rubric || ''); setSaved(true) }, [project.id]) // eslint-disable-line

  // 이미 돌고 있는 심판을 복원한다 — 새로 연 화면(재접속·다른 탭)에서 버튼이
  // "실행" 대기로 보이면 진행 중인 판정을 이중 실행하거나 죽은 줄 안다.
  // 논블로킹 (.then) — 프로젝트 열기 흐름을 지연시키지 않는다 (배치 복원과 동일)
  useEffect(() => {
    api.vlmStatus(project.id).then((s) => {
      if (s.status === 'running') setJob(s)
    }).catch(() => {})
  }, [project.id])

  useEffect(() => {
    if (job?.status !== 'running') return
    const t = setInterval(async () => {
      const s = await api.vlmStatus(project.id)
      setJob(s)
      if (s.status !== 'running') {
        clearInterval(t)
        onDone()
        // 완료가 아닌 종료를 완료로 말하지 않는다 (배치·임포트와 같은 규칙)
        const text = s.status === 'failed' ? `문맥 심판 실패: ${s.error}`
          : s.status === 'interrupted' ? `문맥 심판이 중단됐습니다 (${s.done ?? 0}/${s.total ?? '?'}장) — 다시 실행하세요`
            : s.status === 'idle' ? '문맥 심판 진행 상황을 잃었습니다 (서버 재시작?) — 다시 실행하세요'
              : s.advice || '문맥 심판 완료'
        onMsg(text, true)
      }
    }, 1500)
    return () => clearInterval(t)
  }, [job?.status]) // eslint-disable-line

  return (
    <div className="card">
      <div className="panel-title">🧑‍⚖️ 문맥 심판 (VLM)</div>
      <div className="hint">
        외형만으로 판정할 수 없는 라벨용. 판정 기준을 글로 쓰면 VLM이 박스마다
        부합✓/위반✗/불확실? + 근거를 달아줍니다 — 위반·불확실만 확인하세요.
        같은 기준 재실행은 캐시를 써서 비용이 없습니다.
      </div>
      <textarea value={rubric} rows={4} style={{ width: '100%' }}
        placeholder={'판정 기준을 서술하세요. 예:\n- 파손·충돌 흔적이 있거나 사고 현장에 정차한 차량만 accident_vehicle\n- 단순 주행·주차 중인 차량은 제외'}
        onChange={(e) => { setRubric(e.target.value); setSaved(false) }} />
      <div className="row">
        <button disabled={saved} onClick={async () => {
          await api.saveRubric(project.id, rubric)
          setProject({ ...project, rubric })
          setSaved(true)
          onMsg('판정 기준 저장됨')
        }}>기준 저장</button>
        <button className="primary"
          disabled={job?.status === 'running' || !rubric.trim()}
          title="리뷰 대기(prelabeled) 이미지의 모든 박스를 기준으로 판정합니다"
          onClick={async () => {
            try {
              // 화면의 기준과 판정에 쓰이는 기준이 어긋나면 안 된다 — 미저장이면
              // 먼저 저장한다 (수정해놓고 옛 기준으로 판정되는 사고 방지)
              if (!saved) {
                await api.saveRubric(project.id, rubric)
                setProject({ ...project, rubric })
                setSaved(true)
              }
              const s = await api.vlmJudge(project.id)
              setJob(s)
              if (s.status !== 'running') onMsg(s.advice || '판정할 이미지가 없습니다')
            } catch (e) {
              onMsg(`문맥 심판 시작 실패: ${e.message}${e.message.includes('503')
                ? ' — VLM 제공자 없음 (Claude Code CLI 설치=구독으로 무료, 또는 ANTHROPIC_API_KEY, 또는 Ollama)' : ''}`, true)
            }
          }}>
          {job?.status === 'running'
            ? `심판 중 — 박스 ${job.done_boxes ?? 0}/${job.total_boxes ?? '?'} (이미지 ${job.done ?? 0}/${job.total})`
            : '▶ 문맥 심판 실행'}
        </button>
      </div>
    </div>
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
        // 완료가 아닌 종료를 완료로 말하지 않는다 (배치 폴링과 동일한 규칙).
        // 서버가 재시작하면 sweep이 interrupted로 남긴다 — 5만 장 중 400장만
        // 연결된 걸 "임포트 완료: 400장"으로 알리면 안 된다.
        const text = s.status === 'failed' ? `임포트 실패: ${s.error}`
          : s.status === 'interrupted' ? `임포트가 중단됐습니다 (${s.done ?? 0}장 연결) — ${s.error || '다시 실행하세요'}`
            : s.status === 'idle' ? '임포트 진행 상황을 잃었습니다 (서버 재시작?) — 다시 실행하세요'
              : `임포트 완료: ${s.done}장 연결됨`
        onMsg(text, s.status !== 'completed')
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

// 통계적 배치 검수 — 표본만 보고 배치 전체를 판정
function SamplingPanel({ plan, onSubmit, onClose }) {
  const [defects, setDefects] = useState(0)
  return (
    <div className="card" style={{ borderColor: 'var(--accent)' }}>
      <div className="panel-title">📊 배치 검수 진행 중</div>
      <div className="hint">
        대기 <b>{plan.lot_size}</b>장 중 <b>{plan.sample_size}</b>장만 검사하면 됩니다
        (검수 {(plan.saving * 100).toFixed(0)}% 절감).<br />
        표본을 보고 <b>라벨이 틀린 이미지 수</b>를 세어 입력하세요.
        불량 <b>{plan.max_defects}개 이하</b>면 배치 전체가 승인됩니다.
      </div>
      <div className="row">
        <input type="number" min="0" value={defects} style={{ width: 70 }}
          onChange={(e) => setDefects(+e.target.value)} />
        <button className="primary" onClick={() => onSubmit(defects)}>판정</button>
        <button onClick={onClose}>취소</button>
      </div>
      <div className="hint">
        표본 이미지 id: {plan.sample_image_ids?.slice(0, 12).join(', ')}
        {plan.sample_image_ids?.length > 12 ? ' …' : ''}
      </div>
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
  const best = Math.max(...models.map((m) => m.test_map50 || m.map50 || 0), 0.001)
  const curve = models.filter((m) => m.test_map50 != null)
    .sort((a, b) => a.train_images - b.train_images)
  return (
    <div style={{ marginTop: 8 }}>
      <div className="panel-title">학습 이력 ({models.length})</div>
      <div className="hint">막대 = 홀드아웃 성능(학습·게이트에 안 쓴 데이터 기준)</div>
      {models.map((m) => (
        <div key={m.id} className="row" style={{ fontSize: 12, gap: 8 }}>
          <div style={{ width: 56, height: 8, background: 'var(--bg3)', borderRadius: 4 }}>
            <div style={{ width: `${((m.test_map50 ?? m.map50 ?? 0) / best) * 100}%`,
              height: '100%', background: m.active ? 'var(--ok)' : '#4a5560',
              borderRadius: 4 }} />
          </div>
          <span style={{ minWidth: 40 }} title="홀드아웃 mAP50">
            {m.test_map50?.toFixed(3) ?? '—'}
          </span>
          <span className="hint" style={{ margin: 0 }} title="게이트용 val mAP50">
            (val {m.map50?.toFixed(2) ?? '—'})
          </span>
          <span className="hint" style={{ margin: 0 }}>{m.train_images}장</span>
          {m.meta?.imported && <span className="hint" style={{ margin: 0 }}>임포트</span>}
          {m.active
            ? <span className="ok-text">● 사용 중</span>
            : <button style={{ padding: '1px 6px', fontSize: 11 }}
                onClick={async () => {
                  await api.activateModel(pid, m.id)
                  setModels(await api.listModels(pid))
                  onMsg(`모델 #${m.id}로 전환 (홀드아웃 ${m.test_map50?.toFixed(3) ?? '—'})`)
                }}>사용</button>}
        </div>
      ))}
      {curve.length >= 2 && (
        <div className="hint" style={{ marginTop: 6 }}>
          데이터 효율: {curve.map((m) => `${m.train_images}장→${m.test_map50.toFixed(2)}`).join(' · ')}
          <br />
          {(() => {
            const a = curve[curve.length - 2], b = curve[curve.length - 1]
            const gain = (b.test_map50 - a.test_map50) / Math.max(b.train_images - a.train_images, 1) * 100
            return gain > 0.05
              ? `100장 추가당 약 +${(gain * 100).toFixed(1)}%p — 더 라벨할 가치가 있습니다`
              : '최근 구간에서 이득이 작습니다 — 라벨 양보다 품질·다양성을 보세요'
          })()}
        </div>
      )}
      <button style={{ marginTop: 4, fontSize: 12, padding: '2px 8px' }}
        onClick={() => setOpen(false)}>닫기</button>
    </div>
  )
}

// 목록 행 높이 — 창 렌더링이 스크롤 위치로 구간을 계산하므로 CSS의
// `.ilist li` 높이와 반드시 같아야 한다 (index.css 참조).
const ROW_H = 54

function ImageList({ visible, current, onOpen, filter, setFilter, sortMode, setSortMode,
  onDelete, onBulk }) {
  const badge = { unlabeled: '·', prelabeled: '◐', approved: '✓', rejected: '✗' }
  const pos = visible.findIndex((im) => im.id === current?.id)

  // 창 렌더링 — 보이는 구간만 DOM에 올린다. 예전엔 전부 올려서 310장이면
  // 노드 2364개(행당 7개)였고, 1만 장이면 7만 개가 되어 못 버틴다.
  // 행 높이가 CSS로 고정(ROW_H)이라 스크롤 위치만으로 구간을 계산할 수 있다.
  const listRef = useRef(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewH, setViewH] = useState(600)
  useEffect(() => {
    const el = listRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setViewH(e.contentRect.height))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const OVERSCAN = 6  // 위아래 여유 — 빠르게 스크롤할 때 빈칸이 보이지 않게
  const start = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN)
  const end = Math.min(visible.length, Math.ceil((scrollTop + viewH) / ROW_H) + OVERSCAN)
  const window_ = visible.slice(start, end)

  // 현재 이미지를 보이게 스크롤. 창 렌더링에서는 scrollIntoView를 쓸 수 없다 —
  // 화면 밖 행은 DOM에 없기 때문이다. 위치를 직접 계산한다.
  useEffect(() => {
    const el = listRef.current
    if (!el || pos < 0) return
    const top = pos * ROW_H
    if (top < el.scrollTop) el.scrollTop = top
    else if (top + ROW_H > el.scrollTop + el.clientHeight) {
      el.scrollTop = top + ROW_H - el.clientHeight
    }
    // 상태도 직접 맞춘다. scroll 이벤트만 믿으면 연속 이동에서 상태가 뒤처져
    // 창이 원래 자리에 머물고 현재 행이 DOM에 없는 일이 생긴다.
    setScrollTop(el.scrollTop)
  }, [pos])
  // 다중 선택 — Shift=범위, Cmd/Ctrl=토글 (파일 목록의 표준 관례)
  const [picked, setPicked] = useState(new Set())
  const anchor = useRef(null)

  const onRowClick = (im, i, e) => {
    if (e.shiftKey && anchor.current != null) {
      const [a, b] = [anchor.current, i].sort((x, y) => x - y)
      setPicked(new Set(visible.slice(a, b + 1).map((x) => x.id)))
      return
    }
    if (e.metaKey || e.ctrlKey) {
      const next = new Set(picked)
      if (next.has(im.id)) next.delete(im.id)
      else next.add(im.id)
      setPicked(next)
      anchor.current = i
      return
    }
    // 평소 클릭은 선택을 지우고 그 이미지를 연다 (기존 동작 유지)
    setPicked(new Set())
    anchor.current = i
    onOpen(im)
  }

  const applyBulk = async (status) => {
    const ids = [...picked]
    await onBulk(ids, status)
    setPicked(new Set())
  }


  return (
    <div className="imagelist">
      <div className="row filters">
        {/* 거부도 필터에 둔다 — 예전엔 '전체'에서만 보여 되살릴 방법이 없었다 */}
        {['all', 'prelabeled', 'approved', 'unlabeled', 'rejected', 'flagged'].map((f) => (
          <button key={f} className={filter === f ? 'active' : ''} onClick={() => setFilter(f)}
            title={f === 'rejected' ? '거부한 이미지 — 익스포트에서 제외됩니다'
              : f === 'flagged' ? '문맥 심판이 위반(✗)·불확실(?)로 판정한 박스가 있는 이미지' : undefined}>
            {{ all: '전체', prelabeled: '리뷰 대기', approved: '승인',
              unlabeled: '미라벨', rejected: '거부', flagged: '심판 ✗' }[f]}
          </button>
        ))}
        <button className={sortMode === 'conf' ? 'active' : ''}
          onClick={() => setSortMode(sortMode === 'conf' ? 'none' : 'conf')} title="신뢰도 낮은 이미지 먼저">불확실</button>
        <button className={sortMode === 'qa' ? 'active' : ''}
          onClick={() => setSortMode(sortMode === 'qa' ? 'none' : 'qa')} title="모델과 라벨이 싸우는 이미지 먼저 (QA 분석 후)">의심</button>
      </div>
      {picked.size > 0 ? (
        <div className="row bulkbar">
          <b>{picked.size}장 선택</b>
          <button className="ok" onClick={() => applyBulk('approved')}>✓ 일괄 승인</button>
          <button className="bad" onClick={() => applyBulk('rejected')}>✗ 일괄 거부</button>
          <button onClick={() => setPicked(new Set())}>해제</button>
        </div>
      ) : (
        <div className="hint">
          {pos >= 0 ? <b>{pos + 1} / {visible.length}</b> : `${visible.length}장`}
          {' · ←→ 이동 · Shift/⌘+클릭 다중 선택'}
        </div>
      )}
      {/* 스크롤러(div) 안에 전체 높이만큼의 ul을 두고 보이는 행만 절대 위치로
          얹는다. 스크롤 컨테이너에 패딩을 주는 방식은 패딩 변화가 다시 스크롤
          위치를 흔들어 구간 계산이 어긋났다. */}
      <div className="ilist" ref={listRef}
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}>
        <ul className="ilist-inner" style={{ height: visible.length * ROW_H }}>
        {window_.map((im, wi) => {
          const i = start + wi
          return (
          <li key={im.id} style={{ top: i * ROW_H }}
            className={`${current?.id === im.id ? 'active' : ''}${picked.has(im.id) ? ' picked' : ''}`}
            onClick={(e) => onRowClick(im, i, e)}>
            <img src={api.thumbUrl(im.id)} loading="lazy" alt="" width="44" height="44" />
            <div className="meta">
              <div className="name">{im.file_name}</div>
              <small>
                <span className={`badge ${im.status}`}>{badge[im.status] || '·'}</span>
                {im.ann_count > 0 ? ` ${im.ann_count}개` : ' 라벨 없음'}
                {im.min_conf != null ? ` · conf ${im.min_conf.toFixed(2)}` : ''}
                {im.qa_score != null ? ` · QA ${im.qa_score}` : ''}
                {(im.vlm_flags ?? 0) > 0 ? <span className="vflag"> · 심판 ✗{im.vlm_flags}</span> : null}
              </small>
            </div>
            <button className="x rowdel" title="이미지 삭제"
              onClick={(e) => { e.stopPropagation(); onDelete(im) }}>×</button>
          </li>
          )
        })}
        </ul>
      </div>
    </div>
  )
}

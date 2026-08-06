import { useCallback, useEffect, useRef, useState } from 'react'
import Canvas from './Canvas'
import { api, classColor } from './api'
import { ensembleReviewSummary, mergeAutolabelDrafts } from './autolabel'
import { modelImprovementPlan, modelQuality, rankedClassMetrics, readinessLabel } from './modelQuality'
import { resetEmbed } from './sam'

export default function App() {
  const [projects, setProjects] = useState([])
  const [project, setProject] = useState(null)
  const [images, setImages] = useState([])
  const [current, setCurrent] = useState(null)
  const [anns, setAnns] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [hoverId, setHoverId] = useState(null) // 패널 행 ↔ 캔버스 박스 상호 하이라이트
  const [activeClass, setActiveClass] = useState('')
  const [toast, setToast] = useState(null)
  const [tool, setTool] = useState('box')
  const [trainInfo, setTrainInfo] = useState({ job: { status: 'idle' }, active_model: null })
  const [modelRevision, setModelRevision] = useState(0)
  const [showHelp, setShowHelp] = useState(false)
  const [filter, setFilter] = useState('all')
  const [sortMode, setSortMode] = useState('none')
  const [lastQa, setLastQa] = useState(null)
  const [suggest, setSuggest] = useState([]) // 현재 이미지의 누락 라벨 제안
  const [sampling, setSampling] = useState(null) // 진행 중인 통계 검수 계획
  // 전용 모델 후보를 낮은 임계값+TTA로 더 많이 뽑는다. 자동 승인 기준과는
  // 분리된 검수용 초안 모드라 기본은 빠른 balanced를 유지한다.
  const [recallMode, setRecallMode] = useState(false)
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
  // 필터 술어 맵 — 필터가 늘 때 삼항 체인이 깊어지지 않게
  const FILTER_PRED = {
    all: () => true,
    flagged: (im) => (im.vlm_flags ?? 0) > 0,
  }
  let visible
  if (sampling) {
    // 통계 검수 중에는 뽑힌 표본만, 계획의 순서 그대로 보여준다. 전체
    // 리뷰대기 목록을 그대로 두면 사용자는 표본이 아닌 이미지를 검사하고도
    // 불량 수에 포함하게 되어 검수 통계 자체가 무효가 된다.
    const byId = new Map(images.map((im) => [im.id, im]))
    visible = (sampling.sample_image_ids || []).map((id) => byId.get(id)).filter(Boolean)
  } else {
    visible = images.filter(FILTER_PRED[filter] ?? ((im) => im.status === filter))
    if (sortMode === 'conf') visible = [...visible].sort((a, b) => (a.min_conf ?? 2) - (b.min_conf ?? 2))
    else if (sortMode === 'qa') visible = [...visible].sort((a, b) => (b.qa_score ?? -1) - (a.qa_score ?? -1))
  }
  visibleRef.current = visible
  imagesRef.current = images
  currentRef.current = current
  annsRef.current = anns

  // 읽고 판단해야 하는 안내(배치 진단·심판 결과)는 3초 만에 사라지면 안 된다.
  // sticky는 사용자가 닫을 때까지 남는다.
  const setMsg = useCallback((text, sticky = false) => {
    setToast(text ? { text, sticky } : null)
  }, [])

  // 배치 오토라벨 잡 — 폴링·재진입 복원은 useJob이, 종료 처리만 여기서.
  // 완료가 아닌 종료를 완료로 말하지 않는다. 예전엔 서버 재시작으로 잡 기록이
  // 사라지면 "완료: undefined/undefined장"을 띄웠다 — 절반만 라벨된 데이터를
  // 두고 사용자는 끝난 줄 안다.
  const [job, setJob] = useJob(
    project?.id,
    () => (project ? api.autolabelStatus(project.id) : Promise.resolve({ status: 'idle' })),
    async (s) => {
      setImages(await api.listImages(project.id))
      // 배치가 도는 동안 사용자는 이미지를 옮겨 다닌다 — 완료 시점의 현재
      // 이미지(ref)를 쓰고, 편집 중(dirty)이면 미저장 편집을 덮지 않는다.
      const cur = currentRef.current
      if (cur && !dirty.current) {
        const list = await api.getAnnotations(cur.id)
        if (currentRef.current?.id === cur.id && !dirty.current) {
          const mapped = list.map((a) => ({ ...a, _key: `db-${a.id}` }))
          annsRef.current = mapped
          setAnns(mapped)
        }
      }
      const bad = s.status !== 'completed'
      const text = s.status === 'completed'
        ? (s.advice || `배치 오토라벨 완료: ${s.done}/${s.total}장`)
        : jobEndMessage(s, '배치', '배치 완료')
      // 읽고 판단해야 하는 안내는 사라지지 않게 둔다
      setMsg(text, bad || (s.verdict && s.verdict !== 'good'))
    }, 1500, true)

  // QA 심판 잡 — 결과 본문(result)은 완료 상태에 붙어 온다
  const [qaJob, setQaJob] = useJob(
    project?.id,
    () => (project ? api.qaStatus(project.id) : Promise.resolve({ status: 'idle' })),
    async (s) => {
      setImages(await api.listImages(project.id))
      if (s.result) { setLastQa(s.result); setMsg(qaSummary(s.result), true) }
      else setMsg(`심판 실패: ${s.error || '진행 상황을 잃었습니다 (서버 재시작?) — 다시 실행하세요'}`, true)
    }, 2000)

  useEffect(() => {
    if (!toast || toast.sticky) return
    const t = setTimeout(() => setToast(null), 3000)
    return () => clearTimeout(t)
  }, [toast])

  useEffect(() => {
    if (!project) return
    let cancelled = false
    let timer
    const poll = async () => {
      try {
        const next = await api.trainStatus(project.id)
        if (cancelled) return
        setTrainInfo(next)
        // epoch/ETA는 실제 상태라 실행 중에는 촘촘히, 대기 중에는 조용히 갱신한다.
        timer = setTimeout(poll, next.job?.status === 'running' ? 1000 : 5000)
      } catch {
        if (!cancelled) timer = setTimeout(poll, 5000)
      }
    }
    poll()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [project?.id]) // eslint-disable-line

  // 프로젝트별 추론 정책이므로 다른 프로젝트로 넘어가면 안전한 기본값으로.
  useEffect(() => setRecallMode(false), [project?.id])

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
    // 도는 중인 배치·심판 잡 복원은 useJob 훅이 project.id 변화를 보고 처리한다
    // 첫 이미지 자동 열기 — 빈 캔버스로 시작하지 않게
    if (imgs.length) {
      const first = imgs.find((im) => im.status === 'prelabeled')
        || imgs.find((im) => im.status === 'unlabeled') || imgs[0]
      const list = await api.getAnnotations(first.id)
      setCurrent(first)
      setAnns(list.map((a) => ({ ...a, _key: `db-${a.id}` })))
    }
  }

  const saveAnns = useCallback(async (imageId, list) => {
    // id는 유지해서 보낸다 — 서버가 백그라운드로 채운 meta.vlm(유료 판정)을
    // 이 사본에 없어도 id 기준으로 보존 병합할 수 있게
    const saved = await api.saveAnnotations(imageId, list.map(({ _key, ...a }) => a))
    // 새 어노테이션은 첫 저장 전까지 DB id가 없다. 서버가 돌려준 id를 화면의
    // 안정적인 _key에 연결하지 않으면 다음 자동저장에서 다시 새 행으로 취급돼,
    // 그 사이 백그라운드 VLM이 기록한 판정을 지울 수 있다.
    const byKey = new Map(list.map((a, i) => [a._key, saved.annotations?.[i]]))
    const syncIds = (source) => source.map((a) => {
      const row = byKey.get(a._key)
      return row ? { ...a, id: row.id, meta: row.meta } : a
    })
    // 저장 비행 중 새 편집이 있었으면(annsRef가 이미 다른 참조) dirty를 지우지
    // 않는다 — 지우면 그 편집의 자동저장 타이머와 이탈 경고가 전부 조용히
    // 건너뛰어, "저장됨" 토스트를 보고 닫은 탭에서 마지막 편집이 사라진다
    if (annsRef.current === list) {
      const synced = syncIds(list)
      annsRef.current = synced
      setAnns(synced)
      dirty.current = false
    } else {
      // 비행 중 편집 내용은 유지하되, 방금 생긴 DB id만 _key 기준으로 합친다.
      // 이 id가 있어야 다음 저장이 INSERT가 아니라 UPDATE가 된다.
      const synced = syncIds(annsRef.current)
      annsRef.current = synced
      setAnns(synced)
    }
    // 사이드바 메타(개수·최저 conf)도 즉시 맞춘다 — 박스를 지웠는데 옛 숫자가
    // 남아 있으면 검수 우선순위 판단이 틀어진다
    const confs = list.map((a) => a.confidence).filter((c) => c != null)
    const minConf = confs.length ? Math.min(...confs) : null
    setImages((imgs) => {
      const im = imgs.find((x) => x.id === imageId)
      // 값이 그대로면 배열 참조를 유지한다 — 2초 자동저장마다 목록 전체가
      // 리렌더되지 않게 (박스 이동만 한 저장이 대부분)
      if (!im || (im.ann_count === list.length && im.min_conf === minConf)) return imgs
      return imgs.map((x) => (x.id === imageId
        ? { ...x, ann_count: list.length, min_conf: minConf } : x))
    })
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
      if (target) await openImage(target)
      return setMsg(`${entry.was === 'approved' ? '승인' : '거부'} 취소${label}`)
    }
    // 어노테이션 되돌리기 — 대상 이미지를 "무조건" 다시 연다. currentRef가
    // 이미 대상을 가리켜도 화살표 이동의 openImage가 비행 중이면 currentRef가
    // 낡은 값이라, 지름길로 복구한 직후 비행 중이던 fetch가 화면을 덮는다
    // (실측: DB엔 복구됐는데 화면은 다음 이미지의 빈 캔버스). openImage는
    // 순번(openSeq)을 올려 비행 중인 열기를 무효화하므로 이게 유일한 안전 경로다.
    if (!target) return setMsg('되돌릴 이미지를 찾을 수 없습니다')
    await openImage(target)
    // 그 사이 더 최신 이동이 시작됐으면 여기서 적용하면 안 된다 —
    // 엉뚱한 이미지에 이전 어노테이션이 저장된다. 되밀어 넣고 중단.
    if (currentRef.current?.id !== entry.imageId) {
      undoStack.current.push(entry)
      return setMsg('이미지 이동과 겹쳐 되돌리기를 중단했습니다 — 다시 Cmd+Z')
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
      else if ((e.key === 'a' || e.key === 'A') && !sampling) setStatus('approved')
      else if ((e.key === 'x' || e.key === 'X') && !sampling) setStatus('rejected')
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

  if (!project) {
    return (
      <div className="page">
        <h2>오토라벨</h2>
        <p className="hint">프로젝트를 선택하거나 새로 만드세요. 흐름: 클래스 정의 → 이미지 업로드 → 오토라벨 → 리뷰 → (자동) 전용 모델 학습</p>
        <ProjectPicker projects={projects} onOpen={openProject} onDeleted={refreshProjects} />
      </div>
    )
  }

  const approved = images.filter((im) => im.status === 'approved').length
  const activeQuality = trainInfo.active_model ? modelQuality(trainInfo.active_model) : null

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
          <span className="chip danger" role="alert" title="서버에 저장하지 못했습니다. 5초마다 재시도합니다 — 창을 닫으면 변경이 사라집니다">
            ⚠ 저장 안 됨 — 재시도 중
          </span>
        )}
        {trainInfo.active_model && (
          <span className={`chip model-${activeQuality.tone}`}
            title={trainInfo.active_model.meta?.quality_reason || activeQuality.label}>
            {activeQuality.label} · mAP50 {trainInfo.active_model.map50?.toFixed(2) ?? '—'}
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
                const j = await api.autolabelBatch(project.id, {
                  masks: false,
                  profile: trainInfo.active_model && recallMode ? 'recall' : 'balanced',
                })
                setJob(j)
                // 대상 0장이면 폴링이 돌지 않아 안내가 사라진다 — 즉시 알린다
                if (j.status !== 'running') setMsg(j.advice || '실행할 이미지가 없습니다')
              } catch (e) { setMsg(`배치 시작 실패: ${e.message}`) }
            }}
          />
          {!trainInfo.active_model && (
            <FoundationProfile projectId={project.id} approved={approved}
              jobStatus={job?.status} />
          )}
          {sampling && (
            <SamplingPanel plan={sampling} current={current} sampleImages={visible}
              onOpen={openImage} onClose={() => setSampling(null)}
              onSubmit={async (defects) => {
                try {
                  const r = await api.acceptanceResult(project.id, {
                    sample_size: sampling.sample_size, defects,
                    max_defects: sampling.max_defects,
                    target_error_rate: sampling.target_error_rate,
                    confidence: sampling.confidence,
                    status: sampling.status,
                    lot_token: sampling.lot_token,
                  })
                  setImages(await api.listImages(project.id))
                  setSampling(null)
                  setFilter(r.accepted ? 'approved' : 'prelabeled')
                  setMsg(r.message + (r.approved_images ? ` · ${r.approved_images}장 승인됨` : ''), !r.accepted)
                } catch (e) {
                  setMsg(`배치 판정 실패: ${e.message} · 배치가 바뀌었다면 취소 후 새 계획을 만드세요`, true)
                }
              }} />
          )}

          <TrainPanel trainInfo={trainInfo} approved={approved} pid={project.id} onMsg={setMsg}
            modelRefreshKey={modelRevision}
            onModelChange={async () => {
              setTrainInfo(await api.trainStatus(project.id))
              setModelRevision((revision) => revision + 1)
            }}
            onTrigger={async () => {
              setTrainInfo({ ...trainInfo, job: await api.triggerTrain(project.id) })
            }} />

          {/* 설정·도구는 접힌다. 예전엔 이게 전부 펼쳐진 채 이미지 목록 위에
              쌓여 있어서 좌측 패널이 10화면 높이가 됐고, 정작 라벨링에 매번
              쓰는 이미지 목록은 화면 밖에서 시작했다. */}
          <button className="setup-toggle" aria-expanded={setupOpen} aria-controls="setup-tools"
            onClick={() => setSetupOpen(!setupOpen)}>
            <span>{setupOpen ? '▾' : '▸'} 설정 · 도구</span>
            <small>클래스 · 업로드 · 오토라벨 · 익스포트</small>
          </button>
          {setupOpen && (
            <div className="setup-body" id="setup-tools">
          <OntologyEditor project={project} setProject={setProject} />
          <UploadBox project={project} onMsg={setMsg}
            onUploaded={async () => setImages(await api.listImages(project.id))} />
          <VideoUpload project={project} onMsg={setMsg}
            onDone={async () => setImages(await api.listImages(project.id))} />
          <LinkImport project={project} onMsg={setMsg}
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
              <button className="primary" disabled={job?.status === 'running'}
                title="전 이미지에 모델이 라벨 초안을 생성합니다 (덮어쓰지 않고 모델 라벨만 갱신)"
                onClick={async () => {
                  if (!project.ontology.length) return setMsg('클래스를 먼저 정의하세요')
                  try {
                    const j = await api.autolabelBatch(project.id, {
                      masks: false,
                      profile: trainInfo.active_model && recallMode ? 'recall' : 'balanced',
                    })
                    setJob(j)
                    if (j.status !== 'running') setMsg(j.advice || '실행할 이미지가 없습니다')
                  } catch (e) { setMsg(`배치 시작 실패: ${e.message}`) }
                }}>
                {job?.status === 'running' ? `오토라벨 ${job.done ?? 0}/${job.total}` : '▶ 전체 오토라벨'}
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
                      return t?.tau != null ? {
                        ...c,
                        threshold: t.tau,
                        approval_threshold: t.tau,
                        approval_precision: t.precision,
                        approval_support: t.support,
                        approval_source: 'qa_val',
                        approval_model_id: lastQa.model_id,
                      } : c
                    })
                    await api.saveOntology(project.id, next)
                    setProject({ ...project, ontology: next })
                    setMsg('QA 검증 임계값 적용됨 — 다음 오토라벨과 검증 모델 자동 승인에 반영')
                  }}>✓ QA 임계값 적용</button>
              </div>
            )}
            <div className="row">
              <button className="ok" title="홀드아웃 검증 모델이 예측하고 QA에서 클래스별 정밀도 95%를 확인한 초안만 승인합니다"
                onClick={async () => {
                  const dry = await api.autoApprove(project.id, { min_conf: 0.7, dry_run: true })
                  if (!dry.approved) return setMsg(
                    dry.blocked_reason || `자동 승인 대상 없음 (대기 ${dry.pending}장)`, true)
                  if (!confirm(
                    `리뷰 대기 ${dry.pending}장 중 검증 조건을 통과한 ${dry.approved}장을 승인할까요?\n` +
                    `(저신뢰 ${dry.skipped_low_confidence}장 · 미검증 ${dry.skipped_unsafe_model}장 · 미캘리브레이션 ${dry.skipped_uncalibrated}장은 남깁니다)`
                  )) return
                  const r = await api.autoApprove(project.id, { min_conf: 0.7 })
                  setImages(await api.listImages(project.id))
                  setMsg(`${r.approved}장 검증 자동 승인 (커버리지 ${(r.coverage * 100).toFixed(0)}%) — 나머지만 직접 리뷰하세요`)
                }}>⚡ 검증 모델 자동 승인</button>
              <button title="전수 검사 대신 통계적으로 필요한 만큼만 검사해 배치를 승인합니다"
                onClick={async () => {
                  const p = await api.acceptancePlan(project.id)
                  if (!p.sample_size) return setMsg('리뷰 대기 이미지 없음')
                  setSampling(p)
                  const first = images.find((im) => im.id === p.sample_image_ids?.[0])
                  if (first) await openImage(first)
                  setMsg(`검수 계획: ${p.lot_size}장 중 ${p.sample_size}장만 검사 ` +
                    `(불량 ${p.max_defects}개까지 허용, 검수 ${(p.saving * 100).toFixed(0)}% 절감). ` +
                    `표본만 표시했습니다. 각 장을 정상/오류로 판정하세요`)
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
            </div>
          </div>
            </div>
          )}
          <ImageList
            visible={visible} current={current} onOpen={openImage}
            sampleMode={!!sampling}
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
            {trainInfo.active_model && (
              <button className={recallMode ? 'active' : ''}
                title="낮은 후보 임계값(0.10)+증강 추론으로 누락을 줄입니다. 후보와 검수량은 늘고 속도는 느려집니다. 자동 승인 기준은 바뀌지 않습니다."
                onClick={() => setRecallMode((v) => !v)}>
                {recallMode ? '◎ 누락 최소화 ON' : '○ 누락 최소화'}
              </button>
            )}
            <button onClick={async () => {
              if (!current) return
              // 추론은 수 초 걸리고 그 사이 →로 이미지를 넘기는 게 리뷰의 기본
              // 동선이다. 응답이 늦게 오면 클릭 시점 이미지가 아닌 지금 이미지에
              // 검출이 합쳐져 자동저장이 엉뚱한 라벨을 서버에 덮어쓴다 — 요청
              // 이미지가 그대로일 때만 반영한다.
              const iid = current.id
              setMsg('오토라벨 중…')
              const r = await api.autolabelOne(iid, project.ontology, {
                profile: trainInfo.active_model && recallMode ? 'recall' : 'balanced',
              })
              if (currentRef.current?.id !== iid) {
                return setMsg('이미지를 이동해 오토라벨 결과를 버렸습니다 — 그 이미지에서 다시 실행하세요', true)
              }
              const merged = mergeAutolabelDrafts(
                annsRef.current,
                r.detections.map((d, i) => ({
                  ...d, _key: `auto-${Date.now()}-${i}`, source: 'model',
                })))
              setAnnsDirty(merged.annotations)
              const note = [
                merged.replaced ? `기존 초안 ${merged.replaced}개 교체` : '',
                merged.suppressed ? `사람 라벨 중복 ${merged.suppressed}개 억제` : '',
              ].filter(Boolean).join(' · ')
              setMsg(`오토라벨 ${r.detections.length - merged.suppressed}개 (${r.engine.split('(')[0]})${note ? ` · ${note}` : ''}`)
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
            <button className="ok" disabled={!!sampling} onClick={() => setStatus('approved')}
              title={sampling ? '표본 검수 중에는 왼쪽의 정상/오류 판정을 사용하세요' : '승인 후 다음 (A)'}>✓ 승인</button>
            <button className="bad" disabled={!!sampling} onClick={() => setStatus('rejected')}
              title={sampling ? '표본 검수 중에는 왼쪽의 정상/오류 판정을 사용하세요' : '거부 후 다음 (X)'}>✗ 거부</button>
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
        <div className={`toast${toast.sticky ? ' sticky' : ''}`} role="status"
          aria-live="polite" aria-atomic="true">
          <span>{toast.text}</span>
          {toast.sticky && <button className="x" title="닫기" onClick={() => setToast(null)}>×</button>}
        </div>
      )}
      {showHelp && <HelpOverlay onClose={() => setShowHelp(false)} />}
    </div>
  )
}

const MIN_APPROVED = 8

const TRAIN_PHASES = {
  starting: '학습 준비', export: '데이터 분할·내보내기', training: '모델 학습',
  validation: 'validation 검증', holdout: 'holdout 검증', gating: '기존 모델 비교',
  completed: '적용 결정',
}

function trainingPhaseLabel(phase) {
  return TRAIN_PHASES[phase] || '상태 확인 중'
}

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
  const quality = model ? modelQuality(model) : null
  const training = trainInfo.job?.status === 'running'

  let step
  if (!project.ontology.length || project.ontology.some((c) => !c.name))
    step = { n: 1, title: '클래스를 정의하세요', desc: '찾을 객체 이름 + 검출 프롬프트(영문). 예: helmet / safety helmet' }
  else if (!images.length)
    step = { n: 2, title: '이미지를 업로드하세요', desc: '여러 장 한번에 선택 가능' }
  else if (job?.status === 'running')
    step = { n: 3, title: `오토라벨 중… ${job.done ?? 0}/${job.total}`, desc: '완료되면 리뷰 대기로 넘어갑니다' }
  else if (!labeled)
    step = { n: 3, title: '전체 오토라벨을 실행하세요', desc: '모델이 전 이미지에 라벨 초안을 깝니다', action: { label: '▶ 전체 오토라벨 실행', fn: onAutolabel } }
  else if (pending)
    step = { n: 4, title: `리뷰 ${pending}장 남음`, desc: '맞으면 A(승인), 틀리면 고친 뒤 A. 필요 없으면 X(거부)' }
  else if (training)
    step = { n: 5, title: `전용 모델 학습 중${trainInfo.job.epoch ? ` · ${trainInfo.job.epoch}/${trainInfo.job.epochs} epoch` : '…'}`,
      desc: `${trainingPhaseLabel(trainInfo.job.phase)} — 학습센터에서 진행률과 남은 시간을 확인하세요` }
  else if (approved < MIN_APPROVED)
    step = { n: 5, title: `전용 모델까지 승인 ${approved}/${MIN_APPROVED}장`, desc: `${MIN_APPROVED - approved}장 더 승인하면 자동으로 학습이 시작됩니다` }
  else if (!model)
    step = { n: 5, title: '학습 준비 완료', desc: '"지금 학습"을 누르거나 승인을 더 쌓으세요' }
  else if (!quality.usable)
    step = { n: 5, title: `활성 모델 ${quality.label}`, desc: '학습센터에서 성능표가 포함된 Colab 결과를 가져오고 검증된 후보를 적용하세요' }
  else
    step = { n: 6, title: `${quality.label} 가동 중 (mAP50 ${model.map50?.toFixed(2) ?? '—'})`, desc: '이미지를 더 넣고 오토라벨 → 리뷰를 반복하면 정확도가 계속 올라갑니다' }

  return (
    <div className="nextstep">
      <div className="ns-head">지금 할 일 · {step.n}단계</div>
      <div className="ns-title">{step.title}</div>
      <div className="ns-desc">{step.desc}</div>
      {job?.status === 'completed' && job.engine_plan?.both_engine_images > 0 && (
        <div className={`engine-plan-result ${job.engine_plan.mode}`}>
          <b>두 엔진 교차 검출 {job.engine_plan.both_engine_images}장</b>
          <span>{job.engine_plan.seeded_before < job.engine_plan.seed_target
            ? `비교 표본 ${job.engine_plan.seed_target}장을 채우는 중입니다. 승인하면 클래스별 엔진 선택 근거가 됩니다.`
            : `${job.engine_plan.explore_every}장마다 1장은 두 엔진을 다시 돌려 선택 근거를 갱신합니다.`}</span>
        </div>
      )}
      {step.action && (
        <button className="primary" style={{ marginTop: 8, width: '100%' }}
          onClick={step.action.fn}>{step.action.label}</button>
      )}
    </div>
  )
}

const FOUNDATION_LABELS = {
  comparing: ['비교 중', 'neutral'],
  sam3: ['SAM3', 'sam3'],
  gdino: ['GDINO', 'gdino'],
  ensemble: ['둘 함께', 'ensemble'],
}

function FoundationProfile({ projectId, approved, jobStatus }) {
  const [profile, setProfile] = useState(null)

  useEffect(() => {
    let cancelled = false
    api.foundationProfile(projectId)
      .then((result) => { if (!cancelled) setProfile(result) })
      .catch(() => { if (!cancelled) setProfile(null) })
    return () => { cancelled = true }
  }, [projectId, approved, jobStatus])

  if (!profile) return null
  const ready = profile.status === 'ready'
  const baseReviewed = Math.min(profile.reviewed_images, profile.required_images)
  const reviewLabel = profile.reviewed_images > profile.required_images
    ? `기본 ${baseReviewed}/${profile.required_images}장 · 누적 ${profile.reviewed_images}장`
    : `기본 ${baseReviewed}/${profile.required_images}장`
  return (
    <div className={`card foundation-profile ${profile.status}`}>
      <div className="foundation-head">
        <b>초기 엔진 학습</b>
        <span>{reviewLabel}</span>
      </div>
      <div className="progress mini" aria-label={`기본 교차 검수 ${baseReviewed}/${profile.required_images}장`}>
        <div className="progress-fill" style={{
          width: `${Math.min(100, profile.reviewed_images / profile.required_images * 100)}%`,
        }} />
      </div>
      <p className="foundation-guide">
        {ready
          ? '다음 배치부터 클래스별로 검수 작업이 적었던 엔진을 자동 사용합니다.'
          : profile.remaining_images
            ? `교차 시험 후보를 고쳐 승인하세요. ${profile.remaining_images}장 더 보면 자동 선택합니다.`
            : '정답 객체가 적은 클래스는 표본을 더 승인하는 동안 계속 비교합니다.'}
      </p>
      {profile.reviewed_images > 0 && (
        <div className="foundation-classes">
          {profile.classes.map((item) => {
            const [label, tone] = FOUNDATION_LABELS[item.selection] || FOUNDATION_LABELS.comparing
            return (
              <div className="foundation-class" key={item.name}>
                <span title={`정답 객체 ${item.support}개`}>{item.name}</span>
                <b className={`foundation-choice ${tone}`}>{label}</b>
              </div>
            )
          })}
        </div>
      )}
      <small>기준: 오검출 삭제 1 · 누락 새로 그리기 3</small>
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
  const ensemble = ensembleReviewSummary(anns)
  const ensembleUi = {
    consensus: { label: '둘 합의', title: 'SAM3와 Grounding DINO가 같은 클래스·객체로 합의' },
    sam3_only: { label: 'S만', title: 'SAM3만 찾은 후보 — 우선 검수' },
    gdino_only: { label: 'G만', title: 'Grounding DINO만 찾은 후보 — 우선 검수' },
  }
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
      {ensemble.total > 0 && (
        <div className="ensemble-review-summary" role="status">
          <div><span className="ensemble-chip consensus">둘 합의 {ensemble.consensus}</span>
            <span className="ensemble-chip solo">단독 {ensemble.reviewFirst}</span></div>
          <small>{ensemble.reviewFirst ? '단독 후보가 목록 앞쪽입니다 · 먼저 수정·삭제하세요' : '두 모델이 모든 후보에 합의했습니다 · 표본 확인은 필요합니다'}</small>
        </div>
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
          {a.meta?.ensemble?.agreement && (
            <span className={`ensemble-chip row ${a.meta.ensemble.agreement}`}
              title={ensembleUi[a.meta.ensemble.agreement]?.title || '앙상블 후보'}>
              {ensembleUi[a.meta.ensemble.agreement]?.label || '후보'}
            </span>
          )}
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
  const dialogRef = useRef(null)
  const returnFocus = useRef(document.activeElement)
  useEffect(() => {
    const dialog = dialogRef.current
    const focusTarget = returnFocus.current
    dialog?.focus()
    const trap = (e) => {
      if (e.key !== 'Tab' || !dialog) return
      const focusable = [...dialog.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')]
        .filter((el) => !el.disabled)
      if (!focusable.length) return
      const first = focusable[0], last = focusable[focusable.length - 1]
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
    }
    dialog?.addEventListener('keydown', trap)
    return () => {
      dialog?.removeEventListener('keydown', trap)
      focusTarget?.focus?.()
    }
  }, [])
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
      <div className="help" role="dialog" aria-modal="true" aria-labelledby="help-title"
        tabIndex={-1} ref={dialogRef} onClick={(e) => e.stopPropagation()}>
        <div className="panel-title" id="help-title">단축키</div>
        <table><tbody>
          {keys.map(([k, v]) => <tr key={k}><td><kbd>{k}</kbd></td><td>{v}</td></tr>)}
        </tbody></table>
        <button onClick={onClose}>닫기 (Esc)</button>
      </div>
    </div>
  )
}

function ProjectPicker({ projects, onOpen, onDeleted }) {
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
                onDeleted()
              }}>×</button>
          </li>
        ))}
      </ul>
    </div>
  )
}

function OntologyEditor({ project, setProject }) {
  const [rows, setRows] = useState(project.ontology)
  const [state, setState] = useState('saved')
  const rowsRef = useRef(project.ontology)
  const dirtyRef = useRef(false)
  const timerRef = useRef(null)
  const saveQueue = useRef(Promise.resolve())
  // 임계값 적용·프롬프트 실험 등 에디터 밖에서 온톨로지가 바뀌면 로컬
  // 스냅샷도 따라간다 — 안 하면 다음 키 입력의 save가 이전 스냅샷 전체를
  // PUT해 그 변경을 조용히 되돌린다
  const persist = useCallback((snapshot) => {
    setState('saving')
    // 요청을 직렬화한다. 키 입력 A의 느린 응답이 더 최신 B를 나중에 덮는
    // 온톨로지 전체 PUT 레이스를 막는다.
    const run = saveQueue.current.catch(() => {}).then(
      () => api.saveOntology(project.id, snapshot))
    saveQueue.current = run
    run.then(() => {
      if (rowsRef.current === snapshot) {
        dirtyRef.current = false
        setState('saved')
      }
    }).catch(() => {
      if (rowsRef.current === snapshot) setState('error')
    })
    return run
  }, [project.id])
  useEffect(() => {
    rowsRef.current = project.ontology
    dirtyRef.current = false
    setRows(project.ontology)
    setState('saved')
    return () => {
      clearTimeout(timerRef.current)
      // 프로젝트를 즉시 나가도 마지막 편집을 직렬 큐 끝에 붙여 보낸다.
      if (dirtyRef.current) {
        const snapshot = rowsRef.current
        saveQueue.current = saveQueue.current.catch(() => {}).then(
          () => api.saveOntology(project.id, snapshot))
      }
    }
  }, [project.id]) // eslint-disable-line
  const update = (next) => {
    rowsRef.current = next
    dirtyRef.current = true
    setRows(next)
    setProject((p) => ({ ...p, ontology: next }))
    setState('dirty')
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => persist(next), 500)
  }
  const save = () => {
    clearTimeout(timerRef.current)
    return persist(rowsRef.current)
  }
  return (
    <details open>
      <summary>클래스 ({rows.length})</summary>
      <div className="hint">클래스명 · 검출 프롬프트(영문 권장) · 임계값</div>
      {rows.map((c, i) => (
        <div className="row" key={i}>
          <i className="dot" style={{ background: classColor(rows, c.name) }} />
          <input style={{ width: 66 }} value={c.name} placeholder="클래스"
            onChange={(e) => update(rows.map((r, j) => (j === i ? { ...r, name: e.target.value } : r)))} />
          <input style={{ flex: 1, minWidth: 60 }} value={c.prompt} placeholder="프롬프트"
            onChange={(e) => update(rows.map((r, j) => (j === i ? { ...r, prompt: e.target.value } : r)))} />
          <input style={{ width: 58 }} type="number" step="0.05" min="0" max="1" value={c.threshold}
            onChange={(e) => update(rows.map((r, j) => (j === i ? { ...r, threshold: +e.target.value } : r)))} />
          <button className="x" onClick={() => update(rows.filter((_, j) => j !== i))}>×</button>
        </div>
      ))}
      <div className="row ontology-actions">
        <button onClick={() => update([...rows, { name: '', prompt: '', threshold: 0.35 }])}>+ 클래스</button>
        <button className={state === 'dirty' || state === 'error' ? 'primary' : ''}
          disabled={state === 'saved' || state === 'saving'} onClick={save}>
          {state === 'saving' ? '저장 중…' : state === 'saved' ? '저장됨' : '변경 저장'}
        </button>
        {state === 'error' && <small className="bad-text" role="alert">저장 실패 — 다시 시도하세요</small>}
      </div>
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

// 백그라운드 잡 폴링 + 재진입 복원 훅 — 심판·비디오가 같은 골격을 복사하며
// 이미 갈라졌다 (비디오만 idle 분기 누락). 종료 처리(onDone)만 각자 전달한다.
// 새로 연 화면에서 버튼이 '실행' 대기로 보이면 진행 중인 작업을 이중 실행하거나
// 죽은 줄 알기 때문에, 마운트 시 running이면 복원한다 (논블로킹).
function useJob(projectId, fetchStatus, onFinish, interval = 1500, restoreFinished = false) {
  const [job, setJob] = useState(null)

  useEffect(() => {
    setJob(null) // 프로젝트 전환 시 이전 프로젝트 잡 표시 제거
    fetchStatus().then((s) => {
      if (s.status === 'running' || (restoreFinished && s.status !== 'idle')) setJob(s)
    }).catch(() => {})
  }, [projectId]) // eslint-disable-line

  useEffect(() => {
    if (job?.status !== 'running') return
    const t = setInterval(async () => {
      const s = await fetchStatus()
      setJob(s)
      if (s.status !== 'running') {
        clearInterval(t)
        onFinish(s)
      }
    }, interval)
    return () => clearInterval(t)
  }, [job?.status]) // eslint-disable-line

  return [job, setJob]
}

// 완료가 아닌 종료를 완료로 말하지 않는다 — 잡 공통 종료 메시지 규칙
function jobEndMessage(s, label, fallback) {
  return s.status === 'failed' ? `${label} 실패: ${s.error}`
    : s.status === 'interrupted' ? `${label} 중단됨 (${s.done ?? 0}/${s.total ?? '?'} 처리) — ${s.error || '다시 실행하세요'}`
      : s.status === 'idle' ? `${label} 진행 상황을 잃었습니다 (서버 재시작?) — 다시 실행하세요`
        : s.advice || fallback
}

// 비디오 → 프레임 추출 + SAM 3 전파 트래킹. 프레임은 일반 이미지로 등록되어
// 리뷰·학습·익스포트 레인을 그대로 탄다. 한 프레임에서 잡힌 객체가 메모리
// 전파로 이어지므로 영상 데이터는 라벨 비용이 "전 프레임"에서 "검수"로 준다.
function VideoUpload({ project, onMsg, onDone }) {
  const [stride, setStride] = useState(5)
  const [job, setJob] = useJob(project.id, () => api.videoStatus(project.id), (s) => {
    onDone()
    onMsg(jobEndMessage(s, '비디오 처리', '비디오 처리 완료'), true)
  })

  const running = job?.status === 'running'
  // 클래스는 vupload — .upload를 쓰면 이미지 업로드 셀렉터(label.upload)와 겹친다
  return (
    <label className="vupload" title="mp4/mov 업로드 → 프레임 추출 → SAM 3가 첫 검출을 메모리 전파로 추적해 프레임별 초안 라벨 생성">
      {running
        ? `🎬 ${job.phase === 'extract' ? '프레임 추출 중…' : `트래킹 중 ${job.done ?? 0}/${job.total ?? '?'}`}`
        : <>🎬 비디오 임포트 (프레임 추출 + 자동 트래킹) · <select value={stride} onClick={(e) => e.preventDefault()}
            onChange={(e) => setStride(+e.target.value)}>
            {[2, 5, 10, 15].map((s) => <option key={s} value={s}>{s}프레임마다</option>)}
          </select></>}
      <input type="file" accept="video/mp4,video/quicktime,video/*" hidden disabled={running}
        onChange={async (e) => {
          const f = e.target.files[0]
          e.target.value = ''
          if (!f) return
          try {
            setJob(await api.uploadVideo(project.id, f, stride))
          } catch (err) { onMsg(`비디오 업로드 실패: ${err.message}`, true) }
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
      <button className="panel-title disclosure" aria-expanded={open} onClick={() => {
        setOpen(!open)
        if (!open && !text) {
          // 현재 프롬프트를 첫 줄에 두어 "지금보다 나은가"를 바로 볼 수 있게
          const cur = target?.prompt || target?.name || ''
          setText([cur, target?.name, `a photo of ${target?.name}`]
            .filter((v, i, a) => v && a.indexOf(v) === i).join('\n'))
        }
      }}>
        {open ? '▾' : '▶'} 프롬프트 실험 (어떤 표현이 잘 잡히나)
      </button>
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
  const [job, setJob] = useJob(project.id, () => api.vlmStatus(project.id), (s) => {
    onDone()
    onMsg(jobEndMessage(s, '문맥 심판', '문맥 심판 완료'), true)
  })

  useEffect(() => { setRubric(project.rubric || ''); setSaved(true) }, [project.id]) // eslint-disable-line

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
function LinkImport({ project, onMsg, onDone }) {
  const [open, setOpen] = useState(false)
  const [imagesDir, setImagesDir] = useState('')
  const [labelsDir, setLabelsDir] = useState('')
  const [cocoJson, setCocoJson] = useState('')
  const [limit, setLimit] = useState('')
  const [requireClass, setRequireClass] = useState('')
  const [info, setInfo] = useState(null)
  // 5만 장 중 400장만 연결된 걸 "임포트 완료: 400장"으로 알리면 안 된다 —
  // 종료 메시지 규칙은 jobEndMessage가 공통 담당
  const [job, setJob] = useJob(project.id, () => api.importStatus(project.id), (s) => {
    onDone()
    const text = s.status === 'completed'
      ? `임포트 완료: ${s.done}장 연결됨` : jobEndMessage(s, '임포트', '임포트 완료')
    onMsg(text, s.status !== 'completed')
  }, 1000)

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
    </details>
  )
}

// Colab 결과를 후보로 검증한 뒤 사용자가 별도로 적용한다.
function ModelImport({ pid, onMsg, onChanged }) {
  const [path, setPath] = useState('')
  const [file, setFile] = useState(null)
  const [candidate, setCandidate] = useState(null)
  const [busy, setBusy] = useState(false)
  const quality = candidate
    ? modelQuality({ meta: { quality_status: candidate.quality_status } }) : null
  const classRows = rankedClassMetrics(candidate?.class_metrics)
  const acceptCandidate = async (load) => {
    setBusy(true)
    try {
      const result = await load()
      setCandidate(result)
      onMsg(`모델 후보 등록 · ${result.quality_reason}`, result.quality_status !== 'verified')
      await onChanged?.()
    } catch (e) { onMsg(`모델 가져오기 실패: ${e.message}`, true) }
    finally { setBusy(false) }
  }
  return (
    <div className="model-import">
      <div className="model-import-title"><b>3. Colab 결과 가져오기</b><small>등록 후 검증 결과를 보고 적용합니다</small></div>
      <div className="row model-file-row">
        <input className="model-file-input" aria-label="Colab 모델 번들 파일" type="file"
          accept=".zip,.pt" onChange={(e) => { setFile(e.target.files?.[0] || null); setCandidate(null) }} />
        <button disabled={!file || busy}
          onClick={() => acceptCandidate(() => api.importModelFile(pid, file))}>
          {busy ? '검증 중…' : '선택 파일 검증'}</button>
      </div>
      <div className="hint">Colab에서 받은 <code>autolabel-model.zip</code>을 선택하세요. 기존 .pt는 성능표가 없어 검증 필요 후보로만 등록됩니다.</div>
      <details className="model-path-import">
        <summary>로컬 경로 직접 입력</summary>
        <div className="row">
          <input aria-label="Colab 모델 번들 경로" placeholder="/Users/.../autolabel-model.zip"
            value={path} onChange={(e) => { setPath(e.target.value); setCandidate(null) }} />
          <button disabled={!path.trim() || busy}
            onClick={() => acceptCandidate(() => api.importModel(pid, { path: path.trim() }))}>
            경로 검증</button>
        </div>
      </details>
      {candidate && (
        <div className={`model-candidate model-${quality.tone}`} role="status">
          <div className="model-candidate-head"><b>{quality.label}</b><span>후보 #{candidate.id}</span></div>
          <div>{candidate.quality_reason}</div>
          <div className="model-candidate-metrics">
            <span>val mAP50 <b>{metric(candidate.metrics?.val_map50)}</b></span>
            <span>test mAP50 <b>{metric(candidate.metrics?.test_map50)}</b></span>
            <span>분할 <b>{candidate.split_counts?.train}/{candidate.split_counts?.val}/{candidate.split_counts?.test}</b></span>
          </div>
          {classRows.length > 0 && <div className="model-class-metrics">
            <small>클래스별 test mAP50 · 낮은 순</small>
            <div>{classRows.map((row) => <span key={row.name}
              className={row.test_map50 < 0.1 ? 'weak' : ''}>
              <b>{row.name}</b><strong>{metric(row.test_map50)}</strong>
              <em>{row.test_instances}개 결함</em>
            </span>)}</div>
          </div>}
          {quality.usable ? (
            <button className="primary" disabled={candidate.active} onClick={async () => {
              try {
                await api.activateModel(pid, candidate.id)
                setCandidate({ ...candidate, active: true })
                await onChanged?.()
                onMsg(`${quality.label} #${candidate.id}을 오토라벨에 적용했습니다`)
              } catch (e) { onMsg(`모델 적용 실패: ${e.message}`, true) }
            }}>{candidate.active ? '✓ 적용됨' : '이 후보를 오토라벨에 적용'}</button>
          ) : <div className="model-blocked">적용 차단 · 라벨을 보강해 Colab 학습을 다시 실행하세요</div>}
        </div>
      )}
    </div>
  )
}

const TRAIN_STEPS = ['데이터 준비', '모델 학습', '성능 검증', '기존 모델 비교', '적용 결정']

function trainStepIndex(job) {
  if (job.status === 'completed') return 4
  if (['starting', 'export'].includes(job.phase)) return 0
  if (job.phase === 'training') return 1
  if (['validation', 'holdout'].includes(job.phase)) return 2
  if (job.phase === 'gating') return 3
  return 0
}

function trainOverallProgress(job) {
  if (job.status === 'completed') return 100
  if (job.phase === 'starting') return 2
  if (job.phase === 'export') return 8
  if (job.phase === 'training') return Math.round(10 + (job.progress || 0) * 62)
  if (job.phase === 'validation') return 78
  if (job.phase === 'holdout') return 86
  if (job.phase === 'gating') return 94
  return 0
}

function formatDuration(seconds) {
  if (seconds == null) return '계산 중'
  const sec = Math.max(0, Math.round(seconds))
  if (sec < 60) return `${sec}초`
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}분 ${sec % 60}초`
  return `${Math.floor(min / 60)}시간 ${min % 60}분`
}

function metric(value) {
  return value == null ? '—' : Number(value).toFixed(3)
}

function ModelImprovementCard({ model }) {
  const plan = modelImprovementPlan(model)
  if (!plan) return null
  const titleId = `model-plan-${model.id || 'latest'}`
  return (
    <section className={`model-prescription model-${plan.tone}`} aria-labelledby={titleId}>
      <div className="model-prescription-head">
        <div><small>최신 후보 #{model.id}</small><b id={titleId}>{plan.title}</b></div>
        <span>{plan.actions.length ? `${plan.actions.length}개 조치` : '운영 검수 단계'}</span>
      </div>
      <p>{plan.summary}</p>
      {plan.actions.length > 0 && (
        <div className="model-prescription-actions">
          {plan.actions.map((action) => (
            <div key={`${action.className}-${action.kind}`} className={`prescription-action ${action.kind}`}>
              <div className="prescription-action-head">
                <b>{action.className}</b><span>{action.reason}</span>
              </div>
              <div className="prescription-target">
                <strong>승인 라벨 {action.recommendedImages}장부터</strong>
                {action.score != null && <small>test mAP50 {metric(action.score)}</small>}
                {action.instances != null && <small>시험 결함 {action.instances}개</small>}
              </div>
              <p>{action.guidance}</p>
            </div>
          ))}
        </div>
      )}
      <div className="prescription-holdout"><b>평가 데이터 고정</b><span>{plan.holdoutNote}</span></div>
      <div className="prescription-next"><span>다음 순서</span><b>{plan.nextStep}</b></div>
      {plan.actions.length > 0 && <small className="prescription-disclaimer">권장 장수는 성능 보장값이 아닌 다음 실험의 시작량입니다.</small>}
    </section>
  )
}

function LatestModelDiagnosis({ pid, refreshKey }) {
  const [latest, setLatest] = useState(undefined)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    let alive = true
    setFailed(false)
    api.listModels(pid).then((models) => {
      if (alive) setLatest(models[0] || null)
    }).catch(() => { if (alive) setFailed(true) })
    return () => { alive = false }
  }, [pid, refreshKey])
  if (failed) return <div className="warn-text model-plan-load">최신 모델 진단을 불러오지 못했습니다.</div>
  if (latest === undefined) return <div className="model-plan-loading" aria-live="polite">최신 학습 결과 분석 중…</div>
  if (latest === null) return (
    <div className="model-plan-empty">
      <b>첫 모델을 학습하면 다음 행동을 자동으로 안내합니다</b>
      <span>클래스별 성능·시험 표본 수를 보고 보강 대상을 정합니다.</span>
    </div>
  )
  return <ModelImprovementCard model={latest} />
}

function TrainPanel({ trainInfo, onTrigger, approved = 0, pid, onMsg = () => {}, onModelChange, modelRefreshKey }) {
  const { job = { status: 'idle' }, active_model } = trainInfo
  const activeQuality = active_model ? modelQuality(active_model) : null
  const running = job.status === 'running'
  const [readiness, setReadiness] = useState(null)
  const [readinessError, setReadinessError] = useState(false)
  // 라벨 0장부터 전체 학습센터를 펼치면 이미지 목록이 화면 아래로 밀린다.
  // 요약은 항상 보이되, 실제 학습 가능 시점이나 주의가 필요한 작업에서 연다.
  const [open, setOpen] = useState(false)

  useEffect(() => setOpen(false), [pid])

  useEffect(() => {
    let alive = true
    setReadiness(null)
    setReadinessError(false)
    api.trainReadiness(pid).then((r) => { if (alive) setReadiness(r) })
      .catch(() => { if (alive) setReadinessError(true) })
    return () => { alive = false }
  }, [pid, approved, job.status])

  useEffect(() => {
    if (running || job.status === 'failed' || job.status === 'completed'
      || (!active_model && readiness?.ready_manual)) {
      setOpen(true)
    }
  }, [running, job.status, active_model, readiness?.ready_manual])

  const ready = readiness || {
    approved, min_manual: 4, min_auto: MIN_APPROVED, next_auto_at: MIN_APPROVED,
    remaining_auto: Math.max(0, MIN_APPROVED - approved), ready_manual: approved >= 4,
    recommended_arch: approved < 800 ? 'yolo11n' : 'yolo11s', expected_epochs: approved < 100 ? 60 : 100,
    split_counts: { train: approved, val: 0, test: 0 }, class_count: 0,
    stage: approved >= 4 ? 'experiment' : 'collecting', professional_ready: false, warnings: [],
  }
  const currentStep = trainStepIndex(job)
  const overall = trainOverallProgress(job)
  const nowSec = Date.now() / 1000
  const liveElapsed = running && job.started_at ? nowSec - job.started_at : job.elapsed_sec
  const liveEta = job.eta_sec == null ? null
    : Math.max(0, job.eta_sec - (running && job.updated_at ? nowSec - job.updated_at : 0))
  const statusTone = job.status === 'failed' ? 'bad' : running ? 'running'
    : job.status === 'completed' ? (job.promoted ? 'ok' : 'warn')
      : active_model ? activeQuality.tone : 'idle'
  const summary = running
    ? `${trainingPhaseLabel(job.phase)}${job.epoch ? ` · ${job.epoch}/${job.epochs} epoch` : ''}`
    : job.status === 'failed' ? '학습 실패 · 확인 필요'
      : job.status === 'completed' ? (job.promoted ? '새 모델 적용 완료' : '기존 모델 유지')
        : active_model
          ? (activeQuality.status === 'verified' ? '검증된 전용 모델 가동 중'
            : activeQuality.status === 'provisional' ? '실험 모델 가동 중'
              : activeQuality.status === 'failed' ? '품질 미달 모델 사용 중' : '모델 검증 필요')
          : `${readinessLabel(ready)} · 승인 ${approved}장`

  return (
    <details className={`training-center card ${statusTone}`} open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="training-summary">
        <span className={`train-status-dot ${statusTone}`} aria-hidden="true" />
        <span className="training-summary-copy"><b>학습센터</b><small>{summary}</small></span>
        <span className="training-chevron" aria-hidden="true">{open ? '▾' : '▸'}</span>
      </summary>

      <div className="training-body">
        {running && (
          <div className="train-live" aria-live="polite">
            <div className="train-steps" aria-label="학습 진행 단계">
              {TRAIN_STEPS.map((label, i) => (
                <div key={label} className={`train-step ${i < currentStep ? 'done' : ''} ${i === currentStep ? 'current' : ''}`}>
                  <span>{i < currentStep ? '✓' : i + 1}</span><small>{label}</small>
                </div>
              ))}
            </div>
            <div className="train-live-head">
              <div><b>{trainingPhaseLabel(job.phase)}</b>
                {job.phase === 'training' && <small>{job.epoch || 0}/{job.epochs || ready.expected_epochs} epoch</small>}
              </div>
              <b>{overall}%</b>
            </div>
            <div className="train-meter" role="progressbar" aria-label="전체 학습 진행률"
              aria-valuemin="0" aria-valuemax="100" aria-valuenow={overall}>
              <span style={{ width: `${overall}%` }} />
            </div>
            <div className="train-time">
              <span>경과 <b>{formatDuration(liveElapsed)}</b></span>
              <span>예상 남음 <b>{job.phase === 'training' ? formatDuration(liveEta) : '단계 전환 중'}</b></span>
            </div>
          </div>
        )}

        {!running && job.status === 'failed' && (
          <div className="train-result failure" role="alert">
            <b>{trainingPhaseLabel(job.phase)}에서 멈췄습니다</b>
            <span>{job.error || '학습 워커가 종료되었습니다. 데이터와 로그를 확인한 뒤 다시 시도하세요.'}</span>
          </div>
        )}

        {!running && job.status === 'completed' && (
          <div className={`train-result ${job.promoted ? 'success' : 'kept'}`}>
            <b>{job.promoted ? '새 모델을 오토라벨에 적용했습니다' : '새 모델 대신 기존 모델을 유지했습니다'}</b>
            <span>{job.promoted
              ? '품질 하한과 기존 모델 비교를 통과했습니다.'
              : '새 모델이 품질 하한 또는 기존 모델 비교를 넘지 못해 안전하게 교체하지 않았습니다.'}</span>
            <div className="train-metrics">
              <span>validation mAP50 <b>{metric(job.map50)}</b></span>
              <span>holdout mAP50 <b>{metric(job.test_map50)}</b></span>
              <span>운영 F1 <b>{metric(job.operational_f1)}</b></span>
              <span>checkpoint <b>{job.checkpoint || '—'}.pt</b></span>
            </div>
          </div>
        )}

        {!running && (
          <>
            <div className="train-section-head"><b>학습 준비도</b><small>승인 라벨만 사용</small></div>
            {readinessError && <div className="warn-text">준비도를 불러오지 못했습니다. 승인 수 기준으로 표시합니다.</div>}
            <div className="train-readiness-grid">
              <div><small>승인 데이터</small><b>{ready.approved}장</b></div>
              <div><small>예상 분할</small><b>{ready.split_counts.train}/{ready.split_counts.val}/{ready.split_counts.test}</b>
                <em>train / val / test</em></div>
              <div><small>추천 설정</small><b>{ready.recommended_arch}</b><em>{ready.expected_epochs} epoch</em></div>
              <div><small>현재 단계</small><b>{readinessLabel(ready)}</b>
                <em>{ready.professional_ready ? '독립 평가 가능' : `전문 기준까지 ${ready.remaining_professional ?? '—'}장`}</em></div>
            </div>
            {(ready.warnings || []).length > 0 && (
              <div className="train-warning-list" role="status">
                {(ready.warnings || []).map((warning) => <div key={warning}>⚠ {warning}</div>)}
              </div>
            )}
            {!ready.ready_manual && (
              <div className="train-callout">수동 학습은 승인 <b>{ready.min_manual}장</b>부터 가능합니다. 안정적인 첫 학습은 <b>{ready.min_auto}장 이상</b>을 권장합니다.</div>
            )}
            {active_model && (
              <div className={`active-model-line model-${activeQuality.tone}`}>
                <span>현재 오토라벨 모델 · {activeQuality.label}</span>
                <b>mAP50 {metric(active_model.map50)}</b>
                {active_model.meta?.operational_f1 != null && <b>운영 F1 {metric(active_model.meta.operational_f1)}</b>}
                <small>승인 {active_model.train_images}장 · {active_model.meta?.checkpoint || 'best'}.pt</small>
                {active_model.meta?.quality_reason && <small className="model-quality-reason">{active_model.meta.quality_reason}</small>}
              </div>
            )}
          </>
        )}

        {!running && <LatestModelDiagnosis pid={pid}
          refreshKey={`${job.status}:${modelRefreshKey}`} />}

        <div className="train-actions">
          <button className="primary" disabled={running || !ready.ready_manual} onClick={onTrigger}
            title={ready.ready_manual ? '현재 승인 라벨로 로컬 학습을 시작합니다' : `승인 ${ready.min_manual}장부터 학습할 수 있습니다`}>
            {running ? '학습 진행 중…' : job.status === 'failed' ? '로컬 학습 다시 시도' : '로컬 학습 시작'}
          </button>
          {!running && ready.ready_manual && <small>Mac GPU(MPS) 사용 · 화면을 닫아도 계속 진행</small>}
        </div>

        <details className="cloud-train">
          <summary>Colab GPU 학습 · 결과 적용</summary>
          <ol>
            <li><a href={api.trainingDatasetUrl(pid)}>1. 승인 학습 데이터.zip</a></li>
            <li><a href={api.colabNotebookUrl(pid)}>2. Colab 노트북.ipynb</a> · T4에서 모두 실행</li>
            <li>마지막에 받은 <code>autolabel-model.zip</code>을 아래에서 검증·적용</li>
          </ol>
          <div className="hint">노트북 설정은 위 추천 epoch와 자동으로 맞습니다. test가 부족하면 실험 모델로 표시되고 자동 적용되지 않습니다.</div>
          <ModelImport pid={pid} onMsg={onMsg} onChanged={onModelChange} />
        </details>
        <ModelHistory pid={pid} refreshKey={`${job.status}:${modelRefreshKey}`}
          onMsg={onMsg} onChanged={onModelChange} />
      </div>
    </details>
  )
}

// 통계적 배치 검수 — 표본만 보고 배치 전체를 판정
function SamplingPanel({ plan, current, sampleImages, onOpen, onSubmit, onClose }) {
  const [reviews, setReviews] = useState({})
  const [busy, setBusy] = useState(false)
  useEffect(() => { setReviews({}) }, [plan.lot_token])
  const reviewed = Object.keys(reviews).length
  const defects = Object.values(reviews).filter(Boolean).length
  const complete = sampleImages.length === plan.sample_size && reviewed === plan.sample_size
  const mark = (isDefect) => {
    if (!current || !plan.sample_image_ids.includes(current.id)) return
    const nextReviews = { ...reviews, [current.id]: isDefect }
    setReviews(nextReviews)
    const next = sampleImages.find((im) => !(im.id in nextReviews))
    if (next) onOpen(next)
  }
  return (
    <div className="card" style={{ borderColor: 'var(--accent)' }}>
      <div className="panel-title">📊 배치 검수 진행 중</div>
      <div className="hint">
        대기 <b>{plan.lot_size}</b>장 중 <b>{plan.sample_size}</b>장만 검사하면 됩니다
        (검수 {(plan.saving * 100).toFixed(0)}% 절감).<br />
        왼쪽 목록에는 <b>뽑힌 표본만</b> 표시됩니다. 현재 이미지의 라벨이 맞는지 판정하세요.
        불량 <b>{plan.max_defects}개 이하</b>면 배치 전체가 승인됩니다.
      </div>
      <div className="sampling-progress" aria-live="polite">
        판정 <b>{reviewed}/{plan.sample_size}</b> · 오류 <b>{defects}</b>
      </div>
      {sampleImages.length !== plan.sample_size && (
        <div className="bad-text">표본 일부를 찾을 수 없습니다. 취소 후 새 계획을 만드세요.</div>
      )}
      <div className="row">
        <button className="ok" disabled={!current} onClick={() => mark(false)}>✓ 정상 · 다음</button>
        <button className="bad" disabled={!current} onClick={() => mark(true)}>✗ 라벨 오류 · 다음</button>
      </div>
      <div className="row">
        <button className="primary" disabled={!complete || busy} onClick={async () => {
          setBusy(true)
          try { await onSubmit(defects) } finally { setBusy(false) }
        }}>{busy ? '판정 중…' : `배치 판정 (${defects}개 오류)`}</button>
        <button onClick={onClose}>취소</button>
      </div>
    </div>
  )
}

// 학습 라운드별 성능 추이 + 롤백
function ModelHistory({ pid, refreshKey, onMsg, onChanged }) {
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
      {models.map((m) => {
        const quality = modelQuality(m)
        return <div key={m.id} className="row model-history-row">
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
          <span className={`model-quality-badge model-${quality.tone}`}>{quality.label}</span>
          {m.active
            ? <span className={quality.usable ? 'ok-text' : 'warn-text'}>● 사용 중</span>
            : <button className="model-use" disabled={!quality.usable}
                title={quality.usable ? '이 후보를 오토라벨 모델로 적용합니다' : (m.meta?.quality_reason || quality.label)}
                onClick={async () => {
                  try {
                    await api.activateModel(pid, m.id)
                    setModels(await api.listModels(pid))
                    await onChanged?.()
                    onMsg(`모델 #${m.id}로 전환 (홀드아웃 ${m.test_map50?.toFixed(3) ?? '—'})`)
                  } catch (e) { onMsg(`모델 적용 실패: ${e.message}`, true) }
                }}>{quality.usable ? '사용' : '차단'}</button>}
        </div>
      })}
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
  onDelete, onBulk, sampleMode = false }) {
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
      {sampleMode ? (
        <div className="sample-banner"><b>검수 표본 {visible.length}장</b> · 목록 잠금</div>
      ) : <div className="row filters">
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
      </div>}
      {!sampleMode && picked.size > 0 ? (
        <div className="row bulkbar">
          <b>{picked.size}장 선택</b>
          <button className="ok" onClick={() => applyBulk('approved')}>✓ 일괄 승인</button>
          <button className="bad" onClick={() => applyBulk('rejected')}>✗ 일괄 거부</button>
          <button onClick={() => setPicked(new Set())}>해제</button>
        </div>
      ) : (
        <div className="hint">
          {pos >= 0 ? <b>{pos + 1} / {visible.length}</b> : `${visible.length}장`}
          {sampleMode ? ' · ←→ 이동 · 위에서 정상/오류 판정' : ' · ←→ 이동 · Shift/⌘+클릭 다중 선택'}
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
            role="button" tabIndex={0} aria-current={current?.id === im.id ? 'true' : undefined}
            className={`${current?.id === im.id ? 'active' : ''}${picked.has(im.id) ? ' picked' : ''}`}
            onClick={(e) => onRowClick(im, i, e)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onRowClick(im, i, e)
              }
            }}>
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
            {!sampleMode && <button className="x rowdel" title="이미지 삭제"
              onClick={(e) => { e.stopPropagation(); onDelete(im) }}>×</button>}
          </li>
          )
        })}
        </ul>
      </div>
    </div>
  )
}

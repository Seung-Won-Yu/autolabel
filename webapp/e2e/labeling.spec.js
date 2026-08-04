import { expect, test } from '@playwright/test'

/**
 * 프론트 회귀 잠금.
 *
 * 여기 있는 케이스는 전부 실사용 테스트에서 손으로 찾은 결함이다 —
 * 임포트 후 빈 캔버스, 배경 클릭이 전체 프레임 라벨로 확정되던 것,
 * 한글 프로젝트명 익스포트 500. 포인터 상호작용은 API 테스트로 못 잡는다.
 */

const API = 'http://127.0.0.1:8991/api'

/** 캔버스에 실제로 그려진 이미지의 화면 좌표 (konva Stage 기준). */
async function canvasBox(page) {
  const box = await page.locator('canvas').first().boundingBox()
  expect(box, '캔버스가 렌더되어야 한다').not.toBeNull()
  return box
}

/**
 * 프로젝트를 만들고 화면에서 그 프로젝트를 여는 데 쓸 고유 이름을 돌려준다.
 * e2e DB는 실행 간에 유지되므로 이름이 겹치면 셀렉터가 여러 개를 잡는다.
 */
async function newProject(page, label, ontology) {
  const name = `${label}-${Date.now()}-${Math.floor(Math.random() * 1e4)}`
  const r = await page.request.post(`${API}/projects`, { data: { name, ontology } })
  expect(r.ok(), await r.text()).toBeTruthy()
  return { id: (await r.json()).id, name }
}

// 100x100 단색 PNG — 1x1을 늘린 게 아니라 실제 크기가 있어야 캔버스 좌표
// 변환을 검증할 수 있다
const PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAAPElEQVR4nO3BAQ0AAADCoPdPbQ8H'
  + 'FAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgNxUgAAGKp5MTAAAAAElFTkSuQmCC',
  'base64')

/** 라벨 대상 하나를 만든다 (모델 추론 없음). */
async function uploadImage(page, pid, name = 'a.png') {
  const r = await page.request.post(`${API}/projects/${pid}/images`, {
    multipart: { files: { name, mimeType: 'image/png', buffer: PNG } },
  })
  expect(r.ok()).toBeTruthy()
  return (await r.json()).saved[0]
}

const ONE_CLASS = [{ name: 'sig', prompt: 'signature', threshold: 0.3 }]

test('열려 있는 프로젝트에 이미지를 넣으면 캔버스가 자동으로 열린다', async ({ page }) => {
  // 자동 열기가 프로젝트를 열 때만 돌던 시절, 임포트·업로드 직후에는
  // "좌측에서 이미지를 선택하세요"에 멈춰 있었다. 썸네일 목록은 좌측 패널
  // 아래라 화면 밖 — 방금 오토라벨을 돌린 사용자에게 리뷰할 게 안 보였다.
  // 그래서 프로젝트를 "비어 있는 채로" 먼저 열고 그 다음에 이미지를 넣는다.
  const { name } = await newProject(page, 'auto-open', ONE_CLASS)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.getByText('이미지가 없습니다.')).toBeVisible()

  await page.locator('label.upload input[type=file]').setInputFiles({
    name: 'a.png', mimeType: 'image/png', buffer: PNG,
  })

  await expect(page.locator('canvas').first()).toBeVisible()
  await expect(page.getByText('좌측에서 이미지를 선택하세요')).toHaveCount(0)
})

test('드래그로 박스를 그리면 저장되고 어노테이션 패널에 뜬다', async ({ page }) => {
  const { id, name } = await newProject(page, 'drag-box', ONE_CLASS)
  const iid = await uploadImage(page, id)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()

  const box = await canvasBox(page)
  const cx = box.x + box.width / 2
  const cy = box.y + box.height / 2
  await page.mouse.move(cx - 40, cy - 30)
  await page.mouse.down()
  await page.mouse.move(cx + 40, cy + 30, { steps: 12 })
  await page.mouse.up()

  await expect(page.getByText('어노테이션 (1)')).toBeVisible()

  // 자동 저장까지 확인 — 화면에만 있고 DB에 없으면 의미가 없다
  await expect.poll(async () => {
    const r = await page.request.get(`${API}/images/${iid}/annotations`)
    return (await r.json()).length
  }, { timeout: 10_000 }).toBe(1)
})

test('A로 승인하면 진행률이 오르고 다음 이미지로 넘어간다', async ({ page }) => {
  const { id, name } = await newProject(page, 'approve-key', ONE_CLASS)
  await uploadImage(page, id, 'first.png')
  await uploadImage(page, id, 'second.png')

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()

  await expect(page.getByText('0/2 승인')).toBeVisible()
  await page.locator('canvas').first().click({ position: { x: 5, y: 5 } })
  await page.keyboard.press('a')
  await expect(page.getByText('1/2 승인')).toBeVisible()
})

test('클래스를 지우면 캔버스 도구 힌트가 사라지지 않는다', async ({ page }) => {
  // 클래스 0개 상태에서도 화면이 깨지지 않아야 한다
  const { id, name } = await newProject(page, 'no-class', ONE_CLASS)
  await uploadImage(page, id)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()

  // 클래스 편집은 설정·도구 안이다 (라벨링 중엔 접혀 있다)
  await page.locator('.setup-toggle').click()
  await page.getByRole('button', { name: '×' }).first().click()
  await expect(page.getByText('클래스 (0)')).toBeVisible()
  await expect(page.locator('canvas').first()).toBeVisible()
})

test('이미지 목록이 스크롤 없이 보이고 캔버스가 컨테이너에 딱 맞는다', async ({ page }) => {
  // 예전엔 설정 카드가 전부 펼쳐진 채 목록 위에 쌓여 사이드바가 10화면
  // 높이(scrollHeight 8580 / 화면 855)가 됐고, 목록은 뷰포트 밖에서 시작했다.
  // 캔버스는 window.innerWidth에서 상수를 뺀 크기라 컨테이너와 어긋났다.
  const { id, name } = await newProject(page, 'layout', ONE_CLASS)
  await uploadImage(page, id)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()

  const m = await page.evaluate(() => {
    const side = document.querySelector('.sidebar')
    const list = document.querySelector('.ilist')
    const host = document.querySelector('.canvas-host')
    const cv = document.querySelector('canvas')
    return {
      sideOverflow: side.scrollHeight - side.clientHeight,
      listTop: list.getBoundingClientRect().top,
      vh: window.innerHeight,
      // 백킹스토어(cv.width)는 레티나에서 2배가 된다 — CSS 크기로 비교해야 한다
      fits: host.clientWidth === cv.offsetWidth && host.clientHeight === cv.offsetHeight,
    }
  })
  expect(m.sideOverflow, '사이드바 자체는 스크롤되지 않아야 한다').toBeLessThanOrEqual(0)
  expect(m.listTop, '이미지 목록이 화면 안에서 시작해야 한다').toBeLessThan(m.vh)
  expect(m.fits, '캔버스가 컨테이너 크기와 일치해야 한다').toBe(true)
})

test('줌 컨트롤로 확대하고 맞춤으로 되돌린다', async ({ page }) => {
  // 휠만 있으면 한번 어긋난 뷰를 되돌릴 방법이 없었다
  const { id, name } = await newProject(page, 'zoom', ONE_CLASS)
  await uploadImage(page, id)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('.zoombar')).toBeVisible()

  const pct = () => page.locator('.zoombar span').innerText()
  const fitted = await pct()
  await page.locator('.zoombar button', { hasText: '+' }).click()
  await expect.poll(pct).not.toBe(fitted)
  await page.locator('.zoombar button', { hasText: '맞춤' }).click()
  await expect.poll(pct).toBe(fitted)
})

test('이미지를 넘기면 목록이 따라오고 위치가 표시된다', async ({ page }) => {
  // 예전엔 목록이 스크롤되지 않아, 25장쯤 넘기면 현재 이미지가 목록 밖으로
  // 밀려나 내가 어디쯤인지 알 수 없었다 (143장 리뷰에서 치명적)
  const { id, name } = await newProject(page, 'follow', ONE_CLASS)
  for (let i = 0; i < 25; i++) await uploadImage(page, id, `img${i}.png`)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()
  await expect(page.locator('.imagelist .hint')).toContainText('1 / 25')

  for (let i = 0; i < 20; i++) await page.keyboard.press('ArrowRight')
  await expect(page.locator('.imagelist .hint')).toContainText('21 / 25')

  const visible = await page.evaluate(() => {
    const ul = document.querySelector('.ilist')
    const li = ul.querySelector('li.active').getBoundingClientRect()
    const box = ul.getBoundingClientRect()
    return li.top >= box.top - 1 && li.bottom <= box.bottom + 1
  })
  expect(visible, '현재 이미지가 목록 안에 보여야 한다').toBe(true)
})

test('클래스 칩으로 캔버스에서 숨기고 다시 표시한다', async ({ page }) => {
  // 밀집 장면(박스 20개+)에서 한 종류만 보며 고칠 수 있어야 한다
  const two = [{ name: 'a', prompt: 'a', threshold: 0.3 }, { name: 'b', prompt: 'b', threshold: 0.3 }]
  const { id, name } = await newProject(page, 'chips', two)
  const iid = await uploadImage(page, id)
  await page.request.put(`${API}/images/${iid}/annotations`, {
    data: { annotations: [
      { class_name: 'a', bbox: [1, 1, 10, 10], source: 'human' },
      { class_name: 'a', bbox: [20, 1, 10, 10], source: 'human' },
      { class_name: 'b', bbox: [40, 1, 10, 10], source: 'human' },
    ] },
  })

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('.annrow')).toHaveCount(3)
  await expect(page.locator('.clschip', { hasText: 'a' }).first()).toContainText('2')

  await page.locator('.clschip', { hasText: 'a' }).first().click()
  await expect(page.locator('.annrow')).toHaveCount(1)   // b만 남는다
  await page.locator('.clschip', { hasText: 'a' }).first().click()
  await expect(page.locator('.annrow')).toHaveCount(3)
})

test('Shift/⌘ 다중 선택으로 일괄 승인한다', async ({ page }) => {
  // 리뷰를 한 장씩만 처리하면 수백 장에서 손이 남지 않는다.
  // 백엔드 bulk-status는 있었는데 프론트에서 쓰지 않고 있었다.
  const { id, name } = await newProject(page, 'bulk', ONE_CLASS)
  for (let i = 0; i < 6; i++) await uploadImage(page, id, `b${i}.png`)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.getByText('0/6 승인')).toBeVisible()

  const rows = page.locator('.ilist li')
  await rows.nth(1).click({ modifiers: ['ControlOrMeta'] })
  await rows.nth(4).click({ modifiers: ['Shift'] })
  await expect(page.locator('.ilist li.picked')).toHaveCount(4)
  await expect(page.locator('.bulkbar')).toContainText('4장 선택')

  await page.locator('.bulkbar button', { hasText: '일괄 승인' }).click()
  await expect(page.getByText('4/6 승인')).toBeVisible()
  await expect(page.locator('.ilist li.picked')).toHaveCount(0)
})

test('목록은 보이는 구간만 DOM에 올린다 (창 렌더링)', async ({ page }) => {
  // 예전엔 전부 올려서 310장이면 노드 2364개(행당 7개)였다. 1만 장이면
  // 7만 개가 되어 못 버틴다. 스크롤바 높이와 구간 계산이 맞아야 한다.
  const N = 60
  const { id, name } = await newProject(page, 'window', ONE_CLASS)
  for (let i = 0; i < N; i++) await uploadImage(page, id, `w${i}.png`)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()

  const rows = page.locator('.ilist li')
  const rendered = await rows.count()
  expect(rendered, '전부 올리면 창 렌더링이 아니다').toBeLessThan(N)
  expect(rendered, '화면을 채울 만큼은 있어야 한다').toBeGreaterThan(5)

  // 스크롤 가능 높이는 전체 장수 × 행 높이여야 한다 (스크롤바가 정직해야 한다)
  const geom = await page.evaluate(() => {
    const sc = document.querySelector('.ilist')
    return { scrollHeight: sc.scrollHeight, rowH: sc.querySelector('li').offsetHeight }
  })
  expect(geom.scrollHeight).toBe(N * geom.rowH)

  // 끝까지 스크롤하면 마지막 이미지가 렌더돼야 한다
  await page.evaluate(() => {
    const sc = document.querySelector('.ilist')
    sc.scrollTop = sc.scrollHeight
  })
  await expect(page.locator('.ilist li').last()).toContainText(`w${N - 1}.png`)
})

test('창 렌더링에서도 현재 이미지가 목록에 남는다', async ({ page }) => {
  // 화면 밖 행은 DOM에 없으므로 scrollIntoView를 쓸 수 없다 — 위치를 직접
  // 계산해 스크롤하고 상태도 같이 맞춰야, 연속 이동에서 창이 뒤처지지 않는다.
  const { id, name } = await newProject(page, 'follow-window', ONE_CLASS)
  for (let i = 0; i < 40; i++) await uploadImage(page, id, `f${i}.png`)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()
  for (let i = 0; i < 30; i++) await page.keyboard.press('ArrowRight')

  await expect(page.locator('.imagelist .hint')).toContainText('31 / 40')
  await expect(page.locator('.ilist li.active')).toHaveCount(1)
  const visible = await page.evaluate(() => {
    const sc = document.querySelector('.ilist')
    const b = sc.querySelector('li.active').getBoundingClientRect()
    const box = sc.getBoundingClientRect()
    return b.top >= box.top - 1 && b.bottom <= box.bottom + 1
  })
  expect(visible, '현재 이미지가 목록 안에 보여야 한다').toBe(true)
})

test('선택한 박스는 화살표로 1px씩 옮긴다', async ({ page }) => {
  // 마우스로는 1px을 맞출 수 없어 박스가 늘 어긋난다. 박스를 고른 동안만
  // 화살표가 미세조정이 되고, Escape로 다시 이미지 이동으로 돌아간다.
  const { id, name } = await newProject(page, 'nudge', ONE_CLASS)
  const iid = await uploadImage(page, id)
  await page.request.put(`${API}/images/${iid}/annotations`, {
    data: { annotations: [{ class_name: 'sig', bbox: [10, 20, 30, 40], source: 'human' }] },
  })

  await page.goto('/')
  await page.getByText(name).click()
  // 행 가운데는 클래스 드롭다운이다 — 번호를 눌러 박스만 선택한다
  await page.locator('.annrow .annidx').first().click()
  await expect(page.locator('.annrow.active')).toHaveCount(1)

  await page.keyboard.press('ArrowRight')
  await page.keyboard.press('ArrowDown')
  await page.keyboard.press('Shift+ArrowRight')

  await expect.poll(async () => {
    const r = await page.request.get(`${API}/images/${iid}/annotations`)
    return (await r.json())[0].bbox
  }, { timeout: 10_000 }).toEqual([21, 21, 30, 40])  // x +1 +10, y +1
})

test('중단된 배치를 완료라고 말하지 않는다', async ({ page }) => {
  // 잡 상태가 메모리에만 있어서 서버가 재시작하면 기록이 사라졌고, 프론트는
  // 그걸 완료로 읽어 "배치 오토라벨 완료: undefined/undefined장"을 띄웠다.
  // 절반만 라벨된 데이터를 두고 사용자는 끝난 줄 안다.
  const { id, name } = await newProject(page, 'interrupted', ONE_CLASS)
  await uploadImage(page, id)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()

  // 시작은 running, 이후 폴링은 interrupted (서버 재시작 후 정리된 상태).
  // 글로브에서 *는 /를 넘지 않는다 — status는 따로 잡아야 한다.
  await page.route('**/api/projects/*/autolabel', (route) =>
    route.fulfill({ json: { status: 'running', done: 0, total: 12 } }))
  let first = true
  await page.route('**/api/projects/*/autolabel/status', (route) => {
    if (first) {
      first = false
      return route.fulfill({ json: { status: 'running', done: 5, total: 12 } })
    }
    return route.fulfill({ json: {
      status: 'interrupted', done: 5, total: 12,
      error: '서버가 재시작되어 작업이 중단됐습니다 — 다시 실행하세요' } })
  })

  await page.locator('.setup-toggle').click()
  await page.locator('button', { hasText: '전체 오토라벨' }).first().click()

  const toast = page.locator('.toast')
  await expect(toast).toContainText('중단', { timeout: 10_000 })
  await expect(toast).toContainText('5/12')          // 어디까지 했는지 알려준다
  await expect(toast).not.toContainText('완료')
  await expect(toast).not.toContainText('undefined')
  await expect(toast).toHaveClass(/sticky/)          // 놓치면 안 되는 안내다
})

test('저장이 실패하면 알리고 서버가 돌아오면 스스로 복구한다', async ({ page }) => {
  // 자동 저장에 catch가 없어서, 서버가 죽으면 콘솔에 "Failed to fetch"만 남고
  // 화면은 조용했다. 라벨러는 계속 작업하다 전부 잃는다.
  const { id, name } = await newProject(page, 'savefail', ONE_CLASS)
  const iid = await uploadImage(page, id)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()

  let failing = true
  await page.route('**/api/images/*/annotations', (route) => (
    failing && route.request().method() === 'PUT' ? route.abort('failed') : route.continue()))

  const box = await canvasBox(page)
  await page.mouse.move(box.x + 40, box.y + 40)
  await page.mouse.down()
  await page.mouse.move(box.x + 120, box.y + 110, { steps: 8 })
  await page.mouse.up()

  // 저장 안 됨을 반드시 보여줘야 한다
  await expect(page.locator('.chip.danger')).toBeVisible({ timeout: 10_000 })

  failing = false                                   // 서버 복구
  await expect(page.locator('.chip.danger')).toHaveCount(0, { timeout: 15_000 })
  await expect.poll(async () => {
    const r = await page.request.get(`${API}/images/${iid}/annotations`)
    return (await r.json()).length
  }, { timeout: 15_000 }).toBe(1)                   // 재시도로 실제 저장됨
})

test('A를 연타해도 승인이 유실되지 않는다', async ({ page }) => {
  // 리뷰 핵심 루프의 조용한 데이터 유실이었다. setStatus가 비동기인데 각
  // 호출이 낡은 클로저의 current를 읽어 같은 이미지를 두 번 처리하고 나머지를
  // 건너뛰었다 (실측: a 5연타 → 2장만 승인, 400ms 간격 10회에도 7장).
  const N = 8
  const { id, name } = await newProject(page, 'rapid', ONE_CLASS)
  for (let i = 0; i < N; i++) await uploadImage(page, id, `r${i}.png`)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()

  for (let i = 0; i < N; i++) await page.keyboard.press('a')   // 대기 없이 연타

  await expect.poll(async () => {
    const r = await page.request.get(`${API}/projects/${id}/images`)
    return (await r.json()).filter((im) => im.status === 'approved').length
  }, { timeout: 15_000 }).toBe(N)
})

test('승인을 되돌리면 그 이미지로 돌아가 상태가 복구된다', async ({ page }) => {
  // 되돌릴 수 없는 유일한 파괴적 동작이었다. 이력이 이미지별이라 넘기면
  // 초기화됐고, A로 빠르게 리뷰하다 오승인하면 복구 경로가 아예 없었다.
  const { id, name } = await newProject(page, 'undo-status', ONE_CLASS)
  for (let i = 0; i < 4; i++) await uploadImage(page, id, `u${i}.png`)

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()

  const firstName = await page.locator('.ilist li.active .name').innerText()
  for (let i = 0; i < 3; i++) await page.keyboard.press('a')
  await expect(page.getByText('3/4 승인')).toBeVisible()

  await page.keyboard.press('ControlOrMeta+z')
  await expect(page.getByText('2/4 승인')).toBeVisible()
  await page.keyboard.press('ControlOrMeta+z')
  await page.keyboard.press('ControlOrMeta+z')
  await expect(page.getByText('0/4 승인')).toBeVisible()
  // 마지막 되돌리기는 첫 이미지로 돌아가야 한다
  await expect(page.locator('.ilist li.active .name')).toHaveText(firstName)

  await page.keyboard.press('ControlOrMeta+z')
  await expect(page.getByText('되돌릴 작업이 없습니다')).toBeVisible()
})

test('이미지를 넘긴 뒤에도 어노테이션을 되돌린다', async ({ page }) => {
  const { id, name } = await newProject(page, 'undo-anns', ONE_CLASS)
  const first = await uploadImage(page, id, 'a.png')
  await uploadImage(page, id, 'b.png')
  await uploadImage(page, id, 'c.png')
  await page.request.put(`${API}/images/${first}/annotations`, {
    data: { annotations: [
      { class_name: 'sig', bbox: [1, 2, 3, 4], source: 'human' },
      { class_name: 'sig', bbox: [5, 6, 7, 8], source: 'human' },
    ] },
  })

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('.annrow')).toHaveCount(2)

  await page.locator('.annrow .x').first().click()   // 하나 삭제
  await expect(page.locator('.annrow')).toHaveCount(1)
  for (let i = 0; i < 2; i++) await page.keyboard.press('ArrowRight')  // 두 장 이동

  await page.keyboard.press('ControlOrMeta+z')
  await expect(page.locator('.annrow')).toHaveCount(2)   // 원래 이미지로 돌아와 복구
  await expect.poll(async () => {
    const r = await page.request.get(`${API}/images/${first}/annotations`)
    return (await r.json()).length
  }, { timeout: 10_000 }).toBe(2)                        // DB에도 반영
})

test('거부한 이미지는 익스포트에서 빠지고 필터로 찾을 수 있다', async ({ page }) => {
  const { id, name } = await newProject(page, 'reject', ONE_CLASS)
  const keep = await uploadImage(page, id, 'keep.png')
  const drop = await uploadImage(page, id, 'drop.png')
  for (const iid of [keep, drop]) {
    await page.request.put(`${API}/images/${iid}/annotations`, {
      data: { annotations: [{ class_name: 'sig', bbox: [1, 2, 3, 4], source: 'human' }] },
    })
  }
  await page.request.post(`${API}/images/bulk-status`,
    { data: { image_ids: [drop], status: 'rejected' } })

  const coco = await (await page.request.get(`${API}/projects/${id}/export?fmt=coco`)).json()
  expect(coco.images.map((i) => i.id)).toEqual([keep])

  await page.goto('/')
  await page.getByText(name).click()
  await page.locator('.filters button', { hasText: '거부' }).click()
  await expect(page.locator('.ilist li')).toHaveCount(1)
  await expect(page.locator('.ilist li').first()).toContainText('drop.png')
})

test('목록 썸네일은 원본이 아니라 축소본을 받는다', async ({ page }) => {
  // 예전엔 원본을 그대로 받아 44x44로 줄여 그렸다 (143장 = 9.3MB)
  const { id, name } = await newProject(page, 'thumb', ONE_CLASS)
  await uploadImage(page, id)

  await page.goto('/')
  await page.getByText(name).click()
  const src = await page.locator('.ilist img').first().getAttribute('src')
  expect(src).toContain('/thumb')
  const natural = await page.locator('.ilist img').first()
    .evaluate((el) => el.naturalWidth)
  expect(natural).toBeLessThanOrEqual(128)
})

test('한글 프로젝트명으로도 익스포트가 된다', async ({ page }) => {
  // HTTP 헤더는 latin-1만 담는다. 예전엔 Content-Disposition에 이름을 그대로
  // 넣어 한글이면 500이 났다 — 한국어 사용자는 익스포트가 통째로 막혔다
  const { id } = await newProject(page, '한글 프로젝트', ONE_CLASS)
  await uploadImage(page, id)

  for (const fmt of ['yolo', 'coco']) {
    const r = await page.request.get(`${API}/projects/${id}/export.zip?fmt=${fmt}`)
    expect(r.status(), fmt).toBe(200)
    expect(r.headers()['content-disposition']).toContain("filename*=UTF-8''")
  }
})

test('추론 응답이 이미지 전환 후 도착하면 버린다', async ({ page }) => {
  // 단건 오토라벨은 수 초 걸리고 그동안 →로 넘어가는 게 리뷰의 기본 동선이다.
  // 예전엔 늦게 온 응답이 "클릭 시점 이미지의 라벨 + 검출"로 지금 보고 있는
  // 이미지의 어노테이션을 통째로 교체했고, 2초 뒤 자동저장이 그걸 서버에
  // 기록해 원래 라벨을 파괴했다.
  const { id, name } = await newProject(page, 'stale-infer', ONE_CLASS)
  await uploadImage(page, id, 'first.png')
  const second = await uploadImage(page, id, 'second.png')

  await page.route('**/api/images/*/autolabel', async (route) => {
    await new Promise((r) => setTimeout(r, 1200)) // 느린 추론 흉내
    await route.fulfill({ json: {
      detections: [{ class_name: 'sig', bbox: [10, 10, 30, 30], confidence: 0.9 }],
      engine: 'gdino',
    } })
  })

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()

  await page.getByRole('button', { name: '이 이미지 오토라벨' }).click()
  await page.keyboard.press('ArrowRight') // 응답 도착 전에 다음 이미지로

  await expect(page.getByText(/오토라벨 결과를 버렸습니다/)).toBeVisible()
  await expect(page.getByText('어노테이션 (0)')).toBeVisible()

  // 자동저장 창이 지나도 두 번째 이미지의 서버 라벨이 오염되지 않아야 한다
  await page.waitForTimeout(2500)
  const r = await page.request.get(`${API}/images/${second}/annotations`)
  expect(await r.json()).toEqual([])
})

test('저장 비행 중의 편집이 유실되지 않는다', async ({ page }) => {
  // saveAnns가 완료 시 dirty를 무조건 지우던 시절: 저장 요청이 나가 있는 동안
  // 그린 박스는 dirty가 곧 덮여 자동저장 타이머와 이탈 경고가 전부 건너뛰었다.
  // "저장됨" 토스트를 보고 탭을 닫으면 마지막 편집이 조용히 사라진다.
  const { id, name } = await newProject(page, 'inflight-edit', ONE_CLASS)
  const iid = await uploadImage(page, id)

  let slowed = false
  await page.route('**/api/images/*/annotations', async (route) => {
    if (route.request().method() === 'PUT' && !slowed) {
      slowed = true
      await new Promise((r) => setTimeout(r, 1500)) // 첫 저장을 느리게
    }
    await route.continue()
  })

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()
  const box = await canvasBox(page)
  const draw = async (ox) => {
    await page.mouse.move(box.x + ox, box.y + 40)
    await page.mouse.down()
    await page.mouse.move(box.x + ox + 30, box.y + 90, { steps: 8 })
    await page.mouse.up()
  }

  const firstPut = page.waitForRequest((r) =>
    r.url().includes(`/api/images/${iid}/annotations`) && r.method() === 'PUT')
  await draw(30)
  await firstPut
  await draw(90) // 첫 저장이 비행 중일 때의 편집 — 이게 시험 대상

  await expect.poll(async () => {
    const r = await page.request.get(`${API}/images/${iid}/annotations`)
    return (await r.json()).length
  }, { timeout: 15_000 }).toBe(2)
})

test('중단된 임포트를 완료라고 말하지 않는다', async ({ page }) => {
  // 서버가 재시작하면 sweep이 임포트 기록을 interrupted로 남긴다. 예전 폴링은
  // failed만 실패로 보고 나머지를 전부 "임포트 완료: N장 연결됨"으로 알렸다 —
  // 5만 장 중 400장만 들어온 데이터셋이 완전 임포트로 둔갑한다.
  const { name } = await newProject(page, 'import-interrupt', ONE_CLASS)

  await page.route('**/api/projects/*/import', (route) =>
    route.fulfill({ json: { status: 'running', done: 0, total: 0 } }))
  await page.route('**/api/projects/*/import/status', (route) =>
    route.fulfill({ json: { status: 'interrupted', done: 400,
      error: '서버가 재시작되어 작업이 중단됐습니다 — 다시 실행하세요' } }))

  await page.goto('/')
  await page.getByText(name).click()
  // 빈 프로젝트는 설정·도구가 이미 펼쳐져 있다 — 연결 섹션만 연다
  await page.getByText('기존 데이터셋 연결 (복사 없음)').click()
  await page.getByPlaceholder('이미지 폴더 경로 (필수)').fill('/tmp/somewhere')
  await page.getByRole('button', { name: '연결 임포트' }).click()

  // 공통 종료 메시지 규칙(jobEndMessage) — 처리량을 밝히고 완료라 말하지 않는다
  await expect(page.getByText(/임포트 중단됨 \(400\//)).toBeVisible()
  await expect(page.getByText(/임포트 완료/)).toHaveCount(0)
})

test('문맥 심판 verdict가 어노테이션 패널에 표시된다', async ({ page }) => {
  // VLM 심판이 남긴 meta.vlm이 부합✓/위반✗ 칩으로 보여야 한다 — 리뷰어는
  // 위반·불확실만 골라 확인하는 흐름이라 칩이 없으면 기능이 없는 것과 같다.
  const { id, name } = await newProject(page, 'vlm-chip', ONE_CLASS)
  const iid = await uploadImage(page, id)
  await page.request.put(`${API}/images/${iid}/annotations`, { data: { annotations: [
    { class_name: 'sig', bbox: [10, 10, 30, 30], source: 'model',
      meta: { vlm: { verdict: 'fail', reason: '기준 위반: 테스트', rubric_sha: 'abc' } } },
    { class_name: 'sig', bbox: [50, 50, 30, 30], source: 'model',
      meta: { vlm: { verdict: 'pass', reason: '부합', rubric_sha: 'abc' } } },
  ] } })

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()
  await expect(page.locator('.vchip.fail')).toHaveCount(1)
  await expect(page.locator('.vchip.pass')).toHaveCount(1)
})

test('문맥 심판은 기준 저장 전엔 잠기고 완료 요약을 알린다', async ({ page }) => {
  const { id, name } = await newProject(page, 'vlm-flow', ONE_CLASS)
  await uploadImage(page, id)
  await page.route('**/api/projects/*/vlm-judge', (route) =>
    route.fulfill({ json: { status: 'running', done: 0, total: 1 } }))
  await page.route('**/api/projects/*/vlm-judge/status', (route) =>
    route.fulfill({ json: { status: 'completed', done: 1, total: 1,
      advice: '판정 3건: 부합 1 · 위반 1 · 불확실 1 — 위반·불확실만 확인하면 됩니다' } }))

  await page.goto('/')
  await page.getByText(name).click()
  await page.locator('.setup-toggle').click()

  const runBtn = page.getByRole('button', { name: '문맥 심판 실행' })
  await expect(runBtn).toBeDisabled() // 기준 없는 판정은 의미가 없다
  await page.getByPlaceholder(/판정 기준을 서술/).fill('파손 흔적이 있는 차량만')
  await page.getByRole('button', { name: '기준 저장' }).click()
  await expect(page.getByText('판정 기준 저장됨')).toBeVisible()

  await runBtn.click()
  await expect(page.getByText(/부합 1 · 위반 1/)).toBeVisible()
})

test('마지막 이미지를 승인하면 앞쪽 리뷰 대기로 순환한다', async ({ page }) => {
  // 실측: 15장 중 마지막 장에서 A를 눌러 승인한 뒤엔 A가 죽은 키가 됐다 —
  // 앞쪽에 리뷰할 13장이 남아 있는데도. 끝에서는 앞쪽 미처리 이미지로 돌아
  // 가야 A 연타 흐름이 안 끊긴다.
  const { id, name } = await newProject(page, 'wrap-approve', ONE_CLASS)
  await uploadImage(page, id, 'first.png')
  await uploadImage(page, id, 'second.png')
  await uploadImage(page, id, 'third.png')

  await page.goto('/')
  await page.getByText(name).click()
  await expect(page.locator('canvas').first()).toBeVisible()
  await page.locator('canvas').first().click({ position: { x: 5, y: 5 } })

  // 마지막 장으로 이동한 뒤 A 3연타 — 순환이 없으면 마지막 한 장만 승인된다
  await page.keyboard.press('ArrowRight')
  await page.keyboard.press('ArrowRight')
  await page.keyboard.press('a')
  await expect(page.getByText('1/3 승인')).toBeVisible()
  await page.keyboard.press('a')
  await expect(page.getByText('2/3 승인')).toBeVisible()
  await page.keyboard.press('a')
  await expect(page.getByText('3/3 승인')).toBeVisible()
})

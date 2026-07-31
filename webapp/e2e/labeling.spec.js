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
      fits: host.clientWidth === cv.width && host.clientHeight === cv.height,
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

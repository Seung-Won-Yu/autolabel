import { readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { chromium } from '../../webapp/node_modules/playwright/index.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '../..')
const APP = process.env.APP_URL || 'http://localhost:5274'
const API = process.env.API_URL || 'http://127.0.0.1:8992/api'
const PREFIX = process.env.SCREENSHOT_PREFIX || 'baseline'

const requestJson = async (path, options = {}) => {
  const response = await fetch(`${API}${path}`, options)
  if (!response.ok) throw new Error(`${response.status} ${path}: ${await response.text()}`)
  return response.json()
}

const project = await requestJson('/projects', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: `ux-pipeline-${Date.now()}`,
    ontology: [{ name: 'signature', prompt: 'handwritten signature', threshold: 0.3 }],
  }),
})
const form = new FormData()
for (const name of ['Frame_188.jpg', 'Frame_176.jpg', 'Frame_162.jpg']) {
  const data = await readFile(resolve(ROOT, 'data/signature/images/train', name))
  form.append('files', new Blob([data], { type: 'image/jpeg' }), name)
}
const uploaded = await requestJson(`/projects/${project.id}/images`, { method: 'POST', body: form })
const iid = uploaded.saved[0]
await requestJson(`/images/${iid}/annotations`, {
  method: 'PUT', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ annotations: [
    { class_name: 'signature', bbox: [100, 100, 180, 80], confidence: 0.62, source: 'model' },
  ] }),
})
await requestJson(`/images/${iid}/status`, {
  method: 'PUT', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ status: 'prelabeled' }),
})

const browser = await chromium.launch({ headless: true })
const errors = []
const metrics = []
for (const [label, width, height] of [
  ['desktop', 1440, 900], ['narrow', 768, 800], ['mobile', 390, 844],
]) {
  const page = await browser.newPage({ viewport: { width, height } })
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(`${label}: ${message.text()}`)
  })
  page.on('pageerror', (error) => errors.push(`${label}: ${error.message}`))
  await page.goto(APP)
  await page.waitForLoadState('networkidle')
  await page.getByText(project.name).click()
  await page.locator('canvas').first().waitFor()
  await page.screenshot({ path: resolve(HERE, `${PREFIX}-${label}.png`), fullPage: true })
  metrics.push(await page.evaluate((name) => {
    const canvas = document.querySelector('canvas')?.getBoundingClientRect()
    const toolbar = document.querySelector('.toolbar')?.getBoundingClientRect()
    return {
      viewport: name,
      documentWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      canvasWidth: Math.round(canvas?.width || 0),
      toolbarHeight: Math.round(toolbar?.height || 0),
      lang: document.documentElement.lang,
      title: document.title,
    }
  }, label))
  await page.close()
}
await browser.close()

console.log(JSON.stringify({ metrics, errors }, null, 2))

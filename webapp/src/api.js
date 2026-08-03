// 백엔드 API 클라이언트 — vite proxy로 /api → :8899
const j = (r) => {
  if (!r.ok) throw new Error(`API ${r.status}`)
  return r.json()
}

export const api = {
  listProjects: () => fetch('/api/projects').then(j),
  createProject: (name, ontology) =>
    fetch('/api/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, ontology }),
    }).then(j),
  getProject: (id) => fetch(`/api/projects/${id}`).then(j),
  saveOntology: (id, ontology) =>
    fetch(`/api/projects/${id}/ontology`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ontology }),
    }).then(j),
  uploadImages: (id, files) => {
    const fd = new FormData()
    for (const f of files) fd.append('files', f)
    return fetch(`/api/projects/${id}/images`, { method: 'POST', body: fd }).then(j)
  },
  listImages: (id) => fetch(`/api/projects/${id}/images`).then(j),
  imageUrl: (iid) => `/api/images/${iid}/file`,
  // 목록 썸네일은 원본을 받아 줄이면 안 된다 (143장 = 9.3MB)
  thumbUrl: (iid, size = 96) => `/api/images/${iid}/thumb?size=${size}`,
  getAnnotations: (iid) => fetch(`/api/images/${iid}/annotations`).then(j),
  saveAnnotations: (iid, annotations) =>
    fetch(`/api/images/${iid}/annotations`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ annotations }),
    }).then(j),
  setImageStatus: (iid, status) =>
    fetch(`/api/images/${iid}/status`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    }).then(j),
  embed: (iid) => fetch(`/api/images/${iid}/embed`).then(j),
  autolabelOne: (iid, ontology) =>
    fetch(`/api/images/${iid}/autolabel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ontology, masks: false }),
    }).then(j),
  autolabelBatch: (pid, opts = {}) =>
    fetch(`/api/projects/${pid}/autolabel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }).then(j),
  autolabelStatus: (pid) => fetch(`/api/projects/${pid}/autolabel/status`).then(j),
  saveRubric: (pid, rubric) =>
    fetch(`/api/projects/${pid}/rubric`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rubric }),
    }).then(j),
  vlmJudge: (pid, body = {}) =>
    fetch(`/api/projects/${pid}/vlm-judge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j),
  vlmStatus: (pid) => fetch(`/api/projects/${pid}/vlm-judge/status`).then(j),
  capabilities: () => fetch('/api/capabilities').then(j),
  exportUrl: (pid, fmt) => `/api/projects/${pid}/export.zip?fmt=${fmt}`,
  modelUrl: (pid) => `/api/projects/${pid}/model`,
  deleteImage: (iid) => fetch(`/api/images/${iid}`, { method: 'DELETE' }).then(j),
  deleteProject: (pid) => fetch(`/api/projects/${pid}`, { method: 'DELETE' }).then(j),
  exemplar: (iid, bbox, className) =>
    fetch(`/api/images/${iid}/exemplar`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bbox, class_name: className }),
    }).then(j),
  importPreview: (body) =>
    fetch('/api/import/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j),
  importDataset: (pid, body) =>
    fetch(`/api/projects/${pid}/import`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j),
  importStatus: (pid) => fetch(`/api/projects/${pid}/import/status`).then(j),
  importModel: (pid, body) =>
    fetch(`/api/projects/${pid}/models/import`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j),
  runQa: (pid, background = false) =>
    fetch(`/api/projects/${pid}/qa?background=${background}`, { method: 'POST' }).then(j),
  qaStatus: (pid) => fetch(`/api/projects/${pid}/qa/status`).then(j),
  acceptancePlan: (pid, body = {}) =>
    fetch(`/api/projects/${pid}/acceptance-plan`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j),
  acceptanceResult: (pid, body) =>
    fetch(`/api/projects/${pid}/acceptance-result`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j),
  bulkStatus: (imageIds, status) =>
    fetch('/api/images/bulk-status', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_ids: imageIds, status }),
    }).then(j),
  promptLab: (pid, body) =>
    fetch(`/api/projects/${pid}/prompt-lab`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j),
  autoApprove: (pid, body) =>
    fetch(`/api/projects/${pid}/auto-approve`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j),
  suggestions: (iid, minConf = 0.4) =>
    fetch(`/api/images/${iid}/suggestions?min_conf=${minConf}`).then(j),
  applySuggestions: (iid, boxes) =>
    fetch(`/api/images/${iid}/apply-suggestions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ boxes }),
    }).then(j),
  nextToLabel: (pid, n = 20) =>
    fetch(`/api/projects/${pid}/next-to-label?n=${n}`).then(j),
  listModels: (pid) => fetch(`/api/projects/${pid}/models`).then(j),
  activateModel: (pid, mid) =>
    fetch(`/api/projects/${pid}/models/${mid}/activate`, { method: 'POST' }).then(j),
  triggerTrain: (pid) => fetch(`/api/projects/${pid}/train`, { method: 'POST' }).then(j),
  trainStatus: (pid) => fetch(`/api/projects/${pid}/train/status`).then(j),
}

export const PALETTE = [
  '#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6', '#1abc9c',
  '#e67e22', '#34495e', '#fd79a8', '#00b894',
]
export const classColor = (ontology, name) => {
  const i = ontology.findIndex((c) => c.name === name)
  return PALETTE[(i < 0 ? 9 : i) % PALETTE.length]
}

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
  runQa: (pid) => fetch(`/api/projects/${pid}/qa`, { method: 'POST' }).then(j),
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

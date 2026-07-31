"""종단 QA — 실사용 시나리오를 처음부터 끝까지 돌리며 각 단계를 검증한다.

시나리오: 새 프로젝트 → 데이터 연결 → 제로샷 → 시드 승인 → 자동 학습 →
전용 모델 오토라벨 → 심판 → 누락 제안 → 자동 승인 → 통계 검수 → 익스포트
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import requests

API = "http://127.0.0.1:8899/api"
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "signature"

ok = fail = 0
notes = []


def check(label: str, cond: bool, detail: str = ""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✅ {label}" + (f" — {detail}" if detail else ""))
    else:
        fail += 1
        print(f"  ❌ {label}" + (f" — {detail}" if detail else ""))
        notes.append(label)


def wait(url_fn, key="status", target=("completed", "failed", "idle"), timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = url_fn()
        if s.get(key) in target:
            return s
        time.sleep(5)
    return {"status": "timeout"}


print("=" * 60)
print("종단 QA 시작")
print("=" * 60)

# 1. 프로젝트 생성 + 온톨로지
print("\n[1] 프로젝트 · 온톨로지")
pid = requests.post(f"{API}/projects", json={
    "name": f"qa-{int(time.time())}",
    "ontology": [{"name": "signature", "prompt": "handwritten signature", "threshold": 0.3}],
}).json()["id"]
proj = requests.get(f"{API}/projects/{pid}").json()
check("프로젝트 생성", proj["id"] == pid, f"id={pid}")
check("온톨로지 저장", proj["ontology"][0]["name"] == "signature")

# 2. 연결 임포트 (복사 없이)
print("\n[2] 데이터셋 연결 임포트")
prev = requests.post(f"{API}/import/preview", json={
    "images_dir": str(DATA / "images/train"), "labels_dir": str(DATA / "labels/train")}).json()
check("임포트 미리보기", prev.get("format") == "yolo", f"{prev.get('images')}장 {prev.get('format')}")
requests.post(f"{API}/projects/{pid}/import", json={
    "images_dir": str(DATA / "images/train"), "labels_dir": str(DATA / "labels/train"),
    "class_names": ["signature"], "limit": 120})
st = wait(lambda: requests.get(f"{API}/projects/{pid}/import/status").json())
imgs = requests.get(f"{API}/projects/{pid}/images").json()
check("연결 임포트", st["status"] == "completed" and len(imgs) == 120, f"{len(imgs)}장")
check("라벨 함께 임포트", sum(i["ann_count"] for i in imgs) > 0,
      f"박스 {sum(i['ann_count'] for i in imgs)}개")
up_dir = ROOT / "data" / "uploads" / str(pid)
check("디스크 복사 없음", not up_dir.exists() or not any(up_dir.iterdir()))

# 3. 제로샷 베이스라인
print("\n[3] 제로샷 (전용 모델 없는 상태)")
z = requests.post(f"{API}/images/{imgs[0]['id']}/autolabel", json={"masks": False}).json()
check("제로샷 동작", "detections" in z, f"엔진={z.get('engine')} 검출={len(z.get('detections', []))}")
check("파운데이션 경로 사용", "foundation" in str(z.get("engine")))

# 4. 시드 승인 → 자동 학습
print("\n[4] 시드 승인 → 자동 파인튜닝")
for im in imgs[:100]:
    requests.put(f"{API}/images/{im['id']}/status", json={"status": "approved"})
print("  (승인 100장, 디바운스 대기)")
time.sleep(25)
job = wait(lambda: requests.get(f"{API}/projects/{pid}/train/status").json()["job"], timeout=2400)
check("자동 학습 완료", job.get("status") == "completed", f"{job.get('images')}장 {job.get('arch')}")
check("홀드아웃 평가 기록", job.get("test_map50") is not None,
      f"val {job.get('map50')} · holdout {job.get('test_map50')}")
check("게이트 승격", job.get("promoted") is True)

active = requests.get(f"{API}/projects/{pid}/train/status").json().get("active_model")
check("전용 모델 활성", active is not None,
      f"holdout {active.get('test_map50')}" if active else "없음")

# 5. 전용 모델로 오토라벨
print("\n[5] 전용 모델 오토라벨")
rest = [i for i in imgs[100:]]
r = requests.post(f"{API}/images/{rest[0]['id']}/autolabel", json={"masks": False}).json()
check("학생 엔진 사용", "student" in str(r.get("engine")), r.get("engine"))
check("검출 성공", len(r.get("detections", [])) > 0, f"{len(r.get('detections', []))}개")

# 6. 심판 (QA)
print("\n[6] 라벨 심판")
qa = requests.post(f"{API}/projects/{pid}/qa").json()
check("심판 실행", "estimated_label_error_rate" in qa,
      f"오류율 {qa.get('estimated_label_error_rate', 0) * 100:.1f}% · "
      f"검사 {qa.get('labels_checked')}개")
check("의심 랭킹 생성", len(qa.get("top_suspects", [])) > 0)
check("임계값 추천", "recommended_thresholds" in qa)

# 7. 누락 제안 → 반영
print("\n[7] 누락 제안 → 원클릭 반영")
target = qa["top_suspects"][0]["image_id"] if qa.get("top_suspects") else rest[0]["id"]
sug = requests.get(f"{API}/images/{target}/suggestions").json()
check("제안 조회", "missing_labels" in sug, f"{len(sug.get('missing_labels', []))}건")
if sug.get("missing_labels"):
    before = len(requests.get(f"{API}/images/{target}/annotations").json())
    requests.post(f"{API}/images/{target}/apply-suggestions",
                  json={"boxes": sug["missing_labels"]})
    after = len(requests.get(f"{API}/images/{target}/annotations").json())
    check("제안 반영", after > before, f"{before}→{after}개")

# 8. 자동 승인 + 통계 검수
print("\n[8] 자동 승인 · 통계 검수")
dry = requests.post(f"{API}/projects/{pid}/auto-approve",
                    json={"min_conf": 0.7, "dry_run": True}).json()
check("자동 승인 미리보기", "approved" in dry,
      f"대기 {dry.get('pending')} 중 {dry.get('approved')}장 대상")
plan = requests.post(f"{API}/projects/{pid}/acceptance-plan", json={}).json()
check("통계 검수 계획", plan.get("sample_size", 0) > 0,
      f"{plan.get('lot_size')}장 중 {plan.get('sample_size')}장 검사")

# 9. 능동 선별
print("\n[9] 능동 샘플 선별")
nxt = requests.get(f"{API}/projects/{pid}/next-to-label?n=5").json()
check("다음 라벨 추천", "recommended" in nxt, f"후보 {nxt.get('total_candidates')}개")

# 10. 익스포트 + 모델 다운로드
print("\n[10] 익스포트 · 모델")
coco = requests.get(f"{API}/projects/{pid}/export?fmt=coco").json()
check("COCO 익스포트", len(coco.get("annotations", [])) > 0, f"{len(coco['annotations'])}개")
z = requests.get(f"{API}/projects/{pid}/export.zip?fmt=yolo")
names = zipfile.ZipFile(io.BytesIO(z.content)).namelist() if z.content[:2] == b"PK" else []
n_img = sum(n.startswith("images/") for n in names)
n_lbl = sum(n.startswith("labels/") for n in names)
# 파일 시그니처만 보면 data.yaml 한 장짜리 빈 zip을 통과시킨다 (실제 발생)
check("YOLO zip 내용", n_img == len(imgs) and n_lbl == len(imgs) and "data.yaml" in names,
      f"{len(z.content) // 1024}KB · 이미지 {n_img} · 라벨 {n_lbl}")
check("익스포트 누락 0", z.headers.get("X-Images-Missing") == "0",
      f"누락 {z.headers.get('X-Images-Missing')}")
mdl = requests.get(f"{API}/projects/{pid}/model")
check("모델 .pt 다운로드", mdl.status_code == 200, f"{len(mdl.content) // 1024}KB")
nb = requests.get(f"{API}/projects/{pid}/colab-notebook")
check("Colab 노트북", nb.status_code == 200)

print("\n" + "=" * 60)
print(f"결과: {ok}개 통과 / {fail}개 실패")
if notes:
    print("실패 항목:", ", ".join(notes))
print("=" * 60)
sys.exit(1 if fail else 0)

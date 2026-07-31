"""part 캐스케이드 종단 QA — 계층 라벨이 검출부터 익스포트까지 살아남는지 검증.

일반 오토라벨과 다른 점만 본다: 부모 먼저 찾고 그 crop 안에서 부위를 찾는지,
저장 시 parent_annotation_id로 묶이는지, 익스포트에 'person.head' 표기가
그대로 나가는지. 부위는 부모 박스 안에 있어야 한다.
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import requests

API = "http://127.0.0.1:8899/api"
ROOT = Path(__file__).parent.parent
SAMPLES = ROOT / "data" / "samples"
sys.path.insert(0, str(ROOT))

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


def contains(parent, child, slack=0.15):
    """부위 박스가 부모 박스 안에 (여유를 두고) 들어있는가."""
    px, py, pw, ph = parent
    cx, cy, cw, ch = child
    mx, my = pw * slack, ph * slack
    return (cx >= px - mx and cy >= py - my
            and cx + cw <= px + pw + mx and cy + ch <= py + ph + my)


print("=" * 60)
print("part 캐스케이드 종단 QA")
print("=" * 60)

# 1. 계층 온톨로지
print("\n[1] 계층 온톨로지")
onto = [
    {"name": "person", "prompt": "person", "threshold": 0.35},
    {"name": "person.head", "threshold": 0.3},
    {"name": "person.left_arm", "threshold": 0.3},
]
pid = requests.post(f"{API}/projects", json={
    "name": f"parts-{int(time.time())}", "ontology": onto}).json()["id"]
proj = requests.get(f"{API}/projects/{pid}").json()
check("계층 클래스 저장", any("." in c["name"] for c in proj["ontology"]),
      " · ".join(c["name"] for c in proj["ontology"]))

from server.parts import parse_ontology  # noqa: E402

parents, parts_by = parse_ontology(onto)
kids = {c["child"] for c in parts_by.get("person", [])}
check("온톨로지 파싱", [p["name"] for p in parents] == ["person"]
      and kids >= {"head", "left_arm"},
      f"부모 {[p['name'] for p in parents]} · 부위 {sorted(kids)}")

# 2. 사람이 있는 이미지 연결
print("\n[2] 샘플 연결")
requests.post(f"{API}/projects/{pid}/import", json={
    "images_dir": str(SAMPLES), "limit": 12})
for _ in range(120):
    if requests.get(f"{API}/projects/{pid}/import/status").json()["status"] != "running":
        break
    time.sleep(0.5)
imgs = requests.get(f"{API}/projects/{pid}/images").json()
check("이미지 연결", len(imgs) > 0, f"{len(imgs)}장")

# 3. 캐스케이드 검출 — 부모가 있는 이미지를 찾을 때까지
print("\n[3] 부모 → crop → 부위 캐스케이드")
hit = None
for im in imgs:
    d = requests.post(f"{API}/images/{im['id']}/autolabel", json={"masks": False}).json()
    dets = d.get("detections", [])
    if any("." in x["class_name"] for x in dets):
        hit = (im, dets)
        break
check("부위 검출 성공", hit is not None,
      f"{hit[0]['file_name']}" if hit else "12장에서 부위를 못 찾음")

if hit:
    im, dets = hit
    par = [d for d in dets if "." not in d["class_name"]]
    prt = [d for d in dets if "." in d["class_name"]]
    check("부모·부위 함께 반환", len(par) > 0 and len(prt) > 0,
          f"부모 {len(par)} · 부위 {len(prt)}")
    check("부위에 부모 인덱스 부여", all("_parent_index" in d for d in prt),
          f"{[d['class_name'] for d in prt][:4]}")
    inside = [d for d in prt
              if d.get("_parent_index") is not None
              and d["_parent_index"] < len(par)
              and contains(par[d["_parent_index"]]["bbox"], d["bbox"])]
    check("부위가 부모 박스 안", len(inside) == len(prt),
          f"{len(inside)}/{len(prt)}건 포함")

    # 4. 저장 시 부모-자식 연결
    print("\n[4] 저장 — parent_annotation_id 연결")
    requests.post(f"{API}/projects/{pid}/autolabel",
                  json={"image_ids": [im["id"]], "masks": False})
    for _ in range(240):
        if requests.get(f"{API}/projects/{pid}/autolabel/status").json()["status"] != "running":
            break
        time.sleep(0.5)
    anns = requests.get(f"{API}/images/{im['id']}/annotations").json()
    saved_parts = [a for a in anns if "." in a["class_name"]]
    check("부위 저장됨", len(saved_parts) > 0, f"{len(anns)}개 중 부위 {len(saved_parts)}개")
    linked = [a for a in saved_parts if a.get("parent_annotation_id")]
    check("부모와 연결 저장", len(linked) == len(saved_parts),
          f"{len(linked)}/{len(saved_parts)}건 연결")
    ids = {a["id"] for a in anns}
    check("연결이 실재 부모를 가리킴",
          all(a["parent_annotation_id"] in ids for a in linked))
    parent_of = {a["id"]: a for a in anns}
    check("저장 후에도 부위가 부모 안",
          all(contains(parent_of[a["parent_annotation_id"]]["bbox"], a["bbox"])
              for a in linked))

    # 5. 익스포트에 계층 표기 유지
    print("\n[5] 익스포트")
    coco = requests.get(f"{API}/projects/{pid}/export?fmt=coco").json()
    cats = [c["name"] for c in coco["categories"]]
    check("COCO 카테고리에 계층 표기", any("." in c for c in cats), " · ".join(cats))
    z = requests.get(f"{API}/projects/{pid}/export.zip?fmt=yolo")
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    yaml = zipfile.ZipFile(io.BytesIO(z.content)).read("data.yaml").decode()
    check("YOLO data.yaml에 계층 클래스", "person.head" in yaml, yaml.strip().splitlines()[-1][:70])
    check("익스포트 누락 0", z.headers.get("X-Images-Missing") == "0")
    check("zip에 이미지·라벨", any(n.startswith("images/") for n in names)
          and any(n.startswith("labels/") for n in names))

print("\n" + "=" * 60)
print(f"결과: {ok}개 통과 / {fail}개 실패")
if notes:
    print("실패 항목:", ", ".join(notes))
print("=" * 60)
sys.exit(1 if fail else 0)

"""도그푸딩: 제로샷 불가 도메인(PCB 결함)에서 콜드스타트 → 전용 모델 루프 검증.

시나리오:
  1. pcb-defect 프로젝트 생성 (결함 6종 온톨로지)
  2. 이미지 40장 업로드 (30장 = 시드용, 10장 = 평가용 미라벨)
  3. 제로샷 배치 오토라벨 (10장) → 실패 실측 (베이스라인)
  4. 30장에 GT를 사람 라벨로 주입 + 승인 (수동 라벨링 시뮬레이션)
  → 이후 학습 트리거는 별도 (main 흐름에서)
"""
import json
import sys
from pathlib import Path

import requests

API = "http://127.0.0.1:8899/api"
ROOT = Path(__file__).parent.parent
PCB = ROOT / "data" / "deeppcb" / "PCBData"

CLASSES = {1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pinhole"}
ONTOLOGY = [
    {"name": "open", "prompt": "open circuit defect on pcb trace", "threshold": 0.3},
    {"name": "short", "prompt": "short circuit defect between pcb traces", "threshold": 0.3},
    {"name": "mousebite", "prompt": "mouse bite defect on pcb trace edge", "threshold": 0.3},
    {"name": "spur", "prompt": "spur defect protruding from pcb trace", "threshold": 0.3},
    {"name": "copper", "prompt": "spurious copper defect on pcb", "threshold": 0.3},
    {"name": "pinhole", "prompt": "pin hole defect on pcb pad", "threshold": 0.3},
]

N_SEED, N_EVAL = 30, 10


def collect_samples():
    samples = []
    for group in sorted(PCB.iterdir()):
        if not group.is_dir():
            continue
        gname = group.name.replace("group", "")
        img_dir = group / gname
        ann_dir = group / f"{gname}_not"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*_test.jpg")):
            ann = ann_dir / (img.stem.replace("_test", "") + ".txt")
            if ann.exists():
                samples.append((img, ann))
        if len(samples) >= N_SEED + N_EVAL:
            break
    return samples[: N_SEED + N_EVAL]


def parse_gt(ann_path: Path):
    out = []
    for line in ann_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        x1, y1, x2, y2, t = map(int, p)
        out.append({"class_name": CLASSES[t], "bbox": [x1, y1, x2 - x1, y2 - y1]})
    return out


def main():
    samples = collect_samples()
    print(f"샘플 {len(samples)}장 확보")

    r = requests.post(f"{API}/projects", json={"name": "pcb-defect", "ontology": ONTOLOGY})
    pid = r.json()["id"]
    print(f"프로젝트 id={pid}")

    ids = []
    for img, _ in samples:
        with open(img, "rb") as f:
            r = requests.post(f"{API}/projects/{pid}/images",
                              files=[("files", (img.name, f, "image/jpeg"))])
        ids.extend(r.json()["saved"])
    print(f"업로드 {len(ids)}장")

    seed = list(zip(ids[:N_SEED], samples[:N_SEED]))
    eval_ids = ids[N_SEED:]

    # 제로샷 베이스라인 (평가 10장, 파운데이션 엔진 — 학생 모델 없어서 auto=foundation)
    total_dets = 0
    for iid in eval_ids:
        r = requests.post(f"{API}/images/{iid}/autolabel",
                          json={"masks": False}).json()
        total_dets += len(r["detections"])
    print(f"[베이스라인] 제로샷 GDINO — 평가 10장 검출 총 {total_dets}개 (엔진 {r['engine']})")

    # GT를 사람 라벨로 주입 + 승인 (마지막 1장은 승인 트리거를 main 흐름에서)
    gt_total = 0
    for iid, (_, ann_path) in seed:
        anns = [{**a, "source": "human"} for a in parse_gt(ann_path)]
        gt_total += len(anns)
        requests.put(f"{API}/images/{iid}/annotations", json={"annotations": anns})
        requests.put(f"{API}/images/{iid}/status", json={"status": "approved"})
    print(f"[시드] {N_SEED}장 승인, GT 박스 {gt_total}개 주입 (수동 라벨 시뮬)")
    print("eval_ids:", json.dumps(eval_ids))


if __name__ == "__main__":
    sys.exit(main())

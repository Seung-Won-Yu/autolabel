"""도그푸딩 라운드 2: 학생 프리라벨 → 리뷰 시뮬 → 승인 60장 재학습용 데이터 준비.

- 새 이미지 30장 업로드 (samples[40:70])
- 학생 모델 프리라벨 실행 (실사용 흐름 재현 — 검출 수 기록)
- 리뷰 시뮬레이션: 사람이 프리라벨을 GT로 수정·승인했다고 가정 (GT 주입 + approved)
- 평가 10장(iid 35~44)은 그대로 미접촉 — 라운드 간 동일 기준
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from dogfood_pcb import collect_samples as _collect_seed  # noqa: E402
from dogfood_pcb import CLASSES, PCB, parse_gt  # noqa: E402

API = "http://127.0.0.1:8899/api"
PID = 2
N_NEW = 30


def collect_many(n_total):
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
        if len(samples) >= n_total:
            break
    return samples[:n_total]


def main():
    samples = collect_many(70)[40:70]  # 기존 40장 뒤 새 30장
    print(f"신규 {len(samples)}장")

    ids = []
    for img, _ in samples:
        with open(img, "rb") as f:
            r = requests.post(f"{API}/projects/{PID}/images",
                              files=[("files", (img.name, f, "image/jpeg"))])
        ids.extend(r.json()["saved"])
    print(f"업로드 {len(ids)}장 (id {ids[0]}~{ids[-1]})")

    # 학생 모델 프리라벨 (실사용 재현: 리뷰어가 보게 될 초안)
    pre_total = 0
    for iid in ids:
        r = requests.post(f"{API}/images/{iid}/autolabel", json={"masks": False}).json()
        pre_total += len(r["detections"])
    gt_total_new = sum(len(parse_gt(a)) for _, a in samples)
    print(f"[프리라벨] 학생 모델 초안: {pre_total}개 / GT {gt_total_new}개 "
          f"(리뷰어 작업량 = 차이 보정만)")

    # 리뷰 시뮬: 수정·승인 (GT 주입)
    for iid, (_, ann_path) in zip(ids, samples):
        anns = [{**a, "source": "human"} for a in parse_gt(ann_path)]
        requests.put(f"{API}/images/{iid}/annotations", json={"annotations": anns})
        requests.put(f"{API}/images/{iid}/status", json={"status": "approved"})
    print(f"[리뷰 시뮬] {len(ids)}장 승인 — 총 승인 60장. 재학습은 자동/수동 트리거로.")


if __name__ == "__main__":
    main()

# release — 바로 쓸 수 있는 가중치

## sam_decoder_vit_l.onnx (16MB)

SAM ViT-L 마스크 디코더. 브라우저(onnxruntime-web)에서 실행되는 부분.
`webapp/public/sam_decoder.onnx` 로 복사하면 SAM 클릭 도구가 동작한다.

```bash
cp release/sam_decoder_vit_l.onnx webapp/public/sam_decoder.onnx
```

인코더(`models/sam_vit_l_0b3195.pth`, 1.2GB)는 용량 때문에 미포함 —
[Meta 공식 체크포인트](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)를 받아 `models/`에 두면 된다.
직접 익스포트하려면 `python scripts/export_decoder_onnx.py`.

## pcb_defect_yolo11n_map50_0.738.pt (5.2MB)

**이 도구로 만든 전용 모델 예시.** PCB 결함 6종(open/short/mousebite/spur/copper/pinhole) 검출.

- 학습: 승인 라벨 42장 (DeepPCB), YOLO11n, 로컬 MPS 약 12분
- 학습 당시 골드 val mAP50 **0.738** · 현재 번들 홀드아웃 10장 재현값
  mAP50 **0.571**(전체 PR, conf=0.001) / **0.333**(도구 운용 conf=0.30)
- 도구의 **누락 최소화** 프로필(conf=0.10+TTA)은 같은 홀드아웃에서
  mAP50 **0.482**, 후보 124개(GT 81개). 기본 43개보다 검수량은 늘지만 누락을 줄인다
- 비교: 같은 데이터에서 Grounding DINO 제로샷은 사실상 검출 실패 (10장에 3개)

제로샷이 통하지 않는 도메인에서 수십 장 시드로 실용 모델이 나온다는 증거물.
그대로 추론에 쓸 수도 있다:

```python
from ultralytics import YOLO
YOLO("release/pcb_defect_yolo11n_map50_0.738.pt").predict("pcb.jpg", conf=0.4)
```

번들 DeepPCB 샘플로 수치를 다시 확인하려면:

```bash
.venv/bin/python scripts/eval_release_pcb.py
```

라이선스 주의: YOLO11n 파생이므로 **AGPL-3.0**.

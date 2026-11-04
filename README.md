# CHO Cell Counting Optimization for BioSEM Research

무염색(label-free) 명시야 CHO 세포 이미지에 대한 **YOLOv8 / YOLOv11 세그멘테이션 및
세포 계수** 파이프라인. 10종 모델을 동일 조건으로 비교하고, 중첩 5-fold 교차검증으로
계수정확도 97% 목표를 검증했다.

참고 논문: G. Park, J. Park, S. Lee, *"Optimal Cell Segmentation and Counting Strategy
for Embedding in Low-Power AIoT Devices"*, Appl. Sci. 2026, 16, 357.
(본 저장소는 해당 논문의 CHO 검증 데이터셋에 YOLO 계열을 적용한 후속 작업)

---

## 핵심 결과

| 항목 | 값 | 측정 조건 |
|---|---|---|
| 계수정확도 (단독 모델 최고) | **0.9756 ± 0.0091** | yolov8x-seg, 중첩 5-fold CV |
| 계수정확도 (앙상블 최고) | **0.9768 ± 0.0063** | yolo11s+yolov8n+yolov8x, vote=3 |
| Dice | 0.9450 ~ 0.9506 | 동일 |
| 인스턴스 F1 @IoU0.5 | 0.9680 ~ 0.9801 | 동일 |
| R² (예측 vs 실측 세포 수) | 0.938 | 60장 전수 |

계수정확도 정의는 참고 논문과 동일: `Acc = 1 − MAE / mean(N_GT)`

논문 CHO 결과(Dice 최고 93.1 / 계수 0.91~0.96) 대비 **Dice +2.0 %p, 계수 +1.7 %p**.

---

## 데이터셋 특성 (전수 실측)

| 항목 | 값 |
|---|---|
| 영상 수 | 60 장 (cho01~cho60.png) |
| 해상도 | 1376 × 1032, PNG, R=G=B (실질 8-bit 그레이스케일) |
| 총 세포 수 | 743 개 |
| 장당 세포 수 | 6 ~ 22 (중간값 12) |
| 세포 등가직경 | p5 41.5 / 중간값 **53.8** / p95 72.7 px |
| 전경 비율 | 1.08 ~ 4.14 % (평균 2.14 %) |
| 세포–배경 콘트라스트 | 약 6 계조 → 무염색 저콘트라스트 명시야 |
| 접촉·겹침 세포 | **0 건** |

### ⚠ GT 주석 형식 — 가장 중요한 전처리

`ground_truth/*.png` 는 **채워진 영역 마스크가 아니라 세포 외곽선**(두께 약 1.19 px)이다.
이를 모르고 그대로 쓰면 세포 크기를 3배 과소추정하고 타일/입력해상도 설계가 전부 틀어진다.

판정 근거 (실측):

| 지표 | 윤곽선 그대로 | 채운 후 |
|---|---|---|
| 등가직경 중간값 | 16.0 px | **53.8 px** |
| 전경 비율 | 0.2 % | **2.14 %** |
| ring/filled 면적비 | 0.089 (≪1 → 윤곽선) | — |

증상으로 역추적하는 법: 예측 centroid 는 GT 와 완벽히 일치하는데 **마스크 면적만 ~10배**
크고 Dice 가 0.1 대, 인스턴스 F1 이 0 이면서 **계수만 정상**이면 GT 가 윤곽선이다.

윤곽이 끊긴 세포 20 개(2.7%)는 `code/gt_fill.py` 의 단계적 보정
(flood-fill → dilate-fill-erode → convexHull)으로 처리했고, 미채움 0 건을 검증했다.

---

## 단독 모델 성능표

모든 모델 동일 조건: fold 분할 / 300 epoch / patience 80 / 동일 증강 / 유효배치 16 /
imgsz 512 / AdamW lr0 1e-3 cosine.

증거 등급을 반드시 구분한다. **등급이 다른 행을 직접 비교하면 안 된다.**

- **A** = 중첩 5-fold 교차검증. 세포 743개 전부가 홀드아웃에 포함. 후처리 conf 도
  평가 fold 를 보지 않고 선택 → 완전 홀드아웃.
- **B** = fold0 단일 스크리닝. 후처리 conf 를 같은 fold 에서 선택 → **체계적으로 낙관적**.
  실측 격차: yolo11s-seg B 0.9806 vs A 0.9690 (**1.2 %p**).

| 모델 | 등급 | 파라미터 | ckpt | 계수정확도 | Dice | mIoU | instF1 | MAE | conf |
|---|:-:|---:|---:|---|---:|---:|---:|---:|---:|
| yolov8n-seg | A | 3.3 M | 6.8 MB | 0.9691 ± 0.0081 | 0.9470 | 0.9488 | 0.9704 | 0.383 | 0.55 |
| yolov8s-seg | B | 11.8 M | 23.9 MB | 0.9742 | 0.9523 | 0.9536 | 0.9682 | 0.333 | 0.25 |
| yolov8m-seg | A | 27.2 M | 54.8 MB | 0.9647 ± 0.0094 | 0.9438 | 0.9460 | 0.9650 | 0.433 | 0.65 |
| yolov8l-seg | B | 45.9 M | 92.3 MB | 0.9742 | 0.9539 | 0.9551 | 0.9745 | 0.333 | 0.45 |
| **yolov8x-seg** | **A** | 71.8 M | 144.0 MB | **0.9756 ± 0.0091** | 0.9450 | 0.9478 | 0.9680 | 0.300 | 0.65 |
| yolo11n-seg | B | 2.8 M | 6.0 MB | 0.9742 | 0.9520 | 0.9535 | 0.9742 | 0.333 | 0.65 |
| yolo11s-seg | A | 10.1 M | 20.5 MB | 0.9690 ± 0.0052 | 0.9429 | 0.9457 | 0.9725 | 0.383 | 0.65 |
| yolo11m-seg | B | 22.4 M | 45.2 MB | 0.9742 | 0.9533 | 0.9546 | 0.9808 | 0.333 | 0.55 |
| yolo11l-seg | B | 27.6 M | 55.9 MB | 0.9742 | 0.9517 | 0.9532 | 0.9744 | 0.333 | 0.25 |
| yolo11x-seg | B | 62.1 M | 124.8 MB | 0.9806 | 0.9464 | 0.9485 | 0.9712 | 0.250 | 0.25 |

### A등급 4종 — 선택 방식별 비교

| 모델 | fixed | **nested** | oracle | 95% CI (nested) |
|---|---:|---|---:|---|
| yolov8n-seg | 0.9719 | **0.9691 ± 0.0081** | 0.9770 | [0.9617, 0.9760] |
| yolov8m-seg | 0.9551 | **0.9647 ± 0.0094** | 0.9673 | [0.9555, 0.9717] |
| yolov8x-seg | 0.9756 | **0.9756 ± 0.0091** | 0.9769 | [0.9700, 0.9847] |
| yolo11s-seg | 0.9635 | **0.9690 ± 0.0052** | 0.9703 | [0.9644, 0.9735] |

- `fixed` — conf 를 fold0 에서 골라 전 fold 에 적용. fold0 에 과적합된 값이 다른 fold 에서
  나쁠 수 있다 (v8m conf=0.25 → 0.9551 로 하락).
- `nested` — fold k 의 conf 를 **k 를 제외한 4개 fold** 에서만 선택. ← **헤드라인 수치**
- `oracle` — 각 fold 자기 자신에서 선택. 낙관 편향, 상한으로만 참고.

편향 제거가 반드시 숫자를 낮추지는 않는다 (yolo11s: fixed 0.9635 → nested 0.9690).

### 모델 크기와 성능

**거의 무관하다.** 3.3 M(yolov8n)이 71.8 M(yolov8x)과 0.65 %p 차이. 세포가 크고(54 px)
희소하고(장당 12개) 균일해 과제 난이도가 낮다. 엣지 배포라면 nano 계열이 압도적으로 유리하다.

지표별 승자도 다르다: 계수 최고 yolov8x, Dice 최고 yolov8l(B등급), instF1 최고 yolo11m(B등급).

---

## 추론 지연시간

입력 1장(1376×1032) → 512 타일 9개 일괄 추론. warmup 5회 후 20회 측정, 중간값.
**개발 머신(RTX 5090 / x86 CPU) 기준이며 엣지 실기 수치가 아니다.**

| 모델 | ckpt | GPU FP32 | GPU FP16 | CPU FP32 |
|---|---:|---|---|---|
| yolo11n-seg | 6.0 MB | 7.5 ms (133.9 fps) | 6.6 ms (150.8 fps) | 75.3 ms (13.3 fps) |
| yolov8n-seg | 6.8 MB | 8.0 ms (124.4 fps) | 7.2 ms (139.2 fps) | 86.2 ms (11.6 fps) |
| yolo11s-seg | 20.5 MB | 9.7 ms (102.7 fps) | 7.9 ms (126.3 fps) | 194.4 ms (5.1 fps) |
| yolov8s-seg | 23.9 MB | 9.6 ms (104.3 fps) | 7.8 ms (128.7 fps) | 210.8 ms (4.7 fps) |
| yolo11m-seg | 45.2 MB | 17.7 ms (56.5 fps) | 11.8 ms (84.9 fps) | 562.7 ms (1.8 fps) |
| yolov8m-seg | 54.8 MB | 17.1 ms (58.7 fps) | 11.3 ms (88.3 fps) | 436.0 ms (2.3 fps) |
| yolo11l-seg | 55.9 MB | 20.6 ms (48.5 fps) | 13.9 ms (71.9 fps) | 668.9 ms (1.5 fps) |
| yolov8l-seg | 92.3 MB | 23.8 ms (42.1 fps) | 15.2 ms (65.8 fps) | 781.1 ms (1.3 fps) |
| yolo11x-seg | 124.8 MB | 35.4 ms (28.3 fps) | 21.5 ms (46.5 fps) | 1283.9 ms (0.8 fps) |
| yolov8x-seg | 144.0 MB | 39.4 ms (25.4 fps) | 21.3 ms (47.0 fps) | 1168.6 ms (0.9 fps) |

---

## 파이프라인

```
GT 윤곽선 PNG ─┐
               ├─ gt_fill.py        윤곽 채움 + 끊긴 윤곽 보정 → 인스턴스 라벨맵
원본 PNG ──────┘
                  build_ds.py       512 타일 / stride 432 → YOLO-seg 폴리곤, 파일단위 5-fold
                  train_yolo.py     동일 하이퍼파라미터로 10종 학습
                  evaluate.py       전체 영상 타일 추론 → 마스크 IoU 병합 → 원본 좌표계 평가
                  nested_cv.py      중첩 CV 집계 (fixed / nested / oracle)
                  stage3b.py        D4 TTA · 다중 모델 앙상블
```

### 재현 방법

```bash
pip install ultralytics opencv-python numpy matplotlib

# 1) GT 윤곽선 채우기 → 인스턴스 라벨맵
python code/prep_instances_filled.py

# 2) 5-fold 타일 데이터셋 생성
python code/build_ds.py

# 3) 10종 전수 학습 + 후처리 스윕 (fold0 스크리닝)
python code/stage1_screen.py

# 4) 상위 모델 5-fold 교차검증
python code/stage2_cv.py yolov8x-seg.pt yolo11s-seg.pt yolov8n-seg.pt

# 5) 중첩 CV 집계 / 앙상블 / 표 / 그림 / 보고서
python code/nested_cv.py
python code/stage3b.py ens3x
python code/single_table.py
python code/make_figures.py --lang en
python code/make_report.py
```

경로 상수는 `code/config.py` 에 모여 있다.

---

## 학습 가중치

파일 크기(최대 144 MB, 총 1.4 GB) 때문에 git 본문이 아니라 **GitHub Releases** 에 올렸다.

- 태그 `weights-v1` → [Releases](../../releases) 참고
- 명명 규칙 `<모델>_f<fold>.pt` (예: `yolov8x-seg_f0.pt`)
- A등급 4종은 fold 0~4 전부, B등급 6종은 fold 0 제공

### 모델 명세 (변환·배포용)

| 항목 | 값 |
|---|---|
| 프레임워크 | Ultralytics YOLO 8.4.150 / PyTorch 2.11.0+cu128 |
| 태스크 | instance segmentation (`task=segment`) |
| 입력 해상도 | **512 × 512** (타일 단위), RGB 3채널 uint8 |
| 원본 영상 | 1376 × 1032 → 512 타일 / stride 432 → 3×3 = 9 타일 |
| 클래스 수 | **1** |
| 클래스 목록 | `0: cell` |
| 정규화 | Ultralytics 내부 `/255.0` (별도 mean/std 없음) |
| 출력 | 박스 + 마스크 (`retina_masks=True` 로 원본 해상도 마스크) |
| 권장 conf | 모델별 상이, 위 표 `conf` 열 참조 (0.25 ~ 0.65) |
| NMS IoU | 0.5 |
| 후처리 | 타일 간 마스크 IoU 병합 (IoU ≥ 0.3 또는 포함비 ≥ 0.6 → 중복 제거) |
| watershed | **사용하지 않음** (아래 참조) |

---

## 효과가 없었던 기법 (그대로 기록)

| 기법 | 결과 |
|---|---|
| distance-transform watershed (Tw) | **역효과** — 전 모델 / 전 conf |
| D4 TTA | **역효과** — Dice 0.9487 → 0.9390 |
| 면적 하한 필터 | **역효과** — 진짜 세포 제거 |
| conf 격자 확장 (0.65 → 0.85) | 효과 없음 — 최적이 격자 내부 |
| 대형 단일 모델 (yolov8x) | 성공 — 0.9756 |
| 다중 모델 합의 투표 앙상블 | 성공 — 0.9768 |

### 참고 논문의 watershed Tw=0.65 는 이 데이터에 적용하면 안 된다

논문은 CHO 데이터에서 "계수 성능이 더 넓은 Tw 범위에서 안정적으로 유지된다"고 적었다.
실측한 원인은 **강건성이 아니라 분리할 접촉 세포가 애초에 0 건**이기 때문이다.
GT 에서 Tw 0.4~0.9 전 구간을 훑어도 추가 분리되는 인스턴스가 0 개였다.

따라서 이 데이터셋에서 watershed 단계는 불필요하거나 유해하며(과분할만 유발),
논문의 Tw=0.65 최적값은 **WBC 데이터 고유 결과**로 한정해 서술하는 것이 정확하다.

### ultralytics `augment=True` 가 segment 에서 no-op

8.4.150 에서 `predict(augment=True)` 의 검출 수와 confidence 가 `augment=False` 와
**완전히 동일**했다. 확인법은 두 번 호출해 `sorted(boxes.conf)` 비교.
그래서 D4(회전4 × 반전2)를 `code/tta.py` 에 직접 구현했으나, 회전/반전 증강으로
학습한 모델에는 이득이 없었다.

---

## 한계 (Limitations)

검증된 것:

- 60장 전수 5-fold 교차검증. 세포 743개 전부가 한 번씩 홀드아웃에 포함
- 파일 단위 분할 (세포 단위 분할은 누출이므로 사용하지 않음)
- 후처리 하이퍼파라미터까지 중첩 선택

**검증되지 않은 것:**

- **독립 배치 일반화** — 단일 배치 60장이므로 다른 날·다른 장비로 촬영한 CHO 데이터에
  대한 전이는 **측정 불가**. N=1 배치로는 일반화를 주장할 수 없다.
- **엣지 가속기 실기 성능** — DeepX DX-M1 / Jetson / 라즈베리파이 실측 미수행.
  위 지연시간 표는 개발 머신(RTX 5090, x86 CPU) 수치다.
- **양자화 후 성능** — FP16 / INT8 변환 후 정확도 열화 미측정.
- **통계적 정밀도** — fold 당 홀드아웃이 12장(약 150 세포). 세포 2~3개 오차가 1.5 %p 를
  움직인다. 신뢰구간은 fold 수준 부트스트랩(n=5)이므로 넓게 해석해야 한다.
- **GT 주석 완전성** — FP 23건의 국소 텍스처 지표가 GT 세포와 사실상 동일
  (std 38.2 vs 38.6, Laplacian 398 vs 422). 상당수가 **GT 주석 누락된 실제 세포**로
  추정되며, 이 경우 측정된 정확도는 모델 성능의 **하한**이다.
  증거: `results/montage_fp.png`, `results/montage_fn.png`

---

## 저장소 구조

| 경로 | 내용 |
|---|---|
| `code/` | 전처리·학습·평가·앙상블·그림·보고서 스크립트 17개 |
| `docs/REPORT.txt` | 전체 분석 보고서 |
| `docs/SINGLE_MODEL_TABLE.txt` | 단독 모델 성능표 (원문) |
| `results/*.json` | 원시 측정 결과 (스윕 전조합, fold별 per-file 기록 포함) |
| `results/montage_*.png` | FP/FN 오차 사례 크롭 |
| `figures/en/` | 저널 투고용 영문 그림 6종 |
| `figures/ko/` | 한글 그림 6종 |

데이터셋 원본(이미지·GT)은 라이선스 확인 전까지 포함하지 않았다.

## 라이선스

코드: MIT. 가중치는 Ultralytics 사전학습 모델에서 파인튜닝되었으므로
**AGPL-3.0** 제약을 따른다 (상업적 활용 시 Ultralytics 라이선스 확인 필요).

## 인용

```
이성진 (Sungjin Lee), 순천향대학교 스마트자동차학과
교신저자: sungjinlee@sch.ac.kr
```

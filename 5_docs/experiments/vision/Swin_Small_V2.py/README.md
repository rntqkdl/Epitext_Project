 # Swin Small V2 Production Experiment

## 목적

- 한자 탁본 이미지를 **대규모 클래스(수천 클래스)** 수준으로 분류하기 위한  
  `swinv2_small_window16_256` 모델 학습 파이프라인을 구축한다.
- **A100 High-RAM 환경**에서 최대 성능을 내도록 데이터 로딩·가중치·증강·학습 루프를
  모두 최적화한 **프로덕션용 학습 스크립트**를 만든다.
- 이후 `Epitext_Service` 백엔드에서 바로 로드해 inference 할 수 있도록  
  **모델 저장 형식과 경로 구조를 서비스와 호환**되게 맞춘다.

---

## 데이터 & 전처리

- 입력 데이터: `*.npz` 샤드 파일
  - 키: `images` (uint8 이미지 배열), `labels` (클래스 인덱스), `metadata` (서체 등 메타데이터)
- `MemoryEfficientDataset`
  - 여러 NPZ 샤드를 **한 번에 메모리로 로딩**해서 학습 속도 최적화
  - High-RAM(A100) 환경 기준으로 설계
- EDA 기반 전처리/증강
  - **EDA-1/2:** 클래스 불균형(1:6072) + 서체 불균형(해서 87.9%) →  
    `compute_style_aware_weights()`로 **클래스+서체 가중치** 계산
  - **EDA-3:** 밝기·대비 차이를 줄이기 위한 강한 ColorJitter +  
    확률적 히스토그램 평활화(`ImageOps.equalize`) 적용
  - **EDA-4:** Train에 없는 클래스가 Val에만 존재하는 문제 →  
    `balance_val_classes()`로 **일부 샘플을 Train으로 이동**하여 분포 안정화

---

## 모델 & 학습 설정

- 모델: `timm.create_model("swinv2_small_window16_256", num_classes=NUM_CLASSES)`
- Loss: `CrossEntropyLoss(weight=class_weights)`
  - `class_weights`는 `compute_style_aware_weights()`에서 계산된  
    **클래스×서체 기반 가중치**
- Optimizer: `AdamW`
  - Backbone과 Head에 **다른 learning rate** 적용  
    (`LR_BACKBONE = 3e-5`, `LR_HEAD = 3e-4`)
- Scheduler:
  - `LinearLR`로 **3 epoch warmup**
  - 이후 `CosineAnnealingWarmRestarts`로 주기적 재시작
- Mixed Precision:
  - `torch.amp.GradScaler("cuda")` + `autocast`로 **AMP 학습**
- Gradient Accumulation:
  - `ACCUM_STEPS = 4` → 실질적인 effective batch = `BATCH_SIZE × 4`

---

## 학습 루프

- `train_epoch()`  
  - AMP + gradient accumulation 사용
  - tqdm으로 loss, accuracy 모니터링
- `validate()`  
  - 전체 Val 셋에 대해 **macro F1** 계산
  - 최근 2,000 샘플에 대해 중간 F1을 출력해 학습 추세 확인
- 체크포인트 & Early Stopping
  - macro F1이 갱신될 때마다  
    `best_model_f1_{score}_E{epoch}.pth` 저장
  - 연속 **3 epoch F1 개선이 없으면 Early Stopping**

---

## 실행 예시 (Colab 기준)

```bash
# Colab에서 char_mapping.json / npz 데이터 경로만 맞춰놓고 실행
python swinv2_production_train.py

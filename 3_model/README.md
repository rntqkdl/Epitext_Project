`3_model`의 모든 학습 및 평가 작업은  
**`1_data` 파이프라인에서 준비된 데이터를 전제 및 drive상 데이터로 실행됩니다..**

모델 실행 전에 반드시 아래 단계를 완료해야 합니다.

1. Google Drive에서 연구 데이터 다운로드
2. 데이터 배치 경로 확인 (`1_data/raw_data`, `1_data/processed`)
3. 필요 시 데이터 전처리 파이프라인 실행

📥 **데이터 다운로드 링크 (필수)**  
👉 https://drive.google.com/drive/folders/1dqhfSy4_nnQTqXvZ3yqMpgbpR1r0nOkn?usp=drive_link

📘 **데이터 파이프라인 상세 설명**  
👉 [`1_data/README.md`](../1_data/README.md)

> ⚠️ 데이터가 존재하지 않거나 경로가 맞지 않으면  
> `sikuroberta_train`, `swin_train`, `ocr_eval` 등의 task는  
> **즉시 오류가 발생하거나 실행되지 않습니다.**

# 🤖 AI 모델 파이프라인 (`3_model`)

`3_model/` 디렉토리는 **Epitext 프로젝트의 핵심 AI 모델**에 대한
학습(Training) 및 평가(Evaluation)를 담당하는 **모델 파이프라인 모듈**입니다.

본 파이프라인은 다음 모델들을 **독립적이면서도 통합적으로** 제어할 수 있도록 설계되었습니다.

- **SikuRoBERTa** (NLP · 한자 언어 모델)
- **Swin Transformer V2** (Vision · 탁본 이미지 복원)
- **OCR** (문자 인식 성능 평가)
- **Gemini** (번역 및 복원 보조 평가)

모든 모델은 `3_model/main.py`를 **단일 Entry Point**로 사용하며,
CLI 인자(`--task`)를 통해 원하는 모델/작업만 선택적으로 실행할 수 있습니다.

---

## 1. 디렉토리 및 파일 구성

```text
3_model/
├── __init__.py          # 패키지 초기화
├── config.py            # 모델 공통 설정 (디바이스, 경로, 시드 등)
├── main.py              # 모델 파이프라인 진입점 (CLI 실행)
│
├── nlp/                 # NLP 모델
│   ├── sikuroberta/
│   │   ├── train/
│   │   │   ├── train_task.py   # SikuRoBERTa 학습 로직
│   │   │   ├── config.py       # 학습 파라미터 설정
│   │   │   └── README.md
│   │   └── __init__.py
│   │
│   └── gemini_experiment/
│       ├── eval_task.py        # Gemini 번역/평가 로직
│       └── __init__.py
│
├── vision/              # Vision 모델
│   ├── swin_experiment/
│   │   ├── train/
│   │   │   └── train_task.py   # SwinV2 학습 로직
│   │   └── eval/
│   │       └── eval_task.py    # Swin 평가 로직
│   │
│   └── ocr_experiment/
│       └── eval_task.py        # OCR 평가 로직
│
└── saved_models/        # 학습된 모델 체크포인트 (Git 추적 제외)
```

> ⚠️ `saved_models/` 디렉토리는 대용량 파일이 포함되므로
> `.gitignore`를 통해 Git 추적 대상에서 제외됩니다.

---

## 2. 주요 작업(Task) 정의

`3_model/main.py`는 `--task` 인자를 통해 아래 작업들을 지원합니다.

### ✅ NLP 모델

| Task 이름           | 설명                                  |
| ------------------- | ------------------------------------- |
| `sikuroberta_train` | SikuRoBERTa 언어 모델 학습 (MLM 기반) |
| `sikuroberta_eval`  | SikuRoBERTa 모델 성능 평가            |
| `gemini_eval`       | Gemini API 기반 번역 / 복원 평가      |

### ✅ Vision 모델

| Task 이름    | 설명                                      |
| ------------ | ----------------------------------------- |
| `swin_train` | Swin Transformer V2 이미지 복원 모델 학습 |
| `swin_eval`  | Swin 모델 성능 평가                       |
| `ocr_eval`   | OCR 문자 인식 성능 평가                   |

### ✅ 통합 실행

| Task 이름   | 설명                                    |
| ----------- | --------------------------------------- |
| `all_train` | 모든 학습 작업 순차 실행 (NLP + Vision) |
| `all_eval`  | 모든 평가 작업 순차 실행                |

---

## 🧠 추론 / 복원 파이프라인 (Inference Pipelines)

본 프로젝트는 학습(train) 및 평가(eval) 외에도  
**실제 손상된 탁본 데이터를 복원하기 위한 추론 전용 파이프라인**을 제공합니다.

### ✅ Swin Mask Restoration (`swin_restore`)

- OCR 결과의 `MASK2` 영역 이미지를 입력으로 사용
- SwinV2 모델을 이용해 **손상된 문자 이미지 Top-K 후보 복원**
- 입력:
  - OCR JSON (`box`, `type=MASK2`)
  - 원본 탁본 이미지
- 출력:
  - JSON (각 마스크별 Top-K 문자 + 확률)

````bash
python -m 3_model.main --task swin_restore


## 3. 실행 방법 (중요)

모든 명령은 **프로젝트 루트 디렉토리**에서 실행해야 합니다.
(`3_model`은 숫자로 시작하는 패키지이므로 `-m` 옵션을 사용합니다)

```bash
# SikuRoBERTa 학습
python -m 3_model.main --task sikuroberta_train

# Swin Transformer 학습
python -m 3_model.main --task swin_train

# OCR 평가
python -m 3_model.main --task ocr_eval

# Gemini 번역 평가
python -m 3_model.main --task gemini_eval

# 모든 학습 작업 실행
python -m 3_model.main --task all_train

# 모든 평가 작업 실행
python -m 3_model.main --task all_eval
````

> ❗ `python 3_model/main.py` 방식은 상대 import 문제로 인해
> 정상 동작하지 않을 수 있으므로 **반드시 `-m` 옵션을 사용**하세요.

---

## 4. 설정 파일 (`config.py`)

모델 실행과 관련된 공통 설정은 `3_model/config.py`에서 관리합니다.

### 주요 설정 항목

- `DEVICE`

  - `"cuda"` 또는 `"cpu"`

- `RANDOM_SEED`

  - 재현성을 위한 랜덤 시드 값

- **데이터 경로**

  - `SWIN_DATA_DIR` : Swin 모델용 전처리 이미지 데이터 (`.npz`)
  - `SIKU_TRAIN_DIR`, `SIKU_VALID_DIR`, `SIKU_TEST_DIR`
    : SikuRoBERTa 토큰화 텍스트 데이터셋

> ⚠️ 모든 데이터 경로는 `1_data` 파이프라인의 출력 결과와 **정확히 일치해도 됩니다.**

---

## 5. 설계 철학 (중요)

- `3_model/main.py`
  → **무엇을 실행할지 결정 (Controller)**
- 각 모델의 `train_task.py` / `eval_task.py`
  → **실제 학습·평가 로직 (Execution)**
- `config.py`
  → **설정과 경로 관리**

이 구조를 통해 다음이 가능합니다:

- 모델별 **완전 독립 실행**
- 실험 추가 시 기존 코드 최소 수정
- Slurm / MLflow / W&B 등 실험 관리 도구 확장 용이

---

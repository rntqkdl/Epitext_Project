# 📊 데이터 파이프라인 (`1_data`)

`1_data/` 디렉토리는 **Epitext 프로젝트의 전체 데이터 파이프라인**을 담당합니다.
원본 데이터 수집부터 전처리, 학습용 데이터셋 생성, EDA까지
**모든 모델 실험의 출발점**이 되는 모듈입니다.

> ✅ 이 디렉토리의 출력 결과는
> `3_model` 이하 모든 모델(SikuRoBERTa, Swin, OCR, Gemini)의 **입력 데이터**로 할 수 있습니다.

---

## 1. 디렉토리 구조

```text
1_data/
├── raw_data/              # (Git 제외) 원본 데이터 및 외부 제공 데이터
│   ├── doc_id_transcript_dataset.csv
│   ├── doc_id_transcript_dataset_processed.csv
│   ├── pun_ksm_gsko.csv
│   │
│   ├── processed_dataset/ # 텍스트 전처리 결과
│   └── split_dataset/
│       └── tokenized_sikuroberta_simple_128_split/
│           ├── train/
│           ├── validation/
│           └── test/
│
├── preprocess/            # 데이터 전처리 로직
│   ├── vision/            # 이미지 전처리 (OCR, OpenCV 등)
│   └── text/              # 텍스트 정제 및 토큰화
│
├── crawlers/              # 원문 데이터 수집 크롤러
│
├── eda/                   # 데이터 분석 및 시각화
│
└── processed/             # (Git 제외) 모델 입력용 가공 데이터
    └── swin_data/         # Swin Transformer 학습용 .npz 이미지 데이터
```

> ⚠️ `raw_data/`, `processed/` 디렉토리는
> **대용량 데이터 포함**으로 인해 `.gitignore` 되어 있습니다.

---

## 2. 데이터 다운로드 (필수)

본 프로젝트의 실제 데이터는 **GitHub에 포함되어 있지 않습니다.**
아래 Google Drive 링크를 통해 **반드시 직접 다운로드**해야 합니다.

### 🔗 Google Drive 데이터 링크

👉 [https://drive.google.com/drive/folders/1dqhfSy4_nnQTqXvZ3yqMpgbpR1r0nOkn?usp=drive_link](https://drive.google.com/drive/folders/1dqhfSy4_nnQTqXvZ3yqMpgbpR1r0nOkn?usp=drive_link)

---

### 📂 데이터 배치 방법

다운로드한 데이터를 **압축 해제 후 아래 경로에 배치**하세요.

```text
Epitext_Project/
└── 1_data/
    ├── raw_data/
    │   ├── *.csv
    │   ├── processed_dataset/
    │   └── split_dataset/
    │
    └── processed/
        └── swin_data/
            └── *.npz
```

> ❗ 경로명이 다르면
> `3_model`에서 학습/평가 실행 시 에러가 발생합니다.

---

## 3. 데이터 파이프라인 역할 정리

### ✅ raw_data/

- 외부 기관 제공 원본 데이터
- 크롤링 결과
- 전처리 이전의 **모든 기준 데이터**

📌 Git에 **절대 커밋하지 않음**

---

### ✅ preprocess/

데이터를 **모델 학습 가능한 형태로 가공**합니다.

- Vision

  - OCR 노이즈 제거
  - 이미지 크기/형식 통일

- Text

  - 특수문자 제거
  - 정규화
  - 토큰화 전 텍스트 정제

---

### ✅ processed/

모델에서 직접 사용하는 최종 데이터입니다.

- `swin_data/`

  - Swin Transformer 학습용 `.npz` 파일

- NLP 모델은 `raw_data/split_dataset`을 직접 사용

---

### ✅ eda/

데이터 품질 및 분포를 분석합니다.

- 문장 길이 분포
- 문자 빈도
- OCR 결과 통계
- 시각화 결과

EDA 결과는 **모델 학습 이전 검증 단계**로 사용됩니다.

---

## 4. 실행 방법

### ✅ 데이터 전처리

```bash
# 전체 전처리 실행
python main.py --phase data --step preprocess
```

### ✅ EDA 실행

```bash
python main.py --phase data --step eda
```

> 위 명령은 프로젝트 루트에서 실행해야 합니다.

---

## 5. 3_model과의 관계 (중요)

| 데이터 위치               | 사용 모델          |
| ------------------------- | ------------------ |
| `raw_data/split_dataset/` | SikuRoBERTa        |
| `processed/swin_data/`    | Swin Transformer   |
| OCR 결과 / GT             | OCR Evaluation     |
| 텍스트 결과               | Gemini Translation |

👉 **`1_data`는 모든 모델 실험의 단일 진입 데이터 소스(Single Source of Truth)**입니다.

---

## ⚠️ 주의사항 요약

- ✅ Drive 데이터 다운로드 필수
- ✅ 경로 구조 엄수
- ✅ 대용량 데이터는 Git에 올리지 않음
- ✅ 모델 실행 전 `1_data` 준비 확인

---

## ✅ 마지막 한 줄 요약

> `1_data/`는 **모든 실험의 출발점**이며,
> 이 디렉토리가 깨끗하게 유지될수록
> 모델 실험의 재현성과 신뢰도가 보장됩니다.

---

## 🔗 다음 단계: 모델 파이프라인 (`3_model`)

`1_data` 파이프라인에서 생성된 데이터는  
아래 모델 파이프라인에서 **직접 사용됩니다**.

👉 **모델 학습 및 평가 방법은 아래 문서를 참고하세요.**  
📘 [`3_model/README.md`](../3_model/README.md)

> ⚠️ 모델 실행 전 반드시  
> `1_data/raw_data/` 및 `1_data/processed/`가  
> 올바르게 준비되어 있어야 합니다.

---

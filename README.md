# Epitext Project – Research & Experiment Repository

본 저장소는 **한자 탁본 자동 복원 AI 시스템**을 연구·실험하기 위한
**연구 전용 코드 및 실험 관리 저장소**입니다.

서비스 운영을 위한 백엔드/프론트엔드 코드는
백엔드 저장소 [`jae2022/Epitext_Back`](https://github.com/jae2022/Epitext_Back)
프론트엔드 저장소 [`jae2022/Epitext_Front`](https://github.com/jae2022/Epitext_Front)

---

## 📂 프로젝트 구조 (Project Structure)

이 저장소는 **데이터 파이프라인 → 모델 학습/평가 → 실험 기록**까지
연구 전 과정을 체계적으로 관리하도록 구성되어 있습니다.

```text
Epitext_Project/
├── 1_data/                 # 데이터 파이프라인 (수집, 전처리, EDA)
│   ├── raw_data/           # (Git 제외) 원본 데이터 저장소 (.gitignore 적용)
│   ├── preprocess/         # Vision(EasyOCR) 및 NLP(Text Clean) 전처리 모듈
│   ├── crawlers/           # 데이터 수집 크롤러
│   └── eda/                # 데이터 분석 스크립트
│
├── 2_notebooks/            # 실험 및 프로토타이핑용 Jupyter Notebook
│
├── 3_model/                # 모델 학습 및 평가 파이프라인
│   ├── nlp/                # NLP 모델 (SikuRoBERTa, Gemini)
│   │   ├── sikuroberta/    # MLM 학습 및 평가
│   │   └── gemini_experiment/ # 번역 실험
│   ├── vision/             # Vision 모델 (Swin Transformer, OCR)
│   │   ├── swin_experiment/   # 이미지 복원 학습
│   │   └── ocr_experiment/    # OCR 성능 평가
│   └── saved_models/       # (Git 제외) 학습된 모델 가중치 저장소
│
├── 5_docs/                 # 연구 노트 및 과거 시행착오 기록 (Experiments Archive)
│
├── main.py                 # ✨ 통합 실행 컨트롤러 (Entry Point)
├── config.py               # (Optional) 프로젝트 전역 설정
├── requirements.txt        # 통합 의존성 패키지 목록
└── .env                    # (Git 제외) API Key 및 환경 변수 설정 파일
```

> ⚠️ `raw_data/`, `saved_models/`, `.env`는 대용량/민감 정보 포함 가능성으로
> Git 추적 대상에서 제외됩니다.

---

## 🚀 시작하기 (Getting Started)

### 1. 환경 설정 및 의존성 설치

Python **3.9 이상** 환경을 권장합니다.

```bash
# 1. 저장소 클론
git clone https://github.com/rntqkdl/Epitext_Project.git
cd Epitext_Project

# 2. 가상환경 생성 (Conda 권장)
conda create -n epitext python=3.10
conda activate epitext

# 3. 통합 의존성 설치
pip install -r requirements.txt
```

> ⚠️ 본 프로젝트는 `torch`, `transformers`, `timm` 등
> **딥러닝/비전 패키지 버전에 민감**하므로
> 반드시 `requirements.txt` 기준으로 설치하는 것을 권장합니다.

---

## 📁 데이터 다운로드 및 배치 방법 (중요)

본 프로젝트에서 사용되는 **대규모 학습 및 평가 데이터는 GitHub에 포함되어 있지 않습니다.**
용량 문제 및 라이선스 이슈로 인해, 모든 실제 데이터는 **Google Drive를 통해 제공**됩니다.

### 🔗 데이터 다운로드 링크

아래 Google Drive 링크에서 전체 데이터를 다운로드하세요.

👉 **Google Drive 데이터 폴더**
[https://drive.google.com/drive/folders/1dqhfSy4_nnQTqXvZ3yqMpgbpR1r0nOkn?usp=drive_link](https://drive.google.com/drive/folders/1dqhfSy4_nnQTqXvZ3yqMpgbpR1r0nOkn?usp=drive_link)

---

### 📂 데이터 배치 방법

다운로드한 데이터는 압축 해제 후,
반드시 프로젝트 루트 기준으로 아래 구조에 맞게 배치해야 합니다.

```text
Epitext_Project/
└── 1_data/
    ├── raw_data/
    │   ├── doc_id_transcript_dataset.csv
    │   ├── doc_id_transcript_dataset_processed.csv
    │   ├── pun_ksm_gsko.csv
    │   ├── processed_dataset/
    │   └── split_dataset/
    │       └── tokenized_sikuroberta_simple_128_split/
    │           ├── train/
    │           ├── validation/
    │           └── test/
    │
    └── processed/
        └── swin_data/
            └── *.npz
```

> ⚠️ 주의
>
> - `raw_data/`, `processed/` 하위 데이터는 `.gitignore`로 인해 Git에 커밋되지 않습니다.
> - 경로 구조가 다를 경우 모델 학습/평가가 정상 동작하지 않습니다.

---

## 💻 실행 방법 (Usage)

### ✅ 통합 진입점

```bash
# 데이터 전처리
python main.py --phase data --step preprocess

# EDA 수행
python main.py --phase data --step eda
```

---

### ✅ 모델 파이프라인 (3_model)

> ❗ `3_model` 디렉토리는 숫자로 시작하므로
> **반드시 `-m` 옵션을 사용해 실행해야 합니다.**

```bash
# SikuRoBERTa 학습
python -m 3_model.main --task sikuroberta_train

# Swin Transformer 학습
python -m 3_model.main --task swin_train

# OCR 평가
python -m 3_model.main --task ocr_eval

# Gemini 번역 평가
python -m 3_model.main --task gemini_eval
```

---

## ⚠️ 데이터 관련 주의사항 (중요)

각 모델 task는 **해당 데이터가 존재할 경우에만 정상 실행**됩니다.

- `swin_train`

  - `1_data/processed/swin_data/` 하위 `.npz` 파일 필요

- `sikuroberta_train`

  - 토큰화된 train / validation 데이터 필요

- `ocr_eval`

  - GT / Prediction 데이터 경로 설정 필요

👉 반드시 **`1_data` 파이프라인 수행 또는 Drive 데이터 배치 후 실행**하세요.

---

## ⚙️ 설정 관리 (Configuration)

| 모듈        | 설정 파일                                  |
| ----------- | ------------------------------------------ |
| SikuRoBERTa | `3_model/nlp/sikuroberta/train/config.py`  |
| Swin        | `3_model/vision/swin_experiment/config.py` |
| OCR         | `3_model/vision/ocr_experiment/config.py`  |
| Gemini      | `3_model/nlp/gemini_experiment/config.py`  |

---

## 🧪 실험 기록 및 아카이브

`5_docs/experiments/`에는 다음이 보존됩니다.

- **NLP 실험**

  - 다양한 SikuRoBERTa 학습 전략
  - ExaOne / Qwen 번역 비교

- **Vision 실험**

  - OCR 모델 비교 (EasyOCR, Paddle, AIHub)
  - OpenCV/전처리 실험

---

## 📝 라이선스 및 출처

- **License**: MIT License
- **Data Source**: 서울대학교 규장각, 국사편찬위원회, 국립문화재연구소
- **Team**: 4조 복원왕 김탁본

---

### ✅ 마지막 한 줄 정리

> 이 README는 **연구 재현·확장·협업을 고려한 최종 형태**이며,
> 이후 모델이 추가되더라도 `3_model/main.py`에 task만 등록하면
> 동일한 방식으로 확장할 수 있습니다.

---

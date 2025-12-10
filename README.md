# Epitext Project – Research & Experiment Repository

본 저장소는 **한자 탁본 자동 복원 AI 시스템**을 연구·실험하기 위한 **연구 전용 코드 및 실험 관리 저장소**입니다.

서비스 운영을 위한 백엔드/프론트엔드 코드는 별도 저장소 [`rntqkdl/Epitext_Service`]에서 관리합니다.

---

## 프로젝트 개요

Epitext Project는 다음과 같은 연구 목적을 중심으로 구성되어 있습니다.

- **데이터 파이프라인**: 한자 탁본 데이터 수집 및 전처리 파이프라인 구축
- **모델 연구**: NLP(SikuRoBERTa, Gemini) 및 Vision(SwinV2, OCR) 모델 실험
- **실험 관리**: 모델 학습, 평가, 분석 및 실험 기록 관리
- **아카이빙**: 연구 과정에서 발생하는 시행착오 및 중간 결과 정리

본 저장소는 **재현 가능한(Reproducible) 연구 실험 환경 제공**을 목표로 합니다.

---

## 프로젝트 구조

```text
Epitext_Project/
├── 1_data/                 # 데이터 파이프라인
│   ├── __init__.py
│   ├── config.py           # 데이터 실행 설정
│   ├── main.py             # 데이터 파이프라인 진입점
│   ├── README.md
│   ├── crawlers/           # 데이터 수집 (Selenium, Requests)
│   ├── preprocess/         # 데이터 전처리 (NLP, Vision)
│   ├── eda/                # 탐색적 데이터 분석
│   ├── utils/              # 공통 유틸리티
│   └── sample_data/        # 소규모 샘플 데이터 (Git 포함)
│
├── 2_notebooks/            # Jupyter 노트북 (EDA / 프로토타이핑)
│
├── 3_model/                # 모델 학습 및 평가
│   ├── __init__.py
│   ├── config.py           # 모델 공통 설정
│   ├── main.py             # 모델 파이프라인 진입점
│   ├── README.md
│   ├── nlp/
│   │   ├── sikuroberta/    # MLM 학습 및 평가
│   │   └── gemini_experiment/ # 번역 및 생성 실험
│   ├── vision/
│   │   ├── swin_experiment/   # 이미지 복원 실험
│   │   └── ocr_experiment/    # 문자 인식 실험
│   └── saved_models/       # 학습된 모델 저장 (Git 제외)
│
├── experiments/            # 실험 설정(YAML) 및 로그 (권장)
│
├── 5_docs/                 # 연구 문서, 리포트, 시행착오 기록
│
├── main.py                 # 통합 실행 진입점 (Single Entry Point)
├── requirements.txt        # 의존성 패키지 목록
├── README.md
└── test.py
```

⚠️ 본 프로젝트에서 사용되는 대규모 학습 및 평가 데이터는 용량 문제로 인해 Git 저장소에 포함되어 있지 않습니다.

모든 실제 데이터는 Google Drive에 저장되며, 로컬 환경에 직접 다운로드하여 아래 구조에 맞게 배치해야 합니다.

📁 Google Drive 데이터 링크: (연구실 공유 드라이브 링크)
1_data/
├── raw_data/
│ ├── doc_id_transcript_dataset.csv
│ ├── doc_id_transcript_dataset_processed.csv
│ ├── pun_ksm_gsko.csv
│ ├── processed_dataset/ # SikuRoBERTa 학습 데이터
│ └── split_dataset/ # Train/Val/Test Split
│ └── tokenized_sikuroberta_simple_128_split/
│ ├── train/
│ ├── validation/
│ └── test/
│
├── processed/
│ └── swin_data/ # SwinV2 입력용 데이터 (.npz)
Note: 1_data/raw_data/ 및 1_data/processed/ 하위의 대용량 파일은 .gitignore에 의해 추적되지 않습니다.
빠른 시작 (Quick Start)

1. 환경 설정

# 저장소 클론

git clone [https://github.com/rntqkdl/Epitext_Project.git](https://github.com/rntqkdl/Epitext_Project.git)
cd Epitext_Project

# Conda 가상환경 생성 (Python 3.10 권장)

conda create -n epitext python=3.10
conda activate epitext

# 의존성 패키지 설치

pip install -r requirements.txt
⚠️ 본 프로젝트는 torch, transformers 등 주요 패키지 버전에 민감하므로 requirements.txt에 명시된 버전을 사용하는 것을 권장합니다.

2. 실행 방법
   프로젝트 루트의 main.py를 통해 모든 파이프라인을 제어할 수 있습니다.
   python main.py --phase all

데이터 파이프라인 실행

# 전체 데이터 프로세스 실행

python main.py --phase data --step all

# 개별 단계 실행

python main.py --phase data --step crawl # 데이터 수집
python main.py --phase data --step preprocess # 전처리
python main.py --phase data --step eda # EDA 수행

모델 파이프라인 실행

# SikuRoBERTa 학습

python main.py --phase model --task sikuroberta_train

# 모든 모델 평가

python main.py --phase model --task all_eval

# SikuRoBERTa 학습

python main.py --phase model --task sikuroberta_train

# 모든 모델 평가

python main.py --phase model --task all_eval

실험 대상 모델

NLP 모델

모델 설명 비고
SikuRoBERTa 한자 기반 언어 모델 (Masked Language Modeling) 메인 복원 모델
Gemini 멀티모달 기반 LLM 보조 실험 번역 및 문맥 보정

Vision 모델

모델 설명 비고
SwinV2 탁본 이미지 결측 복원 및 노이즈 제거 이미지 복원
OCR 한자 문자 인식 및 위치 탐지 텍스트 추출

재현성 (Reproducibility)
모든 실험은 고정된 RANDOM_SEED 기반으로 실행됩니다.

동일한 데이터 및 설정을 사용할 경우 결과 재현을 보장합니다.

실험 설정(config) 및 실행 로그는 experiments/ 디렉터리에 기록 및 관리됩니다.

라이선스
MIT License

연락처
GitHub: @rntqkdl

Team: 4조 복원왕 김탁본

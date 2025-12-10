# Epitext Project – 연구 및 실험 저장소

한자 탁본 자동 복원 AI 시스템 연구 및 실험 저장소입니다.

## 개요

이 저장소는 **연구 및 실험 코드**를 포함하며, 다음과 같은 업무를 수행하는 데 사용됩니다:

- 데이터 수집 및 전처리 파이프라인 구축
- NLP/비전 모델 학습 및 평가 실험
- 연구 결과 및 문서 관리

서비스용 백엔드/프론트엔드 코드는 별도 저장소(`rntqkdl/Epitext_Service`)에서 관리합니다.

## 프로젝트 구조

```
Epitext_Project/
├── 1_data/                 # 데이터 파이프라인
│   ├── __init__.py        # 경로 설정
│   ├── config.py          # 실행 설정
│   ├── main.py            # 오케스트레이터
│   ├── README.md          # 상세 설명서
│   ├── crawlers/          # 데이터 크롤러
│   ├── preprocess/        # 데이터 전처리
│   ├── eda/               # 탐색적 데이터 분석
│   ├── utils/             # 공통 유틸리티
│   └── sample_data/       # 샘플 데이터
│
├── 2_notebooks/           # Jupyter 노트북 (EDA, 실험)
│
├── 3_model/               # 모델 학습/평가
│   ├── __init__.py        # 경로 설정
│   ├── config.py          # 모델 설정
│   ├── main.py            # 오케스트레이터
│   ├── README.md          # 상세 설명서
│   ├── nlp/               # NLP 모델
│   │   ├── sikuroberta/   # SikuRoBERTa 실험
│   │   └── gemini_experiment/  # Gemini 실험
│   ├── vision/            # Vision 모델
│   │   ├── swin_experiment/    # SwinV2 실험
│   │   └── ocr_experiment/     # OCR 실험
│   └── saved_models/      # 학습된 모델 저장
│
├── 5_docs/                # 문서 및 리포트
│
├── main.py                # 통합 실행 진입점
├── requirements.txt       # Python 의존성
├── README.md              # 이 파일
└── test.py                # 테스트 스크립트
```

## 빠른 시작

### 설치

```bash
# 저장소 복제
git clone https://github.com/rntqkdl/Epitext_Project.git
cd Epitext_Project

# 가상환경 생성 및 활성화
conda create -n epitext python=3.10
conda activate epitext

# 의존성 설치
pip install -r requirements.txt
```

## 실행 방법

### 방법 1: 통합 진입점 (권장)

`main.py`를 사용하면 데이터와 모델을 한 번에 제어할 수 있습니다.

#### 전체 파이프라인

```bash
# 데이터 처리 + 모든 모델 학습
python main.py --phase all --task all_train
```

#### 데이터만 처리

```bash
python main.py --phase data --step all
python main.py --phase data --step crawl
python main.py --phase data --step preprocess
python main.py --phase data --step eda
```

#### 모델만 실행

```bash
python main.py --phase model --task sikuroberta_train
python main.py --phase model --task all_eval
```

### 방법 2: 모듈별 직접 실행

더 세밀한 제어가 필요하면 각 모듈의 `main.py`를 직접 실행합니다.

#### 데이터 파이프라인 (1_data/main.py)

```bash
# 전체 데이터 처리
python 1_data/main.py --step all

# 크롤링만
python 1_data/main.py --step crawl

# 전처리만
python 1_data/main.py --step preprocess

# EDA만
python 1_data/main.py --step eda
```

자세한 내용: [1_data/README.md](./1_data/README.md)

#### 모델 파이프라인 (3_model/main.py)

```bash
# SikuRoBERTa 학습
python 3_model/main.py --task sikuroberta_train

# SwinV2 학습
python 3_model/main.py --task swin_train

# 모든 모델 학습
python 3_model/main.py --task all_train

# 모든 모델 평가
python 3_model/main.py --task all_eval
```

자세한 내용: [3_model/README.md](./3_model/README.md)

## 실행 옵션

### 공통 옵션

```bash
-v, --verbose     상세 로그 출력
-h, --help        도움말 표시
```

### main.py 옵션

```bash
--phase {data|model|all}              실행 단계 (기본값: all)
--step {crawl|preprocess|eda|all}    데이터 단계 (phase=data일 때, 기본값: all)
--task {TASK_NAME}                    모델 작업 (phase=model일 때, 기본값: sikuroberta_train)
```

## 모델 목록

### NLP 모델

| 모델 | 설명 | 경로 | 실행 명령어 |
|------|------|------|----------|
| SikuRoBERTa | 한자 인식 | `3_model/nlp/sikuroberta/` | `python 3_model/main.py --task sikuroberta_train` |
| Gemini | 멀티모달 평가 | `3_model/nlp/gemini_experiment/` | `python 3_model/main.py --task gemini_eval` |

### Vision 모델

| 모델 | 설명 | 경로 | 실행 명령어 |
|------|------|------|----------|
| SwinV2 | 이미지 복원 | `3_model/vision/swin_experiment/` | `python 3_model/main.py --task swin_train` |
| OCR | 문자 인식 | `3_model/vision/ocr_experiment/` | `python 3_model/main.py --task ocr_eval` |

## 설정 파일

### 1_data/config.py

데이터 파이프라인 제어:

```python
RUN_CRAWL = True          # 크롤링 실행 여부
RUN_PREPROCESS = True     # 전처리 실행 여부
RUN_EDA = True            # EDA 실행 여부
```

### 3_model/config.py

모델 학습/평가 제어:

```python
RANDOM_SEED = 42          # 재현성 보장
DEVICE = "cuda"           # "cuda" 또는 "cpu"
```

## 개발 가이드

### 새 모델 추가

1. `3_model/nlp/` 또는 `3_model/vision/` 아래 새 폴더 생성
2. `train/main.py`와 `evaluation/main.py` 작성
3. `3_model/main.py`에 실행 함수와 task 추가

### 새 전처리 파이프라인 추가

1. `1_data/preprocess/` 아래 새 모듈 생성
2. `main()` 함수 구현
3. `1_data/main.py`의 `run_preprocess()`에 호출 추가

## 주요 의존성

- **PyTorch**: `torch`, `torchvision`
- **NLP**: `transformers`, `tokenizers`
- **Vision**: `opencv-python`, `torchvision`
- **Data**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`

자세한 내용: [requirements.txt](./requirements.txt)

## 참고 사항

- 모든 모델은 `3_model/saved_models/`에 저장됩니다.
- 처리된 데이터는 `1_data/processed/`에 저장됩니다.
- 각 모듈의 상세 문서는 해당 폴더의 README.md를 참고하세요.

## 라이센스

MIT License

## 연락처

- GitHub: [@rntqkdl](https://github.com/rntqkdl)

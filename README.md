# Epitext Project - Research Repository

한자 탁본 자동 복원 AI 시스템 연구 및 실험 저장소

## 개요

이 저장소는 Epitext 프로젝트의 **연구 및 실험 코드**를 포함합니다.
- 데이터 수집 및 전처리 파이프라인
- 모델 학습 및 평가 실험
- 연구 결과 및 문서

## 저장소 구조

    Epitext_Project/
    ├── 1_data/              # 데이터 처리 파이프라인
    ├── 2_notebooks/         # Jupyter 실험 노트북
    ├── 3_model/             # 모델 학습 및 평가
    ├── 5_docs/              # 연구 문서
    ├── requirements.txt     # Python 의존성
    └── README.md

## 주요 구성 요소

### 1_data: 데이터 파이프라인
- **crawlers**: 한국학중앙연구원 등 웹 크롤링
- **preprocess**: NLP 및 Vision 전처리
- **eda**: 탐색적 데이터 분석

### 3_model: 모델 실험
- **nlp/sikuroberta**: 한자 MLM 모델 학습 및 평가
- **nlp/gemini_experiment**: LLM 기반 번역 실험
- **vision/swin_experiment**: Swin Transformer 기반 이미지 복원
- **vision/ocr_experiment**: OCR 성능 평가

## 환경 설정

### 요구사항
- Python 3.8+
- CUDA 11.8+ (GPU 학습 시)
- 최소 32GB RAM (대용량 데이터 처리 시)

### 설치
```bash
# 저장소 복제
git clone https://github.com/rntqkdl/Epitext_Project.git
cd Epitext_Project

# 가상환경 생성
conda create -n epitext python=3.10
conda activate epitext

# 의존성 설치
pip install -r requirements.txt
실행 가이드
1. 데이터 수집
bash
cd 1_data/crawlers
python main.py
2. 데이터 전처리
bash
# NLP 전처리
cd 1_data/preprocess/nlp
python text_clean.py

# Vision 전처리
cd ../vision
python easyocr_filter.py
3. 모델 학습
bash
# SikuRoBERTa 학습
cd 3_model/nlp/sikuroberta/train
python train_task.py

# Swin Transformer 학습
cd ../../../vision/swin_experiment/train
python train_task.py
4. 모델 평가
bash
# SikuRoBERTa 평가
cd 3_model/nlp/sikuroberta/evaluation
python evaluate_task.py
연관 저장소
Epitext_Service: 프로덕션 서비스 코드 (backend/frontend)

모델 공유
학습된 모델은 다음 방법으로 서비스 저장소와 공유됩니다:

Git LFS (소규모 모델)

클라우드 스토리지 (대규모 모델)

모델 레지스트리 (버전 관리)

라이센스
이 프로젝트는 연구 목적으로 사용됩니다.

문의
저장소: https://github.com/rntqkdl/Epitext_Project

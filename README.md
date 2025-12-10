# Epitext Project – 연구 및 실험 저장소

한자 탁본 자동 복원 AI 시스템 연구 및 실험 저장소입니다.

## 개요

이 저장소는 Epitext 프로젝트의 **연구 및 실험 코드**를 포함하며, 다음과 같은 업무를 수행하는 데 사용됩니다:

- 데이터 수집 및 전처리 파이프라인 구축
- NLP/비전 모델 학습 및 평가 실험
- 연구 결과 및 문서 관리

서비스용 백엔드/프론트엔드 코드는 별도 저장소(`rntqkdl/Epitext_Service`)에서 관리합니다. 따라서 이 저장소에는 연구 개발에 필요한 디렉터리와 스크립트만 포함되어 있습니다.

## 저장소 구조

```
Epitext_Project/
├── 1_data/              # 데이터 수집 및 전처리 파이프라인
├── 2_notebooks/         # Jupyter 실험 노트북
├── 3_model/             # 모델 학습 및 평가
├── 5_docs/              # 연구 문서 및 보고서
├── outputs/             # 파이프라인 실행 결과
├── requirements.txt     # Python 의존성 목록
└── main.py              # 전체 연구 파이프라인 실행 스크립트
```

## 주요 구성 요소

### 1_data: 데이터 파이프라인

- **crawlers**: 한국학중앙연구원 등에서 데이터를 수집하는 크롤러 코드
- **preprocess**: NLP 및 Vision 데이터를 정제하는 전처리 스크립트
- **eda**: 탐색적 데이터 분석을 위한 코드

### 2_notebooks

- EDA나 실험 과정을 정리한 Jupyter 노트북을 저장합니다. 노트북 사용 시 `requirements.txt`에 명시된 패키지를 설치하고, 데이터 경로를 적절히 수정하세요.

### 3_model: 모델 실험

- **nlp/sikuroberta**: 한자 MLM 모델(SikuRoBERTa) 학습 및 평가
- **nlp/gemini_experiment**: 대형 언어 모델을 활용한 번역 실험
- **vision/swin_experiment**: Swin Transformer 기반 이미지 복원 모델 학습
- **vision/ocr_experiment**: OCR 성능 비교 및 평가

### 5_docs: 문서

- 연구 결과, 실험 보고서, API 명세 등을 Markdown 형식으로 정리합니다.

### outputs

- `main.py` 실행 후 생성되는 요약 파일이나 모델 성능 지표를 보관하는 디렉터리입니다. 필요에 따라 파일과 서브디렉터리를 추가해 확장할 수 있습니다.

## 환경 설정

### 요구 사항

- Python 3.8 이상
- CUDA 11.8 이상 (GPU 학습 시)
- 최소 32GB RAM (대용량 데이터 처리 시)

### 설치

```bash
# 저장소 복제
git clone https://github.com/rntqkdl/Epitext_Project.git
cd Epitext_Project

# 가상환경 생성 및 활성화 (예: conda 사용)
conda create -n epitext python=3.10
conda activate epitext

# 의존성 설치
pip install -r requirements.txt
```

## 실행 가이드

전체 파이프라인은 루트 디렉터리의 `main.py`에서 순차적으로 실행할 수 있습니다. 단계별로 테스트하거나, 필요한 부분만 실행하려면 아래 명령을 참고하세요.

1. **데이터 수집**

   ```bash
   cd 1_data/crawlers
   python main.py
   ```

2. **데이터 전처리**

   - NLP 전처리:
     ```bash
     cd 1_data/preprocess/nlp
     python text_clean.py
     ```
   - Vision 전처리:
     ```bash
     cd 1_data/preprocess/vision
     python easyocr_filter.py
     ```

3. **모델 학습**

   - SikuRoBERTa 학습:
     ```bash
     cd 3_model/nlp/sikuroberta/train
     python train_task.py
     ```
   - Swin Transformer 학습:
     ```bash
     cd 3_model/vision/swin_experiment/train
     python train.py
     ```

4. **모델 평가**

   - SikuRoBERTa 평가:
     ```bash
     cd 3_model/nlp/sikuroberta/evaluation
     python evaluate_task.py
     ```
   - Swin Transformer 평가:
     ```bash
     cd 3_model/vision/swin_experiment/evaluation
     python evaluate.py
     ```

파이프라인 실행 후 결과는 `outputs` 폴더에 저장되며, `main.py`는 기본적으로 `summary.txt`를 생성합니다. 산출물 생성 함수(`generate_deliverables`)를 수정하여 모델 성능 지표나 보고서를 추가할 수 있습니다.

## 연관 저장소

- **Epitext_Service**: 프로덕션 서비스 코드(백엔드/프론트엔드)를 포함하는 저장소입니다. 모델 서빙, API, 웹 UI 등 서비스 기능은 해당 저장소에서 관리합니다.

## 모델 공유

학습된 모델을 서비스 저장소와 공유할 때는 다음과 같은 방법을 사용할 수 있습니다.

- **Git LFS**: 소규모 모델 파일을 버전 관리하는 데 사용합니다.
- **클라우드 스토리지**: 대용량 모델을 저장하고 배포합니다.
- **모델 레지스트리**: 모델 버전을 체계적으로 관리할 때 활용합니다.

## 라이센스

이 프로젝트의 코드는 연구 목적으로 사용됩니다. 라이센스 조건은 `5_docs`에 포함된 라이센스 파일을 참고하세요.

## 문의

프로젝트에 대한 질문이나 피드백은 저장소의 이슈나 PR을 통해 알려주세요.

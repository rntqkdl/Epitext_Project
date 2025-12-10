
# Epitext Project: 한자 탁본 자동 복원 AI 시스템



Epitext Project는 고문서(한자 탁본)의 이미지를 분석하여 훼손된 글자를 복원하고, 번역 및 디지털화하는 AI 통합 솔루션입니다.



![Project Status](https://img.shields.io/badge/Status-Active-success)

![Python](https://img.shields.io/badge/Python-3.8+-blue)

![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20FastAPI%20%7C%20React-orange)



---



## 프로젝트 구성 (Project Structure)



이 저장소는 데이터 수집부터 모델링, 서비스 배포까지 전체 파이프라인을 포함합니다.



| 모듈 | 경로 | 설명 | 상태 |

| :--- | :--- | :--- | :---: |

| **Data Pipeline** | [`1_data/`](./1_data) | 크롤링, PDF 파싱, 전처리(NLP/Vision), EDA | 완료 |

| **Notebooks** | `2_notebooks/` | 실험 및 프로토타이핑용 Jupyter Notebook | 진행중 |

| **AI Models** | [`3_model/`](./3_model) | Swin Transformer(OCR), Gemini(번역) 학습 및 검증 | 완료 |

| **Backend** | `backend/` | 모델 서빙 API (FastAPI) 및 데이터베이스 연동 | 통합됨 |

| **Frontend** | `frontend/` | 사용자 인터페이스 (React) | 통합됨 |



---



## 시작하기 (Getting Started)



각 모듈은 독립적으로 실행 가능하도록 설계되었습니다. 상세한 실행 방법은 각 폴더 내부의 README.md를 참고하세요.



### 1. 데이터 파이프라인 (Data)

데이터 수집 및 전처리를 수행합니다.

```bash

cd 1_data

pip install -r requirements.txt

python config.py  # 디렉토리 초기화

2. 모델 학습 및 검증 (Model)

Swin Transformer 모델을 학습하거나 성능을 평가합니다.



Bash



# Vision (OCR)cd 3_model/vision/swin_experiment

python train.py      # 학습

python evaluation/evaluate.py  # 검증

3. 전체 시스템 실행 (Web Service)

백엔드와 프론트엔드를 구동합니다.



Bash



# Backendcd backend

python main.py# Frontendcd frontend

npm install

npm start

시스템 아키텍처

수집 (Crawler/PDF): 규장각, 국사편찬위 등에서 이미지/텍스트 수집

전처리 (Preprocess): EasyOCR 필터링 및 텍스트 노이즈 제거

인식 (OCR): Swin Transformer V2 기반 한자 인식

복원/번역 (NLP): LLM (Gemini/RoBERTa) 기반 문맥 복원 및 번역

서비스: React 웹 인터페이스를 통한 검색 및 시각화

라이선스 및 출처

데이터 출처: 서울대학교 규장각, 국사편찬위원회, 국립문화재연구소

개발: 4조 복원왕 김탁본 팀

본 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

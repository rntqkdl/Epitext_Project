# 1_data - 데이터 수집 및 전처리

## 개요
탁본 한문 데이터 수집, 전처리, EDA를 위한 코드 모음

## 폴더 구조
1_data/
├── config.py # 통합 설정
├── crawlers/ # 데이터 수집
│ ├── base_crawler.py
│ ├── 01_crawling_kyu.py # 규장각
│ ├── 01_crawling_historydb.py # 국사편찬위
│ └── 01_crawling_nrich.py # 문화재연구소
├── utils/ # 공통 유틸리티
│ ├── db_manager.py
│ ├── file_handler.py
│ └── retry_handler.py
├── preprocess/nlp/ # 자연어 전처리
│ ├── 01_text_preprocess_github.py
│ └── 02_text_preprocess_colab.py
├── eda/ # 데이터 분석
│ ├── text_eda.py
│ ├── image_quality_eda.py
│ └── easyocr_filter_images.py
├── translation/ # 번역
│ └── gemini_translation.py
└── raw_data/ # 원본 데이터 (Git 제외)

text

## 실행 방법

### 1. 환경 설정
pip install -r requirements.txt

text

### 2. 데이터 수집
cd 1_data/crawlers
python 01_crawling_kyu.py
python 01_crawling_historydb.py
python 01_crawling_nrich.py

text

### 3. 전처리
cd 1_data/preprocess/nlp
python 01_text_preprocess_github.py

text

## 주요 기능
- **데이터 수집**: 규장각, 국사편찬위, 문화재연구소 크롤링
- **전처리**: 노이즈 제거, 구두점 복원, 문장 분할
- **번역**: Gemini API 기반 한문 번역

## 출처
- 크롤링 코드: 팀 자체 작성 (1주차 보고서)
- 전처리 코드: 팀 자체 작성 (4주차 보고서)
- 프로젝트: 4조 복원왕 김탁본 (2025)

## 관련 리포지토리
- https://github.com/jae2022/Epitext_Back
- https://github.com/jae2022/Epitext_Front

---
**작성일**: 2025-12-07

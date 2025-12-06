# 1_data - 탁본 한문 데이터 처리 파이프라인

## 프로젝트 개요

한국학중앙연구원 탁본 이미지 및 판독문 데이터를 수집, 전처리, 분석하는 코드 모음

**주요 기능**
- 웹 크롤링을 통한 데이터 수집
- 자연어 전처리 (노이즈 제거, 필터링)
- 이미지 전처리 (EasyOCR 기반 필터링)
- 탐색적 데이터 분석(EDA)

---

## 폴더 구조

## 폴더 구조

    1_data/
    ├── config.py
    ├── crawlers/
    │   ├── __init__.py
    │   └── klc_crawler.py
    ├── utils/
    │   ├── __init__.py
    │   ├── file_handler.py
    │   └── logger.py
    ├── preprocess/
    │   ├── nlp/
    │   │   ├── __init__.py
    │   │   └── 01_text_clean.py
    │   └── vision/
    │       ├── __init__.py
    │       └── 01_easyocr_filter.py
    ├── eda/
    │   ├── nlp/
    │   │   ├── __init__.py
    │   │   └── 01_text_stats.py
    │   └── vision/
    │       ├── __init__.py
    │       └── 01_quality_analysis.py
    ├── translation/
    ├── raw_data/
    └── README.md


---

## 환경 설정

### 1. 저장소 클론

git clone <repository-url>
cd 1_data


### 2. 가상환경 생성 (권장)

**Windows**
python -m venv venv
venv\Scripts\activate


**Mac/Linux**
python -m venv venv
source venv/bin/activate


### 3. 필요 라이브러리 설치

pip install -r requirements.txt


---

## 데이터 준비

### Option A: 크롤러로 직접 수집

cd crawlers
python klc_crawler.py


결과: `raw_data/` 폴더에 이미지 및 CSV 저장

### Option B: 기존 데이터 사용

`raw_data/` 폴더에 다음 파일들을 준비:
- `doc_id_transcript_dataset.csv` - 판독문 데이터
- `images/` - 원본 탁본 이미지 폴더

---

## 실행 방법

### 1. 자연어 전처리

cd preprocess/nlp
python 01_text_clean.py


**입력**
- `raw_data/doc_id_transcript_dataset.csv`

**출력**
- `raw_data/doc_id_transcript_dataset_processed.csv`

**처리 내용**
- 특수문자 제거 (한글, 숫자, 기호 등)
- 노이즈 키워드 필터링 (譯註, 韓國金石 등)
- 판독불가 기호 통일 (▦, ▧ → ▨)
- 20자 미만 문장 제거

---

### 2. 이미지 전처리

cd preprocess/vision
python 01_easyocr_filter.py


**입력**
- `raw_data/images/`

**출력**
- `raw_data/filtered_takbon/`
- `raw_data/filter_log.csv`

**처리 내용**
- EasyOCR로 텍스트 검출
- 글자가 있는 이미지만 선별
- 로그 기반 중복 방지

**주의사항**
- GPU 사용 권장 (CPU는 매우 느림)
- 이미 처리된 파일은 로그를 참조하여 건너뜀

---

### 3. 텍스트 통계 분석

cd eda/nlp
python 01_text_stats.py


**입력**
- `raw_data/doc_id_split_sentences.csv`

**출력**
- `raw_data/vocab.csv` - 글자 빈도표
- 콘솔 출력: 문장 길이 통계

**분석 내용**
- 문장 길이 통계 (평균, 중위수, 분포)
- 글자 빈도 Top 20
- 전체 Vocab 추출

---

### 4. 이미지 품질 분석

cd eda/vision
python 01_quality_analysis.py


**입력**
- `raw_data/image_quality_metrics.csv`
- `raw_data/filtered_takbon/`

**출력**
- `raw_data/low_quality_removed/` - 저품질 이미지 이동
- 시각화: 품질 지표 분포, 상관관계 히트맵

**분석 내용**
- 7가지 품질 지표 통계
- IQR 기반 이상치 탐지
- 저품질 이미지 자동 제거 (threshold=2)

---

## 설정 파일 수정

`config.py`에서 경로 및 파라미터 수정 가능:

데이터 경로
RAW_DATA_DIR = Path(file).parent / "raw_data"

전처리 파라미터
MIN_SENTENCE_LENGTH = 20
EASYOCR_LANGS = ["ch_tra"]

EDA 파라미터
BAD_INDICATOR_THRESHOLD = 2


---

## 문제 해결

### FileNotFoundError 발생 시

**원인**: 실행 위치가 잘못됨

**해결**: 스크립트가 있는 폴더로 이동
cd 1_data/preprocess/nlp
python 01_text_clean.py


또는 코드에서 절대경로 사용:
from pathlib import Path
INPUT_CSV = Path(file).parent.parent.parent / "raw_data" / "파일명.csv"


---

### EasyOCR 설치 오류

**CUDA 지원 torch 재설치 (GPU 사용 시)**
pip uninstall torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118


**CPU 버전**
pip install torch torchvision


---

### 한글 깨짐 (Windows)

파일 저장 시 encoding 명시:
df.to_csv("output.csv", encoding="utf-8-sig")


---

## 데이터 형식

### 입력 CSV 예시

**doc_id_transcript_dataset.csv**
doc_id,transcript
gsko_001_0001,"大明萬曆四十三年乙卯..."
gsko_001_0002,"崇禎紀元後..."


**doc_id_split_sentences.csv**
doc_id,sentence
gsko_001_0001,大明萬曆四十三年乙卯
gsko_001_0001,王子生員李氏


---

## 출처 및 라이센스

**프로젝트**: 4조 복원왕 김탁본  
**출처**: 4주차 보고서 (2025)  
**작성일**: 2025-12-07  
**라이센스**: MIT License

---

## 참고 자료

- 한국학중앙연구원 한국금석문: http://gsm.nricp.go.kr/
- EasyOCR Documentation: https://github.com/JaidedAI/EasyOCR
- Pandas Documentation: https://pandas.pydata.org/

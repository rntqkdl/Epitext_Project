# PDF 탁본 데이터 추출 모듈 (PDF Extraction Module)

이 모듈은 한국학중앙연구원 등의 탁본 조사 보고서(PDF) 형식을 분석하여 **탁본 이미지**와 **메타데이터**, **한자 원문**을 자동으로 추출하고 매칭합니다.

## 1. 주요 기능
* **항목 자동 분할**: "01", "02" 등 번호를 기준으로 탁본 항목을 자동 식별
* **이미지-캡션 매칭**: 이미지 하단의 "앞면", "뒷면" 텍스트를 인식하여 이미지와 연결
* **한자 원문 추출**: 세로쓰기(우측→좌측)로 된 한자 원문 페이지를 찾아 텍스트 추출
* **데이터 구조화**: 이미지 경로, 시대, 연도, 원문 등을 CSV 및 JSON으로 저장

## 2. 폴더 구조
```text
pdf_processing/
├── config.py          # 설정 (폰트 크기, 정규식 패턴 등)
├── models.py          # 데이터 구조 (TextLine, EntryBundle 등)
├── utils.py           # 공통 유틸리티 (한글 판별, 문자열 정규화)
├── pdf_parser.py      # PDF 구조 분석 및 항목 분할 로직
├── image_processor.py # 이미지 추출 및 면 병합 로직
├── text_extractor.py  # 한자 원문 추출 로직
├── main.py            # 실행 진입점 (CLI 포함)
└── README.md          # 설명서

3. 사용 방법
사전 요구 사항
Python 3.8+
PyMuPDF 설치 필요

pip install pymupdf
실행 방법 (CLI)
터미널에서 main.py를 실행하며 입력/출력 경로를 지정합니다.

# 기본 사용법
cd 1_data/crawlers/pdf_processing
python main.py --input "PDF파일_경로" --output "저장할_경로"

# 예시 (Windows)
python main.py --input "C:\data\pdfs" --output "C:\data\results"

4. 출력 결과
지정된 출력 폴더(--output)에 다음과 같이 저장됩니다.

images/{PDF명}/: 추출된 탁본 이미지 파일들 (PNG)
extracted_takbon_data.csv: 전체 데이터 정리 (엑셀 호환)
extracted_takbon_data.json: 전체 데이터 정리 (JSON 형식)

5. 커스텀 설정
config.py 파일을 수정하여 다음 내용을 변경할 수 있습니다.
FONT_SIZE_MIN/MAX: 원문 텍스트로 인식할 폰트 크기 범위
FACE_PATTERNS: "앞면", "뒷면" 등을 인식하는 정규표현식 패턴 
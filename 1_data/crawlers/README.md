# 데이터 수집 및 처리 모듈 (Data Collection & Processing)
이 모듈은 탁본 복원 AI 프로젝트를 위한 데이터 파이프라인의 핵심 구성 요소입니다. 웹 크롤링을 통해 원본 데이터를 수집하거나, 기존 탁본 조사 보고서(PDF)에서 정형화된 데이터를 추출하는 기능을 제공합니다.

## 1. 폴더 구조
```text
1_data/
├── crawlers/                  # [Web] 웹 크롤러 소스 코드
│   ├── base_crawler.py        # 크롤러 공통 클래스
│   ├── 01_crawling_historydb.py # 국사편찬위원회
│   ├── 01_crawling_nrich.py     # 문화재연구소
│   └── 01_crawling_kyu.py       # 규장각 (Selenium)
│
├── pdf_processing/            # [PDF] PDF 데이터 추출 모듈
│   ├── config.py              # PDF 처리 설정
│   ├── pdf_parser.py          # 구조 분석 및 항목 분할
│   ├── image_processor.py     # 이미지 추출 및 병합
│   ├── text_extractor.py      # 한자 원문 추출
│   └── main.py                # PDF 처리 실행 진입점
│
├── utils/                     # 공통 유틸리티 (DB, 파일, 네트워크)
├── config.py                  # 프로젝트 통합 설정
├── requirements.txt           # 의존성 라이브러리 목록
└── raw_data/                  # 수집된 데이터 저장소 (Git 제외)

## 2. 설치 및 환경 설정

### 가상환경 및 라이브러리 설치
Python 3.8 이상 환경이 필요합니다.

```bash
cd 1_data
pip install -r requirements.txt
필수 라이브러리 안내
Web Crawling: requests, beautifulsoup4, selenium (규장각용)

PDF Processing: pymupdf (fitz)

디렉토리 초기화
데이터 저장을 위한 폴더와 데이터베이스를 초기화합니다.

python config.py
3. 기능 1: 웹 크롤링 (Web Crawling)
주요 웹사이트에서 탁본 이미지와 메타데이터를 수집합니다.

실행 방법
각 크롤러는 독립적으로 실행 가능합니다.

A. 국사편찬위원회 (HistoryDB) 빠른 속도로 데이터를 수집하며, 이미지를 프록시 서버를 통해 다운로드합니다.
python crawlers/01_crawling_historydb.py

B. 문화재연구소 (NRICH) 스레드 풀(ThreadPool)을 사용하여 대량의 데이터를 병렬로 수집합니다.
python crawlers/01_crawling_nrich.py

C. 규장각 한국학연구원 (Kyujanggak) Selenium을 사용하여 동적 페이지를 제어합니다. 실행 전 Chrome 브라우저가 설치되어 있어야 합니다.
python crawlers/01_crawling_kyu.py

데이터베이스 구조 (unified_metadata.db)
수집된 데이터는 SQLite DB에 저장됩니다.
documents: 문서 기본 정보 (제목, 시대, 서체 등)
images: 이미지 파일 경로 및 메타정보
texts: 원문/판독문/번역문 텍스트

4. 기능 2: PDF 데이터 추출 (PDF Processing)
한국학중앙연구원 등의 탁본 조사 보고서(PDF) 형식을 분석하여 구조화된 데이터로 변환합니다.

주요 기능
항목 자동 분할: "01", "02" 등 번호를 기준으로 탁본 항목을 자동 식별
이미지-캡션 매칭: 이미지 하단의 "앞면", "뒷면" 텍스트를 인식하여 이미지와 연결
한자 원문 추출: 세로쓰기(우측->좌측)로 된 한자 원문 페이지를 찾아 텍스트 추출
데이터 구조화: 이미지 경로, 시대, 연도, 원문 등을 CSV 및 JSON으로 저장

실행 방법
PDF 파일이 있는 폴더와 결과 저장 폴더를 지정하여 실행합니다.

기본 실행
# pdf_processing 폴더 내부의 main.py 실행
python pdf_processing/main.py --input "PDF파일_경로" --output "저장할_경로"

실행 예시 (Windows)
python pdf_processing/main.py --input "C:\data\pdfs" --output "C:\data\results"

출력 결과
지정된 출력 폴더(--output)에 다음과 같이 저장됩니다.

images/{PDF명}/: 추출된 탁본 이미지 파일들 (PNG)
extracted_takbon_data.csv: 전체 데이터 정리 (엑셀 호환)
extracted_takbon_data.json: 전체 데이터 정리 (JSON 형식)

PDF 처리 설정 변경
pdf_processing/config.py 파일에서 다음 항목을 수정할 수 있습니다.
FONT_SIZE_MIN/MAX: 원문 텍스트로 인식할 폰트 크기 범위
FACE_PATTERNS: "앞면", "뒷면" 등을 인식하는 정규표현식 패턴

5. 주의사항
데이터 용량: raw_data/ 폴더에는 대용량 이미지 파일이 저장되므로 Git 커밋 시 제외됩니다. (.gitignore 적용됨)
네트워크: 크롤링 시 과도한 요청으로 차단되지 않도록 config.py의 delay 설정을 준수하십시오.
라이선스: 수집된 데이터의 저작권은 각 원본 소유 기관에 있습니다.
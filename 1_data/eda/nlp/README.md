# NLP EDA 모듈 (Text Statistical Analysis)
한자 탁본 데이터셋의 텍스트 특성을 분석하고 통계를 추출하는 모듈입니다.

## 1. 주요 기능
* **문장 길이 통계**: 평균 길이, 최대/최소 길이, 표준편차 등 분포 확인
* **어휘(Vocab) 분석**: 전체 텍스트에서 등장하는 글자(Character) 빈도수 분석
* **노이즈 필터링**: 공백, 구두점 등 불필요한 문자를 제외하고 순수 한자/한글 통계 산출

## 2. 파일 구성
nlp/
├── __init__.py      # 패키지 초기화
├── config.py        # 설정 파일 (입출력 경로, 파라미터)
├── text_stats.py    # 메인 분석 스크립트
└── README.md        # 설명서

## 3. 사용 방법
설정 변경
config.py 파일에서 입력 데이터 경로와 분석 옵션을 수정할 수 있습니다.
INPUT_CSV: 분석할 원본 데이터 파일 경로
COLUMN_NAME: 분석할 텍스트 컬럼 이름 (예: transcript, sentence)

실행
터미널에서 아래 명령어로 실행합니다.

cd 1_data/eda/nlp
python text_stats.py

## 4. 출력 결과
콘솔 출력: 데이터 기초 통계 및 상위 빈도 글자 리스트
파일 저장: raw_data/eda_results/vocab.csv (글자별 빈도수 전체 목록)

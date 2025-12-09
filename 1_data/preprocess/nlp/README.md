
# NLP 전처리 모듈 (Text Preprocessing)



한자 탁본 데이터셋의 텍스트에서 노이즈를 제거하고, 정규화하며, 유효한 길이의 데이터만 필터링하는 모듈입니다.



## 1. 주요 기능

* **노이즈 제거**: "번역문", "조사보고" 등 불필요한 키워드 삭제

* **특수문자 정제**: 한글, 특수기호, 숫자 등을 제거하고 한자 및 판독 불가 기호(▨)만 보존

* **형식 통일**: 줄바꿈을 제거하고 한 줄(Flatten) 텍스트로 변환

* **길이 필터링**: 너무 짧은(예: 20자 미만) 무의미한 데이터 제외



## 2. 파일 구성

```text

nlp/

├── __init__.py      # 패키지 초기화

├── config.py        # 설정 파일 (경로, 노이즈 패턴)

├── text_clean.py    # 메인 전처리 스크립트

└── README.md        # 설명서

3. 사용 방법

설정 변경

config.py 파일에서 다음 항목을 수정할 수 있습니다.



INPUT_CSV: 원본 데이터 파일 경로

MIN_LENGTH: 필터링할 최소 문자 길이 (기본값: 20)

NOISE_KEYWORDS: 제거할 노이즈 단어 리스트 추가

실행

터미널에서 아래 명령어로 실행합니다.







cd 1_data/preprocess/nlp

python text_clean.py

4. 출력 결과

저장 파일: doc_id_transcript_dataset_processed.csv (전처리 완료 데이터)

생성 컬럼: preprocess (정제된 텍스트가 담긴 컬럼)

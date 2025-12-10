
# Vision 전처리 모듈 (Image Filtering)



EasyOCR을 사용하여 텍스트가 포함된 유의미한 탁본 이미지만 선별하는 모듈입니다.



## 1. 주요 기능

* **텍스트 감지**: EasyOCR을 활용해 이미지 내 글자 존재 여부 확인

* **자동 필터링**: 글자가 없는(무의미한) 이미지는 제외하고, 유의미한 이미지만 별도 폴더에 복사

* **로그 관리**: 처리 상태(KEEP/SKIP/ERROR)를 CSV에 기록하여 중단 후 재실행 시 중복 방지

* **GPU 가속**: CUDA 사용 가능 시 자동으로 GPU 모드로 동작하여 속도 향상



## 2. 파일 구성

```text

vision/

├── __init__.py        # 패키지 초기화

├── config.py          # 설정 파일 (경로, 언어 설정)

├── easyocr_filter.py  # 메인 필터링 스크립트

└── README.md          # 설명서

3. 사용 방법

설정 변경

config.py 파일에서 다음 항목을 수정할 수 있습니다.



SRC_DIR: 원본 이미지가 저장된 폴더 경로

LANGUAGES: OCR 인식 언어 (기본값: ["ch_tra"] - 번체 한자)

실행

터미널에서 아래 명령어로 실행합니다.



Bash



cd 1_data/preprocess/vision

python easyocr_filter.py

4. 출력 결과

결과 폴더: raw_data/filtered_takbon/ (선별된 이미지 복사됨)

로그 파일: raw_data/filter_log.csv (처리 결과 기록)

5. 주의사항

GPU 권장: CPU로 실행 시 속도가 매우 느릴 수 있습니다. CUDA 환경을 권장합니다.

대용량 처리: 처리할 이미지가 많을 경우 시간이 오래 걸리므로, 로그 파일을 통해 진행 상황을 저장합니다.

# Vision EDA 모듈 (Image Quality Analysis)

한자 탁본 이미지의 품질 지표를 분석하고, 품질이 낮은 이미지를 자동으로 식별하여 격리하는 모듈입니다.

## 1. 주요 기능

* **품질 지표 분석**: 밝기, 명암비, 블러(Blur), 노이즈 등 7가지 지표 분석
* **이상치 탐지**: IQR(Interquartile Range) 방식을 사용하여 지표별 이상치 식별
* **자동 필터링**: 나쁜 지표가 일정 개수(Threshold) 이상인 이미지를 별도 폴더로 이동

## 2. 파일 구성

```text
vision/
├── __init__.py          # 패키지 초기화
├── config.py            # 설정 파일 (경로, 파라미터)
├── quality_analysis.py  # 메인 분석 스크립트
└── README.md            # 설명서
3. 분석 대상 지표
illumination_variance: 조명 불균형 (낮을수록 좋음)

global_contrast: 전역 명암비 (높을수록 좋음)

local_contrast: 국소 명암비 (높을수록 좋음)

blur_score: 흐림 정도 (낮을수록 좋음)

smear_noise_ratio: 번짐 노이즈 비율 (낮을수록 좋음)

deterioration_mask_ratio: 훼손 영역 비율 (낮을수록 좋음)

bleed_through_likelihood: 뒷면 비침 가능성 (낮을수록 좋음)

4. 사용 방법
설정 변경
config.py 파일에서 다음 항목을 수정할 수 있습니다.

SRC_DIR: 분석할 원본 이미지 폴더

CSV_PATH: 이미지 품질 지표가 담긴 CSV 파일 경로

BAD_INDICATOR_THRESHOLD: 이미지를 제거할 나쁜 지표의 최소 개수 (기본값: 2)

실행
터미널에서 아래 명령어로 실행합니다.

Bash

cd 1_data/eda/vision
python quality_analysis.py
5. 결과 확인
콘솔: 지표별 통계 및 제거 대상 개수 출력

폴더: 저품질 이미지가 raw_data/low_quality_removed 폴더로 이동됨
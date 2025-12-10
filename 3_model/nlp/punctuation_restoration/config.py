# -*- coding: utf-8 -*-
'''구두점 복원 모델 설정 파일.

이 모듈은 구두점 복원에 필요한 경로와 모델 설정을 정의합니다. 경로는
프로젝트 루트 기준으로 상대적으로 설정되며, 필요시 수정하여 사용할
수 있습니다.
'''

from pathlib import Path

# 프로젝트 루트 경로 계산 (현재 파일에서 세 단계 위가 프로젝트 루트)
BASE_DIR = Path(__file__).resolve().parents[3]

# 입력 CSV: 구두점이 제거된 텍스트가 저장된 파일
INPUT_CSV = BASE_DIR / "1_data" / "raw_data" / "doc_id_transcript_dataset_processed.csv"

# 출력 CSV: 구두점 복원 결과를 저장할 파일
OUTPUT_CSV = BASE_DIR / "1_data" / "raw_data" / "doc_id_transcript_with_punctuation.csv"

# Hugging Face 모델 태그 (SikuRoBERTa 기반)
MODEL_TAG = "seyoungsong/SikuRoBERTa-PUNC-AJD-KLC"

# 모델 캐시 디렉터리 (다운로드된 모델을 저장)
MODEL_CACHE_DIR = BASE_DIR / "models" / "punctuation"

# 최대 시퀀스 길이
MAX_LENGTH = 512

# 슬라이딩 윈도우 크기 및 오버랩 크기 (문자 단위)
WINDOW_SIZE_CHARS = 400
OVERLAP_CHARS = 100

# 배치 크기 (진행 상태 표시용)
DISPLAY_BATCH_SIZE = 32

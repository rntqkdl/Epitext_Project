# -*- coding: utf-8 -*-
"""구두점 복원 모델 설정 파일.

경로는 :func:`project_paths.find_project_root`를 통해 동적으로 계산되어
어느 위치에서 실행해도 올바르게 동작합니다.
"""

from pathlib import Path

from project_paths import find_project_root

PROJECT_ROOT = find_project_root(Path(__file__))

# 입력 CSV: 구두점이 제거된 텍스트가 저장된 파일
INPUT_CSV = PROJECT_ROOT / "1_data" / "raw_data" / "doc_id_transcript_dataset_processed.csv"

# 출력 CSV: 구두점 복원 결과를 저장할 파일
OUTPUT_CSV = PROJECT_ROOT / "1_data" / "raw_data" / "doc_id_transcript_with_punctuation.csv"

# Hugging Face 모델 태그 (SikuRoBERTa 기반)
MODEL_TAG = "seyoungsong/SikuRoBERTa-PUNC-AJD-KLC"

# 모델 캐시 디렉터리 (다운로드된 모델을 저장)
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "punctuation"

# 최대 시퀀스 길이
MAX_LENGTH = 512

# 슬라이딩 윈도우 크기 및 오버랩 크기 (문자 단위)
WINDOW_SIZE_CHARS = 400
OVERLAP_CHARS = 100

# 배치 크기 (진행 상태 표시용)
DISPLAY_BATCH_SIZE = 32

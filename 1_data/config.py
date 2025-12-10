"""
데이터 파이프라인 설정.

``python 1_data/main.py`` 또는 ``python main.py --phase data`` 실행 시
파이프라인의 각 단계를 활성화하거나 비활성화하는 플래그를 정의합니다.
경로 상수는 ``1_data.__init__``에서 가져옵니다.
"""

from . import RAW_DIR, PROCESSED_DIR, EDA_DIR

# 실행 단계 제어 플래그 (``--step=all`` 옵션일 때 적용)
RUN_CRAWL = True
RUN_PREPROCESS = True
RUN_EDA = True

# 단계별 입출력 경로 설정
CRAWL_OUTPUT_DIR = RAW_DIR
PREPROCESS_INPUT_DIR = RAW_DIR
PREPROCESS_OUTPUT_DIR = PROCESSED_DIR
EDA_INPUT_DIR = PROCESSED_DIR
EDA_REPORT_DIR = EDA_DIR

__all__ = [
    "RUN_CRAWL",
    "RUN_PREPROCESS",
    "RUN_EDA",
    "CRAWL_OUTPUT_DIR",
    "PREPROCESS_INPUT_DIR",
    "PREPROCESS_OUTPUT_DIR",
    "EDA_INPUT_DIR",
    "EDA_REPORT_DIR",
]
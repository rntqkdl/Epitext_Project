from pathlib import Path

# 1_data 경로 기준으로 루트/데이터 경로 계산
ROOT_DIR = Path(__file__).resolve().parents[1]  # Epitext_Project
DATA_DIR = ROOT_DIR / "1_data"
RAW_DIR = DATA_DIR / "sample_data"
PROCESSED_DIR = DATA_DIR / "processed"
EDA_DIR = DATA_DIR / "eda_outputs"

# 실행 단계 플래그
RUN_CRAWL = True
RUN_PREPROCESS = True
RUN_EDA = True

# 크롤링 설정
CRAWL_OUTPUT_DIR = RAW_DIR

# 전처리 설정
PREPROCESS_INPUT_DIR = RAW_DIR
PREPROCESS_OUTPUT_DIR = PROCESSED_DIR

# EDA 설정
EDA_INPUT_DIR = PROCESSED_DIR
EDA_REPORT_DIR = EDA_DIR

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]  # Epitext_Project
MODEL_DIR = ROOT_DIR / "3_model"
SAVED_MODELS_DIR = MODEL_DIR / "saved_models"

# 공통 설정
RANDOM_SEED = 42
DEVICE = "cuda"  # 또는 "cpu"

# 모델별 저장 경로
SIKUROBERTA_SAVE_DIR = SAVED_MODELS_DIR / "sikuroberta"
SWIN_SAVE_DIR = SAVED_MODELS_DIR / "swin"
OCR_SAVE_DIR = SAVED_MODELS_DIR / "ocr"
GEMINI_SAVE_DIR = SAVED_MODELS_DIR / "gemini"

# 데이터 경로
DATA_DIR = ROOT_DIR / "1_data" / "processed"

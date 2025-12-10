from pathlib import Path

# 프로젝트 루트 경로
ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT_DIR / "3_model"
SAVED_MODELS_DIR = MODEL_DIR / "saved_models"

__all__ = ["ROOT_DIR", "MODEL_DIR", "SAVED_MODELS_DIR"]

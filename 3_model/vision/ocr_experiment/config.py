"""
OCR Evaluation Configuration
======================================================================
목적: OCR 성능 평가를 위한 파일 경로 및 설정 관리
======================================================================
"""
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent
    DEFAULT_GT_PATH = BASE_DIR / "sample_data" / "gt.txt"
    DEFAULT_PRED_PATH = BASE_DIR / "sample_data" / "prediction.json"
    VERBOSE = True

    @staticmethod
    def print_config():
        print("=" * 60)
        print("OCR Evaluation Configuration")
        print("=" * 60)
        print(f"Default GT Path:   {Config.DEFAULT_GT_PATH}")
        print(f"Default Pred Path: {Config.DEFAULT_PRED_PATH}")
        print("=" * 60)
"""
Evaluation Configuration
======================================================================
목적: 모델 검증에 필요한 경로 및 설정 관리
======================================================================
"""
import os
from pathlib import Path

class EvalConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
    MODEL_PATH = PROJECT_ROOT / "3_model" / "saved_models" / "sikuroberta" / "final"
    TEST_DATA_PATH = PROJECT_ROOT / "1_data" / "raw_data" / "split_dataset"
    RESULT_FILE = PROJECT_ROOT / "3_model" / "logs" / "sikuroberta" / "eval_results.txt"
    
    BATCH_SIZE = 32
    MLM_PROBABILITY = 0.15
    
    @staticmethod
    def print_config():
        print("======================================================")
        print(" Evaluation Configuration")
        print("======================================================")
        print(f" Model Path: {EvalConfig.MODEL_PATH}")
        print(f" Data Path:  {EvalConfig.TEST_DATA_PATH}")
        print("======================================================")
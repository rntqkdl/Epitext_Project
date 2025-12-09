"""
Evaluation Configuration
======================================================================
목적: 모델 검증에 필요한 경로 및 설정 관리
작성자: Epitext Project Team
======================================================================
"""

import os
from pathlib import Path

class EvalConfig:
    """검증 전용 설정 클래스"""
    
    # ==================================================================
    # 1. 경로 설정 (자동 감지)
    # ==================================================================
    # 현재 파일: 3_model/nlp/sikuroberta/evaluation/config.py
    # 루트 경로: Epitext_Project/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
    
    # 평가할 모델 경로 (학습 완료된 최종 모델)
    # 기본값: saved_models/sikuroberta/final
    MODEL_PATH = PROJECT_ROOT / "3_model" / "saved_models" / "sikuroberta" / "final"
    
    # 테스트 데이터셋 경로 (학습 시 분할된 데이터)
    # 1_data/raw_data/split_dataset 폴더를 바라봄
    TEST_DATA_PATH = PROJECT_ROOT / "1_data" / "raw_data" / "split_dataset"
    
    # 결과 저장 경로
    RESULT_FILE = PROJECT_ROOT / "3_model" / "logs" / "sikuroberta" / "eval_results.txt"
    
    # ==================================================================
    # 2. 평가 파라미터
    # ==================================================================
    BATCH_SIZE = 32
    
    # 평가 시 마스킹 확률 (동적 마스킹을 통한 평가 시 필요)
    MLM_PROBABILITY = 0.15
    
    @staticmethod
    def print_config():
        """설정 정보 출력"""
        print("======================================================")
        print(" Evaluation Configuration")
        print("======================================================")
        print(f" Model Path: {EvalConfig.MODEL_PATH}")
        print(f" Data Path:  {EvalConfig.TEST_DATA_PATH}")
        print("======================================================")
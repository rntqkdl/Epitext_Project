"""
Training Configuration
======================================================================
목적: SikuRoBERTa 학습에 필요한 경로 및 하이퍼파라미터 관리
작성자: Epitext Project Team
======================================================================
"""

import os
from pathlib import Path

class TrainConfig:
    """학습 전용 설정 클래스"""
    
    # ==================================================================
    # 1. 경로 설정 (자동 감지)
    # ==================================================================
    # 현재 파일: 3_model/nlp/sikuroberta/train/config.py
    # 루트 경로: Epitext_Project/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
    
    # 입력 데이터 경로 (전처리 완료된 데이터셋)
    # 1_data/raw_data/processed_dataset 폴더를 바라봄
    DATASET_PATH = PROJECT_ROOT / "1_data" / "raw_data" / "processed_dataset"
    
    # 모델 저장 경로
    OUTPUT_DIR = PROJECT_ROOT / "3_model" / "saved_models" / "sikuroberta" / "checkpoints"
    FINAL_MODEL_DIR = PROJECT_ROOT / "3_model" / "saved_models" / "sikuroberta" / "final"
    
    # 로그 경로
    LOG_DIR = PROJECT_ROOT / "3_model" / "logs" / "sikuroberta"
    GRAPH_SAVE_PATH = LOG_DIR / "loss_graph.png"
    
    # ==================================================================
    # 2. 학습 파라미터
    # ==================================================================
    # 베이스 모델 (HuggingFace Hub)
    MODEL_NAME = "klue/roberta-base"
    
    # 하이퍼파라미터
    EPOCHS = 10
    BATCH_SIZE = 16
    GRAD_ACCUM = 4          # Gradient Accumulation Steps
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.06
    
    # Masked Language Model 확률 (15%)
    MLM_PROBABILITY = 0.15
    
    # 기타 설정
    SAVE_LIMIT = 2          # 저장할 체크포인트 최대 개수
    LOGGING_STEPS = 100     # 로그 출력 주기
    EARLY_STOPPING_PATIENCE = 3
    
    @staticmethod
    def print_config():
        """설정 정보 출력"""
        print("======================================================")
        print(" Training Configuration")
        print("======================================================")
        print(f" Data Path:  {TrainConfig.DATASET_PATH}")
        print(f" Output Dir: {TrainConfig.OUTPUT_DIR}")
        print(f" Model:      {TrainConfig.MODEL_NAME}")
        print(f" Epochs:     {TrainConfig.EPOCHS}")
        print(f" Batch Size: {TrainConfig.BATCH_SIZE}")
        print("======================================================")
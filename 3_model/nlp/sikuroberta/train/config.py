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
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
    DATASET_PATH = PROJECT_ROOT / "1_data" / "raw_data" / "processed_dataset"
    OUTPUT_DIR = PROJECT_ROOT / "3_model" / "saved_models" / "sikuroberta" / "checkpoints"
    FINAL_MODEL_DIR = PROJECT_ROOT / "3_model" / "saved_models" / "sikuroberta" / "final"
    LOG_DIR = PROJECT_ROOT / "3_model" / "logs" / "sikuroberta"
    GRAPH_SAVE_PATH = LOG_DIR / "loss_graph.png"
    
    MODEL_NAME = "klue/roberta-base"
    EPOCHS = 10
    BATCH_SIZE = 16
    GRAD_ACCUM = 4
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.06
    MLM_PROBABILITY = 0.15
    SAVE_LIMIT = 2
    LOGGING_STEPS = 100
    EARLY_STOPPING_PATIENCE = 3
    
    @staticmethod
    def print_config():
        print("======================================================")
        print(" Training Configuration")
        print("======================================================")
        print(f" Data Path:  {TrainConfig.DATASET_PATH}")
        print(f" Output Dir: {TrainConfig.OUTPUT_DIR}")
        print(f" Model:      {TrainConfig.MODEL_NAME}")
        print("======================================================")
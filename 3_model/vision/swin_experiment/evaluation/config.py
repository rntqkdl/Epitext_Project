"""
Evaluation Configuration
======================================================================
목적: 검증에 필요한 파일 경로, 모델 파라미터, 시각화 설정 관리
작성자: Epitext Project Team
======================================================================
"""

import os
from pathlib import Path

class Config:
    """전역 설정 클래스"""
    
    # ==================================================================
    # 1. 경로 설정
    # ==================================================================
    # 프로젝트 루트 경로 (evaluation -> swin_experiment -> vision -> 3_model -> Project)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
    
    # 체크포인트 파일 경로 (학습된 모델 가중치)
    # 예: 3_model/saved_models/swinv2_best.pth
    CHECKPOINT_PATH = BASE_DIR / "3_model" / "saved_models" / "swinv2_best.pth"
    
    # 검증용 데이터셋 경로 (NPZ 포맷)
    # 예: 1_data/raw_data/val_shards/val_part001.npz
    NPZ_PATHS = [
        BASE_DIR / "1_data" / "raw_data" / "val_shards" / "val_part001.npz"
    ]
    
    # 결과 저장 경로
    OUTPUT_DIR = BASE_DIR / "3_model" / "vision" / "swin_experiment" / "results"
    
    # ==================================================================
    # 2. 데이터 키 설정
    # ==================================================================
    IMAGES_KEY = "images"
    LABELS_KEY = "labels"
    
    # ==================================================================
    # 3. 모델 및 평가 설정
    # ==================================================================
    MODEL_NAME = "swinv2_small_window16_256"
    IMG_SIZE = 256
    BATCH_SIZE = 500       # 한 번에 로드할 샘플 수
    NUM_VIZ_TOP1 = 2       # 시각화할 샘플 수
    
    @staticmethod
    def print_config():
        """설정 정보 출력"""
        print("======================================================")
        print(" Evaluation Configuration")
        print("======================================================")
        print(f" Checkpoint: {Config.CHECKPOINT_PATH}")
        print(f" Data Paths: {len(Config.NPZ_PATHS)} files")
        print(f" Output Dir: {Config.OUTPUT_DIR}")
        print("======================================================")
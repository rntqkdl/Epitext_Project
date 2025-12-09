"""
Swin Transformer Configuration
======================================================================
목적: 학습에 필요한 데이터 경로, 모델 하이퍼파라미터, 시스템 설정 관리
작성자: Epitext Project Team
작성일: 2025-12-09
======================================================================
"""

import os
from pathlib import Path

class Config:
    """
    학습 실행을 위한 전역 설정 클래스
    
    [경로 설정 가이드]
    기본적으로 프로젝트 루트 내의 data 폴더를 바라보도록 설정되어 있습니다.
    실제 데이터 위치에 맞게 BASE_DIR 또는 하위 경로를 수정하십시오.
    """
    
    # ==================================================================
    # 1. 경로 설정 (Path Configuration)
    # ==================================================================
    # 현재 파일 기준 프로젝트 루트 경로 (3_model/vision/swin_experiment -> Project_Root)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    
    # 데이터 기본 경로 (예: Project_Root/1_data/processed/swin_data)
    # 사용자는 이 경로에 .npz 파일들을 위치시키거나, 이곳을 실제 경로로 수정해야 함
    DATA_BASE_DIR = PROJECT_ROOT / "1_data" / "processed" / "swin_data"
    
    DATA = {
        "base_dir": str(DATA_BASE_DIR),
        "train_dir": "train_shards",         # DATA_BASE_DIR 내부의 폴더명
        "val_dir": "val_shards",             # DATA_BASE_DIR 내부의 폴더명
        "train_pattern": "train_part*.npz",  # 파일 검색 패턴
        "val_pattern": "val_part*.npz",
        
        # 결과 저장 경로
        "output_dir": str(PROJECT_ROOT / "3_model" / "checkpoints" / "swin_final"),
        "tensorboard_dir": str(PROJECT_ROOT / "3_model" / "logs" / "swin_tensorboard"),
        
        # 메타데이터 (문자 매핑)
        "char_mapping_path": str(DATA_BASE_DIR / "char_mapping.json")
    }

    # ==================================================================
    # 2. 모델 설정 (Model Configuration)
    # ==================================================================
    MODEL = {
        "name": "swinv2_small_window16_256",  # timm 모델명
        "img_size": 256,
        "num_classes": 13974,  # 전체 클래스 수 (char_mapping과 일치해야 함)
        "pretrained": True
    }

    # ==================================================================
    # 3. 학습 설정 (Training Hyperparameters)
    # ==================================================================
    TRAINING = {
        "batch_size": 192,
        "grad_accumulation": 3,  # Effective Batch Size = 192 * 3 = 576
        "num_epochs": 50,
        "early_stopping_patience": 5,
        "num_workers": 4,        # CPU 코어 수에 맞춰 조정 (메모리 로드 시 중요)
        
        # Learning Rate (Backbone/Head 분리)
        "lr_backbone": 3e-5,
        "lr_head": 3e-4,
        "weight_decay": 0.01,
        
        # Scheduler
        "warmup_epochs": 5,
        "cosine_t0": 15,
        "cosine_tmult": 2,
        
        # Mixed Precision
        "use_amp": True
    }

    # ==================================================================
    # 4. 기타 설정 (Others)
    # ==================================================================
    CHECKPOINT = {
        "save_every_epoch": True,
        "keep_recent_n": 3
    }

    AUGMENTATION = {
        "color_jitter": {
            "brightness": 0.35,
            "contrast": 0.35,
            "saturation": 0.2,
            "hue": 0.1
        },
        "histogram_equalize_prob": 0.4,
        "random_rotation": 15,
        "random_affine": {
            "degrees": 0,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1)
        }
    }

    LOSS = {
        "use_class_weights": True,
        "class_weight_power": 0.5,  # Class Imbalance 완화 지수
        "min_clip": 1.0
    }

    @staticmethod
    def print_config():
        """현재 설정을 콘솔에 출력"""
        print("=" * 60)
        print("Swin Transformer Configuration")
        print("=" * 60)
        print(f"Base Data Dir: {Config.DATA['base_dir']}")
        print(f"Output Dir:    {Config.DATA['output_dir']}")
        print(f"Model Name:    {Config.MODEL['name']}")
        print(f"Batch Size:    {Config.TRAINING['batch_size']}")
        print(f"Use AMP:       {Config.TRAINING['use_amp']}")
        print("=" * 60)
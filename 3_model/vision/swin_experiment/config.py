"""
Swin Transformer Configuration
======================================================================
목적: 학습에 필요한 데이터 경로, 모델 하이퍼파라미터, 시스템 설정 관리
======================================================================
"""
import os
from pathlib import Path

class Config:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    DATA_BASE_DIR = PROJECT_ROOT / "1_data" / "processed" / "swin_data"
    
    DATA = {
        "base_dir": str(DATA_BASE_DIR),
        "train_dir": "train_shards",
        "val_dir": "val_shards",
        "train_pattern": "train_part*.npz",
        "val_pattern": "val_part*.npz",
        "output_dir": str(PROJECT_ROOT / "3_model" / "saved_models" / "swin_checkpoints"),
        "tensorboard_dir": str(PROJECT_ROOT / "3_model" / "logs" / "swin_tensorboard"),
        "char_mapping_path": str(DATA_BASE_DIR / "char_mapping.json")
    }

    MODEL = {
        "name": "swinv2_small_window16_256",
        "img_size": 256,
        "num_classes": 13974,
        "pretrained": True
    }

    TRAINING = {
        "batch_size": 192,
        "grad_accumulation": 3,
        "num_epochs": 50,
        "early_stopping_patience": 5,
        "num_workers": 4,
        "lr_backbone": 3e-5,
        "lr_head": 3e-4,
        "weight_decay": 0.01,
        "warmup_epochs": 5,
        "cosine_t0": 15,
        "cosine_tmult": 2,
        "use_amp": True
    }
    
    CHECKPOINT = { "save_every_epoch": True, "keep_recent_n": 3 }
    
    AUGMENTATION = {
        "color_jitter": { "brightness": 0.35, "contrast": 0.35, "saturation": 0.2, "hue": 0.1 },
        "histogram_equalize_prob": 0.4,
        "random_rotation": 15,
        "random_affine": { "degrees": 0, "translate": (0.1, 0.1), "scale": (0.9, 1.1) }
    }
    
    LOSS = { "use_class_weights": True, "class_weight_power": 0.5, "min_clip": 1.0 }

    @staticmethod
    def print_config():
        print("=" * 60)
        print("Swin Transformer Configuration")
        print("=" * 60)
        print(f"Base Data Dir: {Config.DATA['base_dir']}")
        print(f"Output Dir:    {Config.DATA['output_dir']}")
        print(f"Model Name:    {Config.MODEL['name']}")
        print("=" * 60)
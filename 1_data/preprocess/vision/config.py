"""
Vision Preprocessing Configuration
======================================================================
목적: 이미지 필터링(EasyOCR)에 필요한 파일 경로 및 파라미터 관리
======================================================================
"""
import os
from pathlib import Path

class Config:
    # 1_data/preprocess/vision/config.py 위치 기준
    # .parent -> vision
    # .parent.parent -> preprocess
    # .parent.parent.parent -> 1_data (여기를 타겟으로!)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # 올바른 경로: 1_data/raw_data/images
    SRC_DIR = BASE_DIR / "raw_data" / "images"
    OUTPUT_DIR = BASE_DIR / "raw_data" / "filtered_takbon"
    LOG_PATH = BASE_DIR / "raw_data" / "filter_log.csv"
    
    LANGUAGES = ["ch_tra"]
    USE_GPU = False # CPU 환경이므로 False로 설정 (CUDA 있으면 True)
    
    @staticmethod
    def print_config():
        print("======================================================")
        print(" Vision Preprocessing Configuration")
        print("======================================================")
        print(f" Source: {Config.SRC_DIR}")
        print(f" Output: {Config.OUTPUT_DIR}")
        print("======================================================")
"""
Vision EDA Configuration
======================================================================
목적: 이미지 품질 분석에 필요한 파일 경로 및 파라미터 관리
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
    # 현재 파일 기준 프로젝트 루트 경로 (1_data/eda/vision -> 1_data)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # 입력 CSV 파일 (품질 지표)
    CSV_PATH = BASE_DIR / "raw_data" / "image_quality_metrics.csv"
    
    # 이미지 폴더
    SRC_DIR = BASE_DIR / "raw_data" / "filtered_takbon"      # 원본 이미지 폴더
    DST_DIR = BASE_DIR / "raw_data" / "low_quality_removed"  # 저품질 이미지 이동 폴더
    
    # ==================================================================
    # 2. 분석 설정
    # ==================================================================
    # 분석 대상 품질 지표
    QUALITY_COLS = [
        "illumination_variance",
        "global_contrast",
        "local_contrast",
        "blur_score",
        "smear_noise_ratio",
        "deterioration_mask_ratio",
        "bleed_through_likelihood",
    ]

    # 값이 높을수록 품질이 나쁜 지표
    BAD_HIGH_COLS = [
        "illumination_variance",
        "blur_score",
        "smear_noise_ratio",
        "deterioration_mask_ratio",
        "bleed_through_likelihood",
    ]

    # 값이 낮을수록 품질이 나쁜 지표
    BAD_LOW_COLS = [
        "global_contrast",
        "local_contrast",
    ]

    # 제거 임계값 (나쁜 지표가 N개 이상이면 제거)
    BAD_INDICATOR_THRESHOLD = 2
    
    @staticmethod
    def print_config():
        """설정 정보 출력"""
        print("======================================================")
        print(" Vision EDA Configuration")
        print("======================================================")
        print(f" CSV Path:  {Config.CSV_PATH}")
        print(f" Source:    {Config.SRC_DIR}")
        print(f" Dest:      {Config.DST_DIR}")
        print(f" Threshold: {Config.BAD_INDICATOR_THRESHOLD}")
        print("======================================================")
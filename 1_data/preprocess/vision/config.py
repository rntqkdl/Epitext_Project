"""
Vision Preprocessing Configuration
======================================================================
목적: 이미지 필터링에 필요한 파일 경로 및 파라미터 관리
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
    # 현재 파일 기준 프로젝트 루트 경로 (1_data/preprocess/vision -> 1_data)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # 원본 이미지 폴더 (raw_data/images)
    SRC_DIR = BASE_DIR / "raw_data" / "images"
    
    # 결과 이미지 폴더 (raw_data/filtered_takbon)
    OUTPUT_DIR = BASE_DIR / "raw_data" / "filtered_takbon"
    
    # 로그 파일 경로
    LOG_PATH = BASE_DIR / "raw_data" / "filter_log.csv"
    
    # ==================================================================
    # 2. EasyOCR 설정
    # ==================================================================
    # 인식 언어 (ch_tra: 번체, ko: 한국어, en: 영어)
    LANGUAGES = ["ch_tra"]
    
    @staticmethod
    def print_config():
        """설정 정보 출력"""
        print("======================================================")
        print(" Vision Preprocessing Configuration")
        print("======================================================")
        print(f" Source: {Config.SRC_DIR}")
        print(f" Output: {Config.OUTPUT_DIR}")
        print(f" Log:    {Config.LOG_PATH}")
        print(f" Langs:  {Config.LANGUAGES}")
        print("======================================================")
"""
Gemini Experiment Configuration
======================================================================
목적: Gemini API 기반 한문 번역 및 평가를 위한 전역 설정 관리
작성자: Epitext Project Team
날짜: 2025-12-09
======================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정 (현재 파일 위치 기준 상위 4단계)
# 3_model/nlp/gemini_experiment/config.py -> Epitext_Project/
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# .env 파일 로드
load_dotenv(BASE_DIR / ".env")


class Config:
    """프로젝트 실행을 위한 전역 설정값 클래스"""
    
    # ------------------------------------------------------------------
    # 기본 설정
    # ------------------------------------------------------------------
    API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # ------------------------------------------------------------------
    # 경로 설정
    # ------------------------------------------------------------------
    # 데이터 소스 경로 (1_data/raw_data)
    DATA_DIR = BASE_DIR / "1_data" / "raw_data"
    
    # 결과 저장 경로 (현재 폴더 내 results)
    RESULT_DIR = Path(__file__).parent / "results"
    
    # 파일명
    INPUT_FILE = "pun_ksm_gsko.csv"
    OUTPUT_FILE = "final_translation_bertscore.csv"
    SUMMARY_FILE = "bertscore_model_summary.csv"

    # ------------------------------------------------------------------
    # 모델 및 실행 설정
    # ------------------------------------------------------------------
    # 사용 모델 (gemini-2.0-flash-lite 권장)
    GEMINI_MODEL = "gemini-2.0-flash-lite"
    
    # 실행 옵션
    TARGET_COUNT = 1000  # 번역할 최대 데이터 개수
    SAVE_BATCH = 50      # 중간 저장 단위 (시스템 안정성 확보)
    
    # ------------------------------------------------------------------
    # 평가 설정 (BERTScore)
    # ------------------------------------------------------------------
    BERTSCORE_MODELS = [
        {
            'name': 'KLUE RoBERTa Large', 
            'model_type': 'klue/roberta-large', 
            'num_layers': 24, 
            'lang': 'ko'
        },
        {
            'name': 'mDeBERTa v3 Base', 
            'model_type': 'microsoft/mdeberta-v3-base', 
            'num_layers': 12, 
            'lang': 'ko'
        }
    ]

    @staticmethod
    def print_config():
        """현재 설정 출력"""
        print("\n--- Configuration ---")
        print(f"Base Dir: {BASE_DIR}")
        print(f"Data Source: {Config.DATA_DIR / Config.INPUT_FILE}")
        print(f"Result Dir: {Config.RESULT_DIR}")
        print(f"Target Model: {Config.GEMINI_MODEL}")
        print(f"Target Count: {Config.TARGET_COUNT}")
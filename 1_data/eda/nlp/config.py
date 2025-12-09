"""
NLP EDA Configuration
======================================================================
목적: 텍스트 통계 분석에 필요한 파일 경로 및 파라미터 관리
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
    # 현재 파일 위치: 1_data/eda/nlp/config.py
    # 프로젝트 루트(1_data) 기준 경로 설정
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # 입력 데이터 경로 (전처리 완료된 CSV)
    # 예: 1_data/raw_data/processed/doc_id_split_sentences.csv
    INPUT_CSV = BASE_DIR / "raw_data" / "doc_id_transcript_dataset_processed.csv"
    
    # 결과 저장 경로
    OUTPUT_DIR = BASE_DIR / "raw_data" / "eda_results"
    OUTPUT_VOCAB = OUTPUT_DIR / "vocab.csv"
    
    # ==================================================================
    # 2. 분석 옵션
    # ==================================================================
    COLUMN_NAME = "transcript"  # 분석할 텍스트 컬럼명 (데이터셋에 맞게 수정 필요)
    TOP_N_VOCAB = 20            # 빈도수 상위 N개 출력
    
    @staticmethod
    def print_config():
        """설정 정보 출력"""
        print("======================================================")
        print(" NLP EDA Configuration")
        print("======================================================")
        print(f" Input Path:  {Config.INPUT_CSV}")
        print(f" Output Path: {Config.OUTPUT_VOCAB}")
        print(f" Target Col:  {Config.COLUMN_NAME}")
        print("======================================================")
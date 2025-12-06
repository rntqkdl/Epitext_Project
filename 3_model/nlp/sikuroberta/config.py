"""
SikuRoBERTa 학습/평가 설정
======================================================================
절대경로 자동 설정 및 학습/테스트 분리
작성자: 4조 복원왕 김탁본
작성일: 2025-12-07
출처: 로컬 개발 (팀 자체 작성)
======================================================================
"""

import os
from pathlib import Path


class PathConfig:
    """경로 설정 (자동으로 절대경로 생성)"""
    
    # 프로젝트 루트 (이 파일 기준 3단계 상위)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    
    # 데이터 경로
    DATA_ROOT = PROJECT_ROOT / "data" / "sikuroberta"
    PREPROCESSED_PATH = DATA_ROOT / "tokenized_sikuroberta_simple_128_extended"
    SPLIT_DATASET_PATH = DATA_ROOT / "tokenized_sikuroberta_simple_128_split"
    
    # 모델 저장 경로
    MODEL_ROOT = PROJECT_ROOT / "3_model" / "nlp" / "sikuroberta"
    CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
    FINAL_MODEL_DIR = MODEL_ROOT / "final_model"
    
    # 로그 및 결과
    LOG_DIR = MODEL_ROOT / "logs"
    TB_LOG_DIR = LOG_DIR / "tensorboard"
    GRAPH_SAVE_PATH = LOG_DIR / "loss_graph.png"
    RESULTS_PATH = LOG_DIR / "test_results.txt"
    
    @classmethod
    def create_directories(cls):
        """필요한 모든 디렉토리 생성"""
        dirs = [
            cls.DATA_ROOT,
            cls.CHECKPOINT_DIR,
            cls.FINAL_MODEL_DIR,
            cls.LOG_DIR,
            cls.TB_LOG_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        print(f"디렉토리 생성 완료")
    
    @classmethod
    def print_paths(cls):
        """경로 정보 출력"""
        print("\n" + "=" * 70)
        print("경로 설정")
        print("=" * 70)
        print(f"프로젝트 루트    : {cls.PROJECT_ROOT}")
        print(f"데이터 루트      : {cls.DATA_ROOT}")
        print(f"전처리 데이터    : {cls.PREPROCESSED_PATH}")
        print(f"분할 데이터      : {cls.SPLIT_DATASET_PATH}")
        print(f"체크포인트       : {cls.CHECKPOINT_DIR}")
        print(f"최종 모델        : {cls.FINAL_MODEL_DIR}")
        print(f"TensorBoard      : {cls.TB_LOG_DIR}")
        print(f"그래프           : {cls.GRAPH_SAVE_PATH}")
        print(f"테스트 결과      : {cls.RESULTS_PATH}")
        print("=" * 70 + "\n")


class TrainingConfig:
    """학습 하이퍼파라미터"""
    
    # 모델
    MODEL_NAME = "SIKU-BERT/sikuroberta"
    
    # 학습 파라미터
    BATCH_SIZE = 4              # GPU 메모리에 따라 조정
    GRAD_ACCUM = 8              # 유효 배치: 4 * 8 = 32
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.06
    
    # MLM 설정
    MLM_PROBABILITY = 0.15      # 마스킹 비율
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3
    
    # 데이터 분할
    TRAIN_RATIO = 0.8           # 80%
    VALID_RATIO = 0.1           # 10%
    TEST_RATIO = 0.1            # 10%
    
    # 로깅
    LOGGING_STEPS = 50
    SAVE_TOTAL_LIMIT = 5        # 최근 5개 체크포인트만 유지
    
    @classmethod
    def print_config(cls):
        """설정 출력"""
        print("\n" + "=" * 70)
        print("학습 설정")
        print("=" * 70)
        print(f"모델             : {cls.MODEL_NAME}")
        print(f"배치 사이즈      : {cls.BATCH_SIZE}")
        print(f"Gradient Accum   : {cls.GRAD_ACCUM}")
        print(f"유효 배치 사이즈 : {cls.BATCH_SIZE * cls.GRAD_ACCUM}")
        print(f"Epochs           : {cls.EPOCHS}")
        print(f"Learning Rate    : {cls.LEARNING_RATE}")
        print(f"MLM 확률         : {cls.MLM_PROBABILITY}")
        print(f"Early Stopping   : {cls.EARLY_STOPPING_PATIENCE} epochs")
        print("=" * 70 + "\n")


class EvalConfig:
    """평가 설정"""
    
    EVAL_BATCH_SIZE = 8
    MLM_PROBABILITY = 0.15      # 학습 때와 동일하게
    
    @classmethod
    def print_config(cls):
        """설정 출력"""
        print("\n" + "=" * 70)
        print("평가 설정")
        print("=" * 70)
        print(f"배치 사이즈      : {cls.EVAL_BATCH_SIZE}")
        print(f"MLM 확률         : {cls.MLM_PROBABILITY}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # 테스트
    PathConfig.print_paths()
    TrainingConfig.print_config()
    EvalConfig.print_config()
    PathConfig.create_directories()

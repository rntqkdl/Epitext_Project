"""
OCR Evaluation Configuration
======================================================================
목적: OCR 성능 평가를 위한 파일 경로 및 설정 관리
작성자: Epitext Project Team
작성일: 2025-12-09
======================================================================
"""

from pathlib import Path

class Config:
    """
    평가 실행을 위한 설정 클래스
    
    기본 사용법:
    - 실행 시 CLI 인자로 경로를 주입하거나,
    - 아래 DEFAULT 경로를 수정하여 사용
    """
    
    # 기본 경로 설정 (프로젝트 내 상대 경로 예시)
    # 실제 사용 시에는 절대 경로를 입력하거나 CLI 인자 사용 권장
    BASE_DIR = Path(__file__).resolve().parent
    
    # 정답 파일 (Ground Truth) 경로
    DEFAULT_GT_PATH = BASE_DIR / "sample_data" / "gt.txt"
    
    # 예측 파일 (Prediction) 경로
    DEFAULT_PRED_PATH = BASE_DIR / "sample_data" / "prediction.json"
    
    # 결과 출력 설정
    VERBOSE = True  # 상세 로그 출력 여부

    @staticmethod
    def print_config():
        """현재 설정 정보 출력"""
        print("=" * 60)
        print("OCR Evaluation Configuration")
        print("=" * 60)
        print(f"Default GT Path:   {Config.DEFAULT_GT_PATH}")
        print(f"Default Pred Path: {Config.DEFAULT_PRED_PATH}")
        print("=" * 60)
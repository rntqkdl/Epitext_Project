"""
OCR Evaluation Module
======================================================================
설명: OCR 모델 예측 결과(JSON)와 정답(TXT) 간의 CER 성능 평가 모듈
작성자: Epitext Project Team
======================================================================
"""

from .config import Config
from .evaluate import calculate_cer, run_evaluation, main

__all__ = [
    "Config",
    "calculate_cer",
    "run_evaluation",
    "main"
]
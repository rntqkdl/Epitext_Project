"""
SikuRoBERTa Evaluation Module
======================================================================
설명: 학습된 모델 성능 평가 모듈 패키지
작성자: Epitext Project Team
======================================================================
"""

from .config import EvalConfig
from .evaluate_task import main

__all__ = [
    "EvalConfig",
    "main"
]
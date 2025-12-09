"""
Gemini Experiment Module
======================================================================
설명: Gemini API 기반 번역 및 평가 실험 모듈 패키지 초기화
작성자: Epitext Project Team
======================================================================
"""

# 주요 모듈 임포트 (외부에서 쉽게 접근 가능하도록 설정)
from .config import Config
from .prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from .run_evaluation import main, run_translation_step, run_bertscore_step

# 모듈 외부로 노출할 요소 정의
__all__ = [
    "Config",
    "SYSTEM_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "main",
    "run_translation_step",
    "run_bertscore_step"
]
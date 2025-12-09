"""
Common Utilities
======================================================================
목적: 문자열 처리, 정규화, 판별 등 공통 헬퍼 함수 모음
작성자: Epitext Project Team
======================================================================
"""

import re
import unicodedata
from typing import Optional

try:
    from config import Config
except ImportError:
    from .config import Config

def is_cjk(char: str) -> bool:
    """입력된 문자가 한자 범위에 속하는지 확인"""
    code = ord(char)
    return any(code in block for block in Config.CJK_RANGES)

def is_hangul(char: str) -> bool:
    """입력된 문자가 한글 범위에 속하는지 확인"""
    code = ord(char)
    return any(code in block for block in Config.HANGUL_RANGES)

def contains_hangul(text: str) -> bool:
    """문자열에 한글이 하나라도 포함되어 있는지 확인"""
    return any(is_hangul(ch) for ch in text)

def cjk_ratio(text: str) -> float:
    """문자열 내 한자 비율 계산 (0.0 ~ 1.0)"""
    chars = text.replace(" ", "")
    if not chars:
        return 0.0
    return sum(1 for ch in chars if is_cjk(ch)) / len(chars)

def normalize_spaces(text: str) -> str:
    """연속된 공백을 단일 공백으로 치환"""
    return re.sub(r"\s+", " ", text.replace("\t", " ")).strip()

def slugify(text: str) -> str:
    """파일명 사용을 위한 문자열 슬러그화 (영문/숫자/언더바)"""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in normalized if ord(ch) < 128)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text.replace(".", " ")).strip("_")
    return slug.lower()

def normalize_caption(text: str) -> Optional[str]:
    """
    탁본 이미지 캡션 정규화 (예: "앞 면" -> "앞면")
    매칭되지 않거나 너무 긴 텍스트는 None 반환
    """
    if not text or len(text) > 25:
        return None

    # 기본 정제
    cleaned = normalize_spaces(text).replace("面", "면")
    cleaned = re.sub(r"[·•∙‧]", "", cleaned)
    
    # 불필요한 숫자/괄호 제거
    condensed = re.sub(r"[\d０-９①-⑳()\[\]...]", "", cleaned)

    if not condensed:
        return None

    # 패턴 매칭
    for pattern, label in Config.FACE_PATTERNS:
        if pattern.search(condensed):
            return label

    return condensed if condensed in Config.ALLOWED_CAPTIONS else None
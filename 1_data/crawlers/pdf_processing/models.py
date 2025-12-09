"""
PDF Data Models
======================================================================
목적: PDF 파싱 과정에서 사용되는 데이터 구조(Data Class) 정의
작성자: Epitext Project Team
======================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class TextLine:
    """PDF 내의 단일 텍스트 라인 정보"""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_size: float = 12.0

@dataclass
class ImageInfo:
    """PDF 내의 이미지 객체 정보"""
    bbox: Tuple[float, float, float, float]
    xref: Optional[int] = None

@dataclass
class PageBundle:
    """단일 페이지의 모든 콘텐츠 (텍스트 + 이미지)"""
    page_index: int
    text_lines: List[TextLine] = field(default_factory=list)
    images: List[ImageInfo] = field(default_factory=list)

@dataclass
class EntryBundle:
    """
    하나의 탁본 항목(Entry) 정보
    예: "01 서울 봉은사..." 항목 전체
    """
    number: str
    name: str
    hanja_name: Optional[str]
    pages: List[PageBundle]
    metadata: Dict
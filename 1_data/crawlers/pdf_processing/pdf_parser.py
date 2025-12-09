"""
PDF Parser Engine
======================================================================
목적: PDF 파일을 읽어 구조화된 EntryBundle 리스트로 변환
작성자: Epitext Project Team
======================================================================
"""

import re
import fitz  # PyMuPDF
from typing import List, Optional, Sequence, Tuple, Dict

try:
    from models import TextLine, ImageInfo, PageBundle, EntryBundle
    from utils import contains_hangul, cjk_ratio, normalize_spaces
except ImportError:
    from .models import TextLine, ImageInfo, PageBundle, EntryBundle
    from .utils import contains_hangul, cjk_ratio, normalize_spaces

def extract_page_content(page: fitz.Page) -> Tuple[List[TextLine], List[ImageInfo]]:
    """단일 페이지에서 텍스트와 이미지 객체를 추출"""
    text_dict = page.get_text("dict")
    text_lines = []
    images = []

    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:  # 텍스트 블록
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    bbox = tuple(span["bbox"])
                    text = span["text"].strip()
                    size = span["size"]
                    if text:
                        text_lines.append(TextLine(text, bbox, size))
        elif block.get("type") == 1:  # 이미지 블록
            bbox = tuple(block["bbox"])
            images.append(ImageInfo(bbox=bbox))

    return text_lines, images

def find_entry_heading(text_lines: Sequence[TextLine]) -> Optional[Dict]:
    """
    새로운 탁본 항목의 시작을 알리는 헤딩을 감지
    예: "01 서울 봉은사..."
    """
    if not text_lines:
        return None

    # 헤딩 후보 라인 수집 (상단부)
    candidates = text_lines[:10]
    
    number_pattern = re.compile(r"^(\d{2})$")
    found_number = None
    start_idx = -1

    # 두 자리 숫자(01, 02 등) 찾기
    for i, line in enumerate(candidates):
        if number_pattern.match(line.text.strip()):
            found_number = line.text.strip()
            start_idx = i
            break
    
    if not found_number:
        return None

    # 이름 추출 (숫자 다음 라인부터)
    name_lines = []
    hanja_lines = []
    
    for i in range(start_idx + 1, min(len(text_lines), start_idx + 5)):
        text = text_lines[i].text.strip()
        if not text: 
            continue
        # 메타데이터 키워드가 나오면 중단
        if any(k in text for k in ["시대", "제작연도", "◆"]):
            break
            
        if cjk_ratio(text) > 0.7:
            hanja_lines.append(text)
        else:
            name_lines.append(text)

    full_name = " ".join(name_lines)
    full_hanja = " ".join(hanja_lines) if hanja_lines else None

    return {
        "number": found_number,
        "name": full_name,
        "hanja_name": full_hanja
    }

def parse_metadata(text_lines: Sequence[TextLine]) -> Dict:
    """텍스트 라인에서 메타데이터(시대, 연도 등) 추출"""
    metadata = {}
    full_text = " ".join([line.text for line in text_lines])
    
    # 시대 추출
    era_match = re.search(r"시\s*대\s*[:]?\s*(\S+)", full_text)
    if era_match:
        metadata["시대"] = era_match.group(1)
        
    # 제작연도 추출
    year_match = re.search(r"제작연도\s*[:]?\s*([^\n]+)", full_text)
    if year_match:
        # 다음 키워드 전까지만 추출
        year_text = year_match.group(1)
        for keyword in ["소재지", "◆"]:
            if keyword in year_text:
                year_text = year_text.split(keyword)[0]
        metadata["제작연도"] = year_text.strip()

    return metadata

def build_entry_bundles(doc: fitz.Document) -> List[EntryBundle]:
    """PDF 문서를 순회하며 항목(Entry) 단위로 구조화"""
    bundles = []
    current_entry = None

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        text_lines, images = extract_page_content(page)
        
        # 페이지 번호 확인 (페이지 하단)
        heading = find_entry_heading(text_lines)

        page_bundle = PageBundle(
            page_index=page_idx,
            text_lines=text_lines,
            images=images
        )

        if heading:
            # 이전 항목 저장
            if current_entry:
                bundles.append(current_entry)
            
            # 새 항목 시작
            current_entry = EntryBundle(
                number=heading["number"],
                name=heading["name"],
                hanja_name=heading["hanja_name"],
                pages=[page_bundle],
                metadata=parse_metadata(text_lines)
            )
        elif current_entry:
            # 현재 항목에 페이지 추가
            current_entry.pages.append(page_bundle)

    # 마지막 항목 추가
    if current_entry:
        bundles.append(current_entry)

    return bundles
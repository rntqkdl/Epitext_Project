# -*- coding: utf-8 -*-
"""
PDF Extraction Main Script
======================================================================
목적: PDF 처리 모듈들을 조합하여 최종 데이터셋 생성 (CSV/JSON/Images)
작성자: Epitext Project Team
작성일: 2025-12-09
======================================================================
"""

import sys
import csv
import json
import logging
import fitz
from pathlib import Path
from typing import List, Dict

# 로컬 모듈 임포트
try:
    from config import Config
    from pdf_parser import build_entry_bundles
    from image_processor import find_takbon_images, merge_adjacent_faces, save_image_area
    from text_extractor import collect_hanja_pages
    from utils import slugify
except ImportError:
    from .config import Config
    from .pdf_parser import build_entry_bundles
    from .image_processor import find_takbon_images, merge_adjacent_faces, save_image_area
    from .text_extractor import collect_hanja_pages
    from .utils import slugify

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def process_single_pdf(pdf_path: Path, output_base_dir: Path) -> List[Dict]:
    """
    단일 PDF 파일을 처리하여 데이터 추출 및 이미지 저장
    """
    logger.info(f"Processing PDF: {pdf_path.name}")
    
    # 1. PDF 열기 및 파싱
    doc = fitz.open(pdf_path)
    bundles = build_entry_bundles(doc)
    logger.info(f"Found {len(bundles)} entries in {pdf_path.name}")

    # 2. 출력 경로 설정
    pdf_slug = slugify(pdf_path.stem)
    images_dir = output_base_dir / "images" / pdf_slug
    
    results = []

    # 3. 항목별 처리
    for bundle in bundles:
        # 3.1 이미지 찾기
        all_images = []
        for page_bundle in bundle.pages:
            for img, caption in find_takbon_images(page_bundle):
                all_images.append((page_bundle, img, caption))
        
        # 3.2 면 병합 (필요 시)
        merged_images, save_info = merge_adjacent_faces(all_images)
        
        # 3.3 한자 원문 수집
        hanja_pages = collect_hanja_pages(bundle.pages)
        hanja_text_full = "\n".join([t[1] for t in hanja_pages])

        # 3.4 데이터 결합 및 이미지 저장
        for (page_idx, bbox, caption) in save_info:
            # 이미지 파일명 생성
            safe_name = slugify(bundle.name)
            img_filename = f"{bundle.number}_{safe_name}_{page_idx}_{slugify(caption)}.png"
            img_path = images_dir / img_filename
            
            # 이미지 저장
            save_image_area(doc, page_idx, bbox, img_path)
            
            # 결과 데이터 생성
            entry_data = {
                "source_pdf": pdf_path.name,
                "entry_number": bundle.number,
                "name_kor": bundle.name,
                "name_hanja": bundle.hanja_name,
                "era": bundle.metadata.get("시대", ""),
                "year": bundle.metadata.get("제작연도", ""),
                "face_caption": caption,
                "original_text": hanja_text_full, # 해당 항목 전체 원문 연결
                "image_path": str(img_path.relative_to(output_base_dir))
            }
            results.append(entry_data)

    doc.close()
    return results

def extract_from_pdf(pdf_dir: Path, output_dir: Path) -> None:
    """
    디렉토리 내의 모든 PDF 처리
    """
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    all_data = []

    # PDF 파일 탐색
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        try:
            data = process_single_pdf(pdf_file, output_dir)
            all_data.extend(data)
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")

    # 결과 저장 (CSV & JSON)
    save_results(all_data, output_dir)

def save_results(data: List[Dict], output_dir: Path) -> None:
    """결과를 CSV 및 JSON으로 저장"""
    if not data:
        logger.warning("No data extracted.")
        return

    # CSV 저장
    csv_path = output_dir / "extracted_takbon_data.csv"
    keys = data[0].keys()
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    
    # JSON 저장
    json_path = output_dir / "extracted_takbon_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved results to {output_dir}")

if __name__ == "__main__":
    # 실행 예시 (사용자가 직접 수정하여 사용)
    # 예: python main.py --input "C:/pdfs" --output "C:/result"
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF 탁본 데이터 추출기")
    parser.add_argument("--input", type=str, required=False, help="PDF 파일이 있는 폴더 경로")
    parser.add_argument("--output", type=str, required=False, help="결과 저장 폴더 경로")
    
    args = parser.parse_args()
    
    # 인자가 없으면 Config 기본값 또는 현재 폴더 사용
    input_path = Path(args.input) if args.input else Config.BASE_DIR / "raw_pdfs"
    output_path = Path(args.output) if args.output else Config.BASE_DIR / "processed_results"
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    extract_from_pdf(input_path, output_path)
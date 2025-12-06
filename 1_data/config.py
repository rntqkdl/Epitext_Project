"""
1_data 통합 설정 파일
======================================================================
목적: 모든 크롤러에서 사용하는 상수, 경로, URL 등을 중앙 관리
작성자: 4조 복원왕 김탁본
작성일: 2025-12-07
출처: 팀 자체 작성 (1주차 보고서)
======================================================================
"""

import os
import logging
from pathlib import Path
from datetime import datetime


# ======================================================================
# 경로 설정
# ======================================================================
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

RAW_DATA_DIR = BASE_DIR / "raw_data"
KYUDB_DIR = RAW_DATA_DIR / "kyudb"
HISTORYDB_DIR = RAW_DATA_DIR / "historydb"
NRICH_DIR = RAW_DATA_DIR / "nrich"

UNIFIED_DB_PATH = BASE_DIR / "unified_metadata.db"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"


# ======================================================================
# 크롤러별 설정
# ======================================================================

KYUDB_CONFIG = {
    "base_url": "https://kyudb.snu.ac.kr",
    "home_url": "https://kyudb.snu.ac.kr/main.do?mid=GSD&submain=Y",
    "list_url_template": "https://kyudb.snu.ac.kr/book/list.do?mid={mid}&book_cate={cate}&_ts={ts}",
    "view_url": "https://kyudb.snu.ac.kr/book/view.do",
    "image_download_url": "https://kyudb.snu.ac.kr/ImageDown.do",
    "periods": [("미상", "GSD0414")],
    "page_load_timeout": 30,
    "request_timeout": 25,
    "retry_max": 3,
    "delay_between_items": 0.3,
    "delay_between_pages": 0.5,
    "output_dir": KYUDB_DIR,
    "metadata_filename": "detail_info.csv",
}

HISTORYDB_CONFIG = {
    "base_url": "https://db.history.go.kr",
    "list_url": "https://db.history.go.kr/goryeo/gskokrList.do",
    "detail_url": "https://db.history.go.kr/goryeo/level.do",
    "image_viewer_url": "https://db.history.go.kr/common/gskoImageViewer.do",
    "image_proxy_url": "https://db.history.go.kr/common/imageProxy.do",
    "start_page": 1,
    "end_page": 55,
    "records_per_page": 20,
    "request_timeout": 10,
    "retry_max": 3,
    "delay_between_requests": 0.3,
    "output_dir": HISTORYDB_DIR,
    "images_subdir": "images",
    "transcriptions_subdir": "transcriptions",
    "translations_subdir": "translations",
    "punctuated_subdir": "punctuated",
    "db_filename": "goryeo_data.db",
}

NRICH_CONFIG = {
    "base_url": "https://portal.nrich.go.kr",
    "view_url_template": "https://portal.nrich.go.kr/kor/ksmUsrView.do?menuIdx=584&ksm_idx={ksm_idx}",
    "ajax_image_url": "https://portal.nrich.go.kr/kor/imageFileListAjax.do",
    "list_url": "https://portal.nrich.go.kr/kor/ksmUsrList.do?menuIdx=584",
    "list_endpoint": "https://portal.nrich.go.kr/kor/ksmUsrList.do",
    "start_page": 1,
    "end_page": 639,
    "page_unit": 10,
    "max_workers": 20,
    "batch_size": 200,
    "batch_delay": 5,
    "main_timeout": 20,
    "download_timeout": 30,
    "retry_max": 3,
    "output_dir": NRICH_DIR,
    "images_subdir": "images",
    "transcriptions_subdir": "transcriptions",
    "translations_subdir": "translations",
    "indices_filename": "ksm_indices.txt",
}


# ======================================================================
# 공통 HTTP 설정
# ======================================================================
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

IMAGE_HEADERS = {
    "User-Agent": DEFAULT_HEADERS["User-Agent"],
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


# ======================================================================
# 데이터베이스 스키마
# ======================================================================
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    original_id TEXT,
    status TEXT DEFAULT 'pending',
    retries INTEGER DEFAULT 0,
    title TEXT,
    period TEXT,
    script_style TEXT,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_msg TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS texts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    text_type TEXT NOT NULL,
    version TEXT,
    file_path TEXT NOT NULL,
    char_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_images_doc_id ON images(doc_id);
CREATE INDEX IF NOT EXISTS idx_texts_doc_id ON texts(doc_id);
CREATE INDEX IF NOT EXISTS idx_texts_type ON texts(text_type);
"""


# ======================================================================
# 유틸리티 함수
# ======================================================================
def ensure_directories():
    """필요한 모든 디렉토리를 생성합니다"""
    directories = [
        RAW_DATA_DIR,
        KYUDB_DIR / "metadata",
        KYUDB_DIR / "images",
        HISTORYDB_DIR / "images",
        HISTORYDB_DIR / "transcriptions",
        HISTORYDB_DIR / "translations",
        HISTORYDB_DIR / "punctuated",
        NRICH_DIR / "images",
        NRICH_DIR / "transcriptions",
        NRICH_DIR / "translations",
        SAMPLE_DATA_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"디렉토리 생성 완료: {len(directories)}개")


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    크롤러별 로거 설정
    
    Args:
        name: 로거 이름
        log_file: 로그 파일 경로
    
    Returns:
        logging.Logger 인스턴스
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


if __name__ == "__main__":
    print("=" * 60)
    print("프로젝트 디렉토리 초기화")
    print("=" * 60)
    ensure_directories()
    print(f"\n통합 DB 경로: {UNIFIED_DB_PATH}")
    print(f"원본 데이터 경로: {RAW_DATA_DIR}")
    print("\n설정 파일 로드 완료!")

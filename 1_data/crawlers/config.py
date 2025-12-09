"""
Integrated Configuration File
======================================================================
목적: 모든 크롤러에서 사용하는 상수, 경로, URL 등을 중앙 관리
작성자: Epitext Project Team
======================================================================
"""

import os
from pathlib import Path

# ======================================================================
# 경로 설정
# ======================================================================
# 프로젝트 루트 디렉토리 (1_data/ 폴더 기준)
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

# 원본 데이터 저장 경로
RAW_DATA_DIR = BASE_DIR / "raw_data"
KYUDB_DIR = RAW_DATA_DIR / "kyudb"
HISTORYDB_DIR = RAW_DATA_DIR / "historydb"
NRICH_DIR = RAW_DATA_DIR / "nrich"

# 통합 메타데이터 DB
UNIFIED_DB_PATH = BASE_DIR / "unified_metadata.db"

# 샘플 데이터 경로
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

# ======================================================================
# 크롤러별 설정
# ======================================================================

# --- 규장각(Kyudb) 설정 ---
KYUDB_CONFIG = {
    "base_url": "https://kyudb.snu.ac.kr",
    "home_url": "https://kyudb.snu.ac.kr/main.do?mid=GSD&submain=Y",
    "list_url_template": "https://kyudb.snu.ac.kr/book/list.do?mid={mid}&book_cate={cate}&_ts={ts}",
    "view_url": "https://kyudb.snu.ac.kr/book/view.do",
    "image_download_url": "https://kyudb.snu.ac.kr/ImageDown.do",
    
    # 수집 대상 연대 (이름, 카테고리 코드)
    "periods": [
        ("미상", "GSD0414"),
        ("조선전기", "GSD0401"),
        ("조선중기", "GSD0402"),
    ],
    
    # 크롤링 설정
    "page_load_timeout": 30,
    "request_timeout": 25,
    "retry_max": 3,
    "delay_between_items": 0.3,
    "delay_between_pages": 0.5,
    
    # 저장 경로
    "output_dir": KYUDB_DIR,
    "metadata_filename": "detail_info.csv",
}

# --- 국사편찬위원회(HistoryDB) 설정 ---
HISTORYDB_CONFIG = {
    "base_url": "https://db.history.go.kr",
    "list_url": "https://db.history.go.kr/goryeo/gskokrList.do",
    "detail_url": "https://db.history.go.kr/goryeo/level.do",
    "image_viewer_url": "https://db.history.go.kr/common/gskoImageViewer.do",
    "image_proxy_url": "https://db.history.go.kr/common/imageProxy.do",
    
    # 수집 범위
    "start_page": 1,
    "end_page": 55,
    "records_per_page": 20,
    
    # 크롤링 설정
    "request_timeout": 10,
    "retry_max": 3,
    "delay_between_requests": 0.3,
    
    # 저장 경로
    "output_dir": HISTORYDB_DIR,
    "images_subdir": "images",
    "transcriptions_subdir": "transcriptions",
    "translations_subdir": "translations",
    "punctuated_subdir": "punctuated",
    "db_filename": "goryeo_data.db",
}

# --- 문화재연구소(NRICH) 설정 ---
NRICH_CONFIG = {
    "base_url": "https://portal.nrich.go.kr",
    "view_url_template": "https://portal.nrich.go.kr/kor/ksmUsrView.do?menuIdx=584&ksm_idx={ksm_idx}",
    "ajax_image_url": "https://portal.nrich.go.kr/kor/imageFileListAjax.do",
    "list_url": "https://portal.nrich.go.kr/kor/ksmUsrList.do?menuIdx=584",
    "list_endpoint": "https://portal.nrich.go.kr/kor/ksmUsrList.do",
    
    # 수집 범위
    "start_page": 1,
    "end_page": 639,
    "page_unit": 10,
    
    # 병렬 처리 설정
    "max_workers": 20,
    "batch_size": 200,
    "batch_delay": 5,
    
    # 크롤링 설정
    "main_timeout": 20,
    "download_timeout": 30,
    "retry_max": 3,
    
    # 저장 경로
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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
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
-- 메인 문서 테이블
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

-- 이미지 파일 테이블
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

-- 텍스트 파일 테이블
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

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_images_doc_id ON images(doc_id);
CREATE INDEX IF NOT EXISTS idx_texts_doc_id ON texts(doc_id);
CREATE INDEX IF NOT EXISTS idx_texts_type ON texts(text_type);
"""

# ======================================================================
# 유틸리티 함수
# ======================================================================
import logging
from datetime import datetime

def ensure_directories():
    """필요한 모든 디렉토리를 생성합니다."""
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
    print(f"[Info] Created {len(directories)} directories.")

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """크롤러별 로거를 설정합니다."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

if __name__ == "__main__":
    print("========================================================================")
    print(" Project Directory Initialization")
    print("========================================================================")
    ensure_directories()
    print(f" Unified DB Path: {UNIFIED_DB_PATH}")
    print(f" Raw Data Path:   {RAW_DATA_DIR}")
    print(" [Success] Configuration loaded.")
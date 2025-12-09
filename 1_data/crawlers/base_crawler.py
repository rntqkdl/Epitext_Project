"""
Base Crawler Class
======================================================================
목적: 모든 크롤러가 상속받는 공통 기능 제공 (DB 연동, 파일 저장, 재시도 등)
작성자: Epitext Project Team
======================================================================
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import sys

# 상위 경로 설정
sys.path.append(str(Path(__file__).parent.parent))
from config import DEFAULT_HEADERS, setup_logger
from utils import DatabaseManager, FileHandler, RetrySession

class BaseCrawler(ABC):
    """모든 크롤러의 기본 클래스"""
    
    def __init__(
        self,
        source: str,
        config: Dict[str, Any],
        db_manager: Optional[DatabaseManager] = None,
        file_handler: Optional[FileHandler] = None
    ):
        self.source = source
        self.config = config
        
        # DB & 파일 핸들러
        self.db = db_manager or DatabaseManager()
        self.fh = file_handler or FileHandler()
        
        # 로거
        self.logger = setup_logger(f"{source}_crawler")
        
        # HTTP 세션
        self.session = RetrySession(
            max_retries=config.get("retry_max", 3),
            backoff_base=config.get("backoff_base", 1.0)
        )
        self.session.headers.update(DEFAULT_HEADERS)
        
        # 통계
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "images_downloaded": 0,
            "texts_saved": 0
        }
        
    @abstractmethod
    def fetch_item_list(self) -> List[Dict[str, Any]]:
        """수집할 항목 목록 가져오기"""
        pass
        
    @abstractmethod
    def scrape_item(self, item_info: Dict[str, Any]) -> bool:
        """개별 항목 스크래핑"""
        pass
        
    def generate_doc_id(self, original_id: str) -> str:
        """통합 문서 ID 생성"""
        prefix_map = {
            "kyudb": "KYU",
            "historydb": "HIS",
            "nrich": "NRI"
        }
        prefix = prefix_map.get(self.source, "UNK")
        return f"{prefix}_{original_id}"
        
    def save_document_metadata(
        self,
        doc_id: str,
        original_id: str,
        title: Optional[str] = None,
        period: Optional[str] = None,
        script_style: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """문서 메타데이터를 DB에 저장"""
        success = self.db.insert_document(
            doc_id=doc_id,
            source=self.source,
            original_id=original_id,
            title=title,
            period=period,
            script_style=script_style,
            metadata=metadata
        )
        
        if success:
            self.logger.info(f"[DB Saved] {doc_id}")
        else:
            self.logger.warning(f"[DB Failed] {doc_id}")
            
        return success
        
    def update_status(
        self,
        doc_id: str,
        status: str,
        error_msg: Optional[str] = None,
        increment_retries: bool = False
    ):
        """문서 상태 업데이트"""
        self.db.update_document_status(doc_id, status, error_msg, increment_retries)
        
    def save_image(
        self,
        doc_id: str,
        image_url: str,
        filename: str,
        referer: Optional[str] = None
    ) -> bool:
        """이미지 다운로드 및 저장"""
        sourcedir = self.fh.get_source_dir(self.source)
        imagesdir = sourcedir / "images"
        savepath = imagesdir / filename
        
        # 이미 존재하면 스킵
        if self.fh.file_exists(savepath):
            self.logger.debug(f"[Skip] Image exists: {filename}")
            return True
            
        # Referer 헤더 추가
        headers = self.session.headers.copy()
        if referer:
            headers["Referer"] = referer
            
        # 이미지 다운로드
        image_info = self.fh.save_image_from_url(
            url=image_url,
            savepath=savepath,
            session=self.session,
            headers=headers,
            timeout=self.config.get("download_timeout", 30)
        )
        
        if image_info:
            # DB에 기록
            relpath = self.fh.get_relative_path(savepath)
            self.db.insert_image(
                doc_id=doc_id,
                file_path=relpath,
                file_size=image_info["size"],
                width=image_info.get("width"),
                height=image_info.get("height")
            )
            self.stats["images_downloaded"] += 1
            self.logger.debug(f"[Downloaded] {filename} ({image_info['size']:,}B)")
            return True
        else:
            self.logger.warning(f"[Failed] Image download: {filename}")
            return False
            
    def save_text_file(
        self,
        doc_id: str,
        content: str,
        text_type: str,
        filename: str,
        version: Optional[str] = None
    ) -> bool:
        """텍스트 파일 저장"""
        sourcedir = self.fh.get_source_dir(self.source)
        
        # 텍스트 타입별 하위 디렉토리
        type_dir_map = {
            "transcript": "transcriptions",
            "translation": "translations",
            "punctuated": "punctuated"
        }
        subdir = type_dir_map.get(text_type, "texts")
        textdir = sourcedir / subdir
        savepath = textdir / filename
        
        result = self.fh.save_text(content, savepath, skip_if_empty=True)
        
        if result:
            # DB에 기록
            relpath = self.fh.get_relative_path(savepath)
            char_count = len(content) if content else 0
            self.db.insert_text(
                doc_id=doc_id,
                text_type=text_type,
                file_path=relpath,
                version=version,
                char_count=char_count
            )
            self.stats["texts_saved"] += 1
            self.logger.debug(f"[Saved] {filename} ({char_count} chars)")
            return True
        else:
            self.logger.debug(f"[Skip] Empty content: {filename}")
            return False
            
    def delay(self, seconds: Optional[float] = None):
        """대기 시간"""
        if seconds is None:
            seconds = self.config.get("delay_between_items", 0.3)
            
        if seconds > 0:
            time.sleep(seconds)
            
    def run(self):
        """크롤러 실행 (메인 루프)"""
        self.logger.info("============================================================")
        self.logger.info(f" {self.source.upper()} Crawler Started")
        self.logger.info("============================================================")
        
        try:
            # 1. 수집 대상 목록 가져오기
            self.logger.info("Collecting item list...")
            items = self.fetch_item_list()
            self.stats["total"] = len(items)
            self.logger.info(f"[Found] {len(items)} items\n")
            
            # 2. 각 항목 스크래핑
            self.logger.info(f"Start scraping {len(items)} items...\n")
            for idx, item_info in enumerate(items, 1):
                original_id = item_info.get("original_id", "unknown")
                self.logger.info(f"[{idx}/{len(items)}] {original_id}")
                
                try:
                    success = self.scrape_item(item_info)
                    if success:
                        self.stats["success"] += 1
                    else:
                        self.stats["failed"] += 1
                        
                    self.delay()
                    
                except Exception as e:
                    self.logger.error(f"[Error] {e}")
                    self.stats["failed"] += 1
            
            # 3. 최종 통계
            self.print_final_stats()
            
        except KeyboardInterrupt:
            self.logger.warning("\n[Warning] User interrupted")
            self.print_final_stats()
            
        except Exception as e:
            self.logger.error(f"[Critical Error] {e}", exc_info=True)
            
        finally:
            self.cleanup()
            
    def print_final_stats(self):
        """최종 통계 출력"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info(" Crawling Finished - Final Stats")
        self.logger.info("=" * 60)
        self.logger.info(f"  Total:            {self.stats['total']:5} items")
        self.logger.info(f"  Success:          {self.stats['success']:5} items")
        self.logger.info(f"  Failed:           {self.stats['failed']:5} items")
        self.logger.info(f"  Skipped:          {self.stats['skipped']:5} items")
        self.logger.info(f"  Images Downloaded:{self.stats['images_downloaded']:5} items")
        self.logger.info(f"  Texts Saved:      {self.stats['texts_saved']:5} items")
        self.logger.info("=" * 60)
        
        # DB 통계도 출력
        self.db.print_statistics()
        
    def cleanup(self):
        """정리 작업"""
        if self.session:
            self.session.close()
            self.logger.info("[Session Closed]")
"""
Korean History DB Crawler
======================================================================
목적: 국사편찬위원회 고려사 데이터베이스 수집
URL: https://db.history.go.kr/goryeo
작성자: Epitext Project Team
======================================================================
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import sys

# 상위 경로 설정
sys.path.append(str(Path(__file__).parent.parent))
from config import HISTORYDB_CONFIG
from crawlers.base_crawler import BaseCrawler

class HistoryDBCrawler(BaseCrawler):
    """국사편찬위원회 크롤러 (requests 기반)"""
    
    def __init__(self, config: Dict[str, Any] = HISTORYDB_CONFIG):
        super().__init__(source="historydb", config=config)
        self.EMPTY_TEXT = "[EMPTY]"  # 빈 텍스트 표시용
        
    def scrape_id_list(self) -> List[str]:
        """전체 문서 ID 목록 수집"""
        self.logger.info("Collecting Document IDs...")
        
        base_params = {
            "itemId": "gsko",
            "levelId": "",
            "types": "o",
            "recordCountPerPage": "20",
            "orderColumn": "LEVEL_ID",
            "orderDir": "asc"
        }
        
        all_ids = []
        start_page = self.config["start_page"]
        end_page = self.config["end_page"]
        
        for page_num in range(start_page, end_page + 1):
            params = base_params.copy()
            params["pageIndex"] = page_num
            
            try:
                response = self.session.get(
                    self.config["list_url"],
                    params=params,
                    timeout=self.config["request_timeout"]
                )
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, "html.parser")
                subject_links = soup.select("div.subject a[onclick]")
                
                if not subject_links:
                    self.logger.info(f"Page {page_num}: No more items found.")
                    break
                
                for a_tag in subject_links:
                    match = re.search(r"fnSearchDetailClick\('(.+?)'\)", a_tag["onclick"])
                    if match:
                        full_id = match.group(1)
                        if full_id not in all_ids:
                            all_ids.append(full_id)
                            
                if page_num % 10 == 0:
                    self.logger.info(f"Page {page_num}/{end_page}: Collected {len(all_ids)} IDs so far.")
                    
                self.delay(self.config["delay_between_requests"])
                
            except Exception as e:
                self.logger.warning(f"Page {page_num} Error: {e}")
                continue
                
        self.logger.info(f"[Done] Total {len(all_ids)} IDs collected.\n")
        return all_ids
        
    def parse_metadata_and_texts(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """메타데이터 및 텍스트 파싱"""
        data = {
            "title": None,
            "period": None,
            "script_style": None,
            "transcript": self.EMPTY_TEXT,
            "punctuated": self.EMPTY_TEXT,
            "translation": self.EMPTY_TEXT
        }
        
        try:
            # 제목
            title_tag = soup.select_one("section.section-top div.title")
            if title_tag:
                data["title"] = title_tag.get_text(strip=True)
                
            # 메타데이터
            all_tits = soup.find_all("div", class_="tit")
            for tit_tag in all_tits:
                tit_text = tit_tag.get_text(strip=True)
                cont_tag = tit_tag.find_next_sibling("div", class_="cont")
                
                if cont_tag:
                    cont_text = cont_tag.get_text(strip=True)
                    
                    if "시대" in tit_text:
                        data["period"] = cont_text.replace(" ", "")
                    elif "서체" in tit_text:
                        data["script_style"] = cont_text.split(" ")[0]
            
            # 탭별 텍스트 (판독문, 구두점, 번역문)
            tab_spans = soup.select("div.wh-tabmenu div.item span")
            content_blocks = soup.select("div.wh-tab-cont div.tab-cont")
            
            for i, tab_span in enumerate(tab_spans):
                tab_name = tab_span.get_text(strip=True)
                
                if i < len(content_blocks):
                    content_block = content_blocks[i]
                    txt_wraps = content_block.find_all("div", class_="txt-wrap")
                    
                    if txt_wraps:
                        all_text_parts = []
                        for txt_wrap in txt_wraps:
                            # 팝업 제거
                            for popup in txt_wrap.find_all("span", class_="fc1"):
                                popup.decompose()
                            all_text_parts.append(txt_wrap.get_text(strip=True))
                        
                        text = "\n".join(part for part in all_text_parts if part)
                        text = text if text else self.EMPTY_TEXT
                    else:
                        text = self.EMPTY_TEXT
                        
                    if "판독문" in tab_name:
                        data["transcript"] = text
                    elif "구두점" in tab_name:
                        data["punctuated"] = text
                    elif "번역문" in tab_name:
                        data["translation"] = text
                        
        except Exception as e:
            self.logger.warning(f"[Metadata Parsing Error] {e}")
            
        return data
        
    def get_image_paths(self, original_id: str) -> List[str]:
        """이미지 경로 목록 가져오기"""
        params = {"levelId": original_id}
        
        try:
            response = self.session.get(
                self.config["image_viewer_url"],
                params=params,
                timeout=self.config["request_timeout"]
            )
            response.raise_for_status()
            
            # JavaScript 변수에서 이미지 배열 추출
            match = re.search(r"var\s+imgArr\s*=\s*(\[.+?\]);", response.text)
            if not match:
                return []
                
            image_paths = json.loads(match.group(1))
            return image_paths
            
        except Exception as e:
            self.logger.debug(f"[Image Path Error] {e}")
            return []
            
    def download_image_proxy(
        self,
        fullpath: str,
        doc_id: str,
        filename: str
    ) -> bool:
        """imageProxy.do를 통한 이미지 다운로드"""
        if not fullpath.startswith("/"):
            fullpath = "/" + fullpath
            
        params = {"filePath": fullpath}
        url = self.config["image_proxy_url"]
        
        sourcedir = self.fh.get_source_dir("historydb")
        imagesdir = sourcedir / "images"
        savepath = imagesdir / filename
        
        if self.fh.file_exists(savepath):
            return True
            
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config["request_timeout"],
                stream=True
            )
            
            if response.status_code == 404:
                return False
                
            response.raise_for_status()
            
            data = b"".join(response.iter_content(chunk_size=8192))
            image_info = self.fh.save_image_from_bytes(data, savepath)
            
            if image_info:
                relpath = self.fh.get_relative_path(savepath)
                self.db.insert_image(
                    doc_id=doc_id,
                    file_path=relpath,
                    file_size=image_info["size"],
                    width=image_info.get("width"),
                    height=image_info.get("height")
                )
                self.stats["images_downloaded"] += 1
                return True
                
            return False
            
        except Exception as e:
            self.logger.debug(f"[Download Error] {filename}: {e}")
            return False
            
    # BaseCrawler 메서드 구현
    def fetch_item_list(self) -> List[Dict[str, Any]]:
        """수집 대상 ID 목록"""
        id_list = self.scrape_id_list()
        
        items = []
        for original_id in id_list:
            doc_id = self.generate_doc_id(original_id)
            
            # DB에 등록
            self.db.insert_document(
                doc_id=doc_id,
                source=self.source,
                original_id=original_id
            )
            
            items.append({
                "original_id": original_id,
                "doc_id": doc_id
            })
            
        return items
        
    def scrape_item(self, item_info: Dict[str, Any]) -> bool:
        """개별 항목 스크래핑"""
        original_id = item_info["original_id"]
        doc_id = item_info.get("doc_id") or self.generate_doc_id(original_id)
        
        try:
            # 1. 메인 페이지 가져오기
            params = {"levelId": original_id}
            response = self.session.get(
                self.config["detail_url"],
                params=params,
                timeout=self.config["request_timeout"]
            )
            response.raise_for_status()
            main_soup = BeautifulSoup(response.text, "html.parser")
            
            # 2. 텍스트 탭 페이지
            text_params = params.copy()
            text_params.update({"tabgubun": "10", "tabclickat": "Y"})
            text_response = self.session.get(
                self.config["detail_url"],
                params=text_params,
                timeout=self.config["request_timeout"]
            )
            text_response.raise_for_status()
            text_soup = BeautifulSoup(text_response.text, "html.parser")
            
            # 3. 메타데이터 및 텍스트 파싱
            data = self.parse_metadata_and_texts(text_soup)
            
            # 4. DB에 메타데이터 저장
            self.save_document_metadata(
                doc_id=doc_id,
                original_id=original_id,
                title=data["title"],
                period=data["period"],
                script_style=data["script_style"]
            )
            
            # 5. 텍스트 파일 저장
            text_types = [
                ("transcript", data["transcript"], "국사편찬위원회"),
                ("punctuated", data["punctuated"], "국사편찬위원회"),
                ("translation", data["translation"], "국사편찬위원회"),
            ]
            
            for text_type, content, version in text_types:
                if content and content != self.EMPTY_TEXT:
                    filename = f"{original_id}_{text_type}.txt"
                    self.save_text_file(
                        doc_id=doc_id,
                        content=content,
                        text_type=text_type,
                        filename=filename,
                        version=version
                    )
                    
            # 6. 이미지 다운로드
            image_paths = self.get_image_paths(original_id)
            if image_paths:
                self.logger.debug(f"Found {len(image_paths)} images")
                for idx, fullpath in enumerate(image_paths, 1):
                    filename = f"{original_id}_img{idx}{Path(fullpath).suffix}"
                    self.download_image_proxy(fullpath, doc_id, filename)
                    self.delay(0.3)
                    
            # 7. 상태 업데이트
            self.update_status(doc_id, "completed")
            self.logger.info(f"[Completed] {original_id} (Images: {len(image_paths)})")
            return True
            
        except Exception as e:
            self.logger.error(f"[Failed] {original_id}: {e}")
            self.update_status(doc_id, "error", str(e))
            return False

def main():
    crawler = HistoryDBCrawler()
    crawler.run()

if __name__ == "__main__":
    main()
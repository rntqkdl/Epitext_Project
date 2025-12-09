"""
NRICH Crawler
======================================================================
목적: 국립문화재연구소 금석문 종합영상정보시스템 수집
URL: https://portal.nrich.go.kr
작성자: Epitext Project Team
======================================================================
"""

import re
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import sys

# 상위 경로 설정
sys.path.append(str(Path(__file__).parent.parent))
from config import NRICH_CONFIG
from crawlers.base_crawler import BaseCrawler

class NRICHCrawler(BaseCrawler):
    """문화재연구소 크롤러 (병렬 처리 지원)"""
    
    def __init__(self, config: Dict[str, Any] = NRICH_CONFIG):
        super().__init__(source="nrich", config=config)
        self.max_workers = config.get("max_workers", 20)
        self.batch_size = config.get("batch_size", 200)
        
    def load_indices_from_file(self) -> List[str]:
        """ksm_indices.txt에서 ID 목록 로드"""
        indices_file = self.fh.basedir / self.config["indices_filename"]
        
        if not indices_file.exists():
            self.logger.warning(f"[Warning] Index file not found: {indices_file}")
            return []
            
        try:
            with open(indices_file, "r", encoding="utf-8") as f:
                indices = [line.strip() for line in f if line.strip()]
                
            unique_indices = list(set(indices))
            self.logger.info(f"[Loaded] {len(unique_indices)} IDs")
            return unique_indices
            
        except Exception as e:
            self.logger.error(f"[Error] Failed to load indices: {e}")
            return []
            
    def parse_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """메타데이터 파싱"""
        metadata = {
            "title": "",
            "period": "",
            "script_style": ""
        }
        
        try:
            # 제목
            title_element = soup.select_one("h3.tit")
            if title_element:
                metadata["title"] = title_element.get_text(strip=True)
                
            # dt-dd 쌍에서 정보 추출
            all_dt_elements = soup.find_all("dt")
            
            for dt in all_dt_elements:
                dt_text = dt.get_text(strip=True)
                dd_element = dt.find_next_sibling("dd")
                
                if dd_element:
                    if "서체" in dt_text:
                        metadata["script_style"] = dd_element.get_text(strip=True)
                    elif "시대" in dt_text or "시기" in dt_text:
                        metadata["period"] = dd_element.get_text(strip=True)
                        
            # main-content dl에서 시대 정보 (백업)
            main_content_dl = soup.select_one("dl.main-content")
            if main_content_dl:
                all_dds = main_content_dl.find_all("dd", recursive=False)
                if len(all_dds) > 1:
                    period_base = all_dds[0].get_text(strip=True)
                    if period_base != "!":
                        metadata["period"] = period_base
                        
        except Exception as e:
            self.logger.debug(f"Metadata Parse Error: {e}")
            
        return metadata
        
    def parse_texts(
        self,
        soup: BeautifulSoup,
        doc_id: str,
        original_id: str
    ) -> int:
        """텍스트 추출 및 저장"""
        saved_count = 0
        
        try:
            text_divs = soup.select("div.ksmPanTxtDiv")
            
            for idx, div in enumerate(text_divs):
                text = div.get_text(strip=True)
                if not text:
                    continue
                    
                # tr에서 레이블과 출처 찾기
                tr = div.find_parent("tr")
                if not tr:
                    continue
                    
                all_tds = tr.find_all("td")
                if len(all_tds) < 4:
                    continue
                    
                label = all_tds[1].get_text(strip=True)
                source = all_tds[2].get_text(strip=True)
                version = source if source else None
                
                # 텍스트 타입 판별
                if "판독문" in label:
                    text_type = "transcript"
                    filename = f"{original_id}_transcript{idx+1}.txt"
                elif label in ["번역문", "해석문"]:
                    text_type = "translation"
                    filename = f"{original_id}_translation{idx+1}.txt"
                else:
                    continue
                    
                if self.save_text_file(doc_id, text, text_type, filename, version):
                    saved_count += 1
                    
        except Exception as e:
            self.logger.debug(f"Text Parse Error: {e}")
            
        return saved_count
        
    def get_image_urls(self, ksm_idx: str, data_link: str) -> List[str]:
        """Ajax를 통해 이미지 URL 목록 가져오기"""
        image_urls = []
        
        try:
            rubbing_url = data_link + "?tabgubun=11&tabclickat=Y"
            headers = self.session.headers.copy()
            headers["Referer"] = data_link
            
            response = self.session.get(
                rubbing_url,
                headers=headers,
                timeout=self.config["main_timeout"]
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "lxml")
            gallery_links = soup.select("div.galleryBox a")
            
            for link in gallery_links:
                onclick_attr = link.get("onclick")
                if not onclick_attr or "fnCommonImgViewerCall" not in onclick_attr:
                    continue
                    
                params_match = re.search(r"\((.*?)\)", onclick_attr)
                if not params_match:
                    continue
                    
                params_list = [p.strip().strip("\'\"") for p in params_match.group(1).split(",")]
                
                if len(params_list) < 5 or params_list[4] != "3302":
                    continue
                    
                ajax_params = {
                    "relMenuCd": params_list[0],
                    "relKey": params_list[1],
                    "fileTypeCd": params_list[2],
                    "fileIdx": params_list[3],
                    "st": params_list[4],
                    "subFileTypeCd": params_list[4]
                }
                
                ajax_response = self.session.get(
                    self.config["ajax_image_url"],
                    params=ajax_params,
                    headers=headers,
                    timeout=self.config["main_timeout"]
                )
                
                if ajax_response.status_code == 200:
                    image_data_list = ajax_response.json().get("rs", [])
                    for img_data in image_data_list:
                        file_path = img_data.get("savePath")
                        filename = img_data.get("saveFileNm")
                        
                        if file_path and filename:
                            full_url = f"{self.config['base_url']}{file_path}{filename}"
                            if full_url not in image_urls:
                                image_urls.append(full_url)
                                
        except Exception as e:
            self.logger.debug(f"Image URL Error: {e}")
            
        return image_urls
        
    def download_images(
        self,
        doc_id: str,
        original_id: str,
        image_urls: List[str],
        referer: str
    ) -> int:
        """이미지 다운로드"""
        downloaded = 0
        headers = self.session.headers.copy()
        headers["Referer"] = referer
        
        sourcedir = self.fh.get_source_dir("nrich")
        imagesdir = sourcedir / "images"
        
        for idx, img_url in enumerate(image_urls, 1):
            parsed_url = urllib.parse.urlparse(img_url)
            file_ext = Path(parsed_url.path).suffix or ".jpg"
            filename = f"{original_id}_img{idx}{file_ext}"
            savepath = imagesdir / filename
            
            if self.fh.file_exists(savepath):
                downloaded += 1
                continue
                
            try:
                response = self.session.get(
                    img_url,
                    headers=headers,
                    timeout=self.config["download_timeout"],
                    stream=True
                )
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
                    downloaded += 1
                    self.stats["images_downloaded"] += 1
                    
            except Exception as e:
                self.logger.debug(f"Download Error ({filename}): {e}")
                
        return downloaded
        
    # BaseCrawler 메서드 구현
    def fetch_item_list(self) -> List[Dict[str, Any]]:
        """ID 목록 로드"""
        indices = self.load_indices_from_file()
        
        if not indices:
            self.logger.error("[Error] Empty Index List")
            return []
            
        items = []
        for ksm_idx in indices:
            doc_id = self.generate_doc_id(ksm_idx)
            
            self.db.insert_document(
                doc_id=doc_id,
                source=self.source,
                original_id=ksm_idx
            )
            
            items.append({"original_id": ksm_idx, "doc_id": doc_id})
            
        return items
        
    def scrape_item(self, item_info: Dict[str, Any]) -> bool:
        """개별 항목 스크래핑"""
        ksm_idx = item_info["original_id"]
        doc_id = item_info.get("doc_id") or self.generate_doc_id(ksm_idx)
        data_link = self.config["view_url_template"].format(ksm_idx=ksm_idx)
        
        try:
            # 1. 메인 페이지
            response = self.session.get(data_link, timeout=self.config["main_timeout"])
            response.raise_for_status()
            main_soup = BeautifulSoup(response.content, "lxml")
            
            # 2. 텍스트 탭
            text_link = data_link + "?tabgubun=10&tabclickat=Y"
            text_response = self.session.get(text_link, timeout=self.config["main_timeout"])
            text_response.raise_for_status()
            text_soup = BeautifulSoup(text_response.content, "lxml")
            
            # 3. 메타데이터
            metadata = self.parse_metadata(main_soup)
            
            # 4. DB 저장
            self.save_document_metadata(
                doc_id=doc_id,
                original_id=ksm_idx,
                title=metadata["title"],
                period=metadata["period"],
                script_style=metadata["script_style"]
            )
            
            # 5. 텍스트 저장
            text_count = self.parse_texts(text_soup, doc_id, ksm_idx)
            
            # 6. 이미지 다운로드
            has_rubbing = any("탁본" in btn.get_text(strip=True) 
                             for btn in main_soup.select("li.bugali button"))
            image_count = 0
            
            if has_rubbing:
                image_urls = self.get_image_urls(ksm_idx, data_link)
                if image_urls:
                    image_count = self.download_images(doc_id, ksm_idx, image_urls, data_link)
                    
            # 7. 상태 업데이트
            self.update_status(doc_id, "completed")
            self.logger.info(f"[Completed] {ksm_idx} (Texts: {text_count}, Images: {image_count})")
            return True
            
        except Exception as e:
            self.logger.error(f"[Failed] {ksm_idx}: {e}")
            self.update_status(doc_id, "error", str(e))
            return False
            
    def run(self):
        """병렬 처리 실행"""
        self.logger.info("============================================================")
        self.logger.info(f" {self.source.upper()} Crawler Started")
        self.logger.info(f" Max Workers: {self.max_workers}")
        self.logger.info("============================================================")
        
        try:
            # 1. 항목 목록
            self.logger.info("Collecting item list...")
            items = self.fetch_item_list()
            self.stats["total"] = len(items)
            self.logger.info(f"[Found] {len(items)} items\n")
            
            # 2. 배치 단위 병렬 처리
            batch_num = 1
            while True:
                # pending 항목 가져오기
                pending_items = self.db.get_pending_documents(
                    source=self.source,
                    max_retries=10,
                    limit=self.batch_size
                )
                
                if not pending_items:
                    self.logger.info("[Done] No more pending items.")
                    break
                    
                self.logger.info("============================================================")
                self.logger.info(f" Batch {batch_num}: Processing {len(pending_items)} items")
                self.logger.info("============================================================")
                
                # 병렬 실행
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(
                            self.scrape_item,
                            {"original_id": item["original_id"], "doc_id": item["doc_id"]}
                        ): item for item in pending_items
                    }
                    
                    for future in as_completed(futures):
                        item = futures[future]
                        try:
                            success = future.result()
                            if success:
                                self.stats["success"] += 1
                            else:
                                self.stats["failed"] += 1
                        except Exception as e:
                            self.logger.error(f"[Error] {item['original_id']}: {e}")
                            self.stats["failed"] += 1
                            
                self.logger.info(f"[Batch {batch_num}] Completed\n")
                batch_num += 1
                
                # 배치 간 대기
                batch_delay = self.config.get("batch_delay", 5)
                if batch_delay > 0:
                    self.logger.info(f"Waiting {batch_delay}s...")
                    import time
                    time.sleep(batch_delay)
                    
            # 3. 최종 통계
            self.print_final_stats()
            
        except KeyboardInterrupt:
            self.logger.warning("\n[Warning] User interrupted")
            self.print_final_stats()
            
        except Exception as e:
            self.logger.error(f"[Critical Error] {e}", exc_info=True)
            
        finally:
            self.cleanup()

def main():
    crawler = NRICHCrawler()
    crawler.run()

if __name__ == "__main__":
    main()
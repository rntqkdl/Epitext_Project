"""
Kyujanggak Institute for Korean Studies Crawler
======================================================================
목적: 규장각 한국학연구원 고문서 데이터베이스 수집 (Selenium 기반)
URL: https://kyudb.snu.ac.kr
작성자: Epitext Project Team
======================================================================
"""

import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

import sys

# 상위 경로 설정
sys.path.append(str(Path(__file__).parent.parent))
from config import KYUDB_CONFIG
from crawlers.base_crawler import BaseCrawler

class KyudbCrawler(BaseCrawler):
    """규장각 크롤러 (Selenium을 활용한 동적 웹페이지 수집)"""
    
    def __init__(self, config: Dict[str, Any] = KYUDB_CONFIG):
        super().__init__(source="kyudb", config=config)
        
        # Selenium 드라이버 및 세션 상태
        self.driver = None
        self.selenium_session = None
        
        # 현재 처리 중인 상태 정보
        self.current_period = None
        self.current_cate_code = None
        self.current_page = 1
        
        # 중복 방지를 위한 처리 완료 집합
        self.done_items = set()
    
    # ======================================================================
    # Selenium 드라이버 관리 및 설정
    # ======================================================================
    
    def _create_driver(self) -> webdriver.Chrome:
        """Chrome 드라이버 생성 및 옵션 설정"""
        opts = ChromeOptions()
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1400,1000")
        opts.add_argument("--lang=ko-KR,ko")
        
        # 자동화 탐지 방지
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option('useAutomationExtension', False)
        
        # 다운로드 경로 및 팝업 설정
        prefs = {
            "download.prompt_for_download": False,
            "profile.default_content_setting_values.automatic_downloads": 1,
            "profile.default_content_settings.popups": 0
        }
        opts.add_experimental_option("prefs", prefs)
        
        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(self.config["page_load_timeout"])
        
        self.logger.info("[Driver] Chrome driver initialized.")
        return driver
    
    def _selenium_to_requests_session(self):
        """Selenium의 쿠키와 세션 정보를 requests 세션으로 동기화"""
        if not self.driver:
            return
        
        # 쿠키 복사
        for cookie in self.driver.get_cookies():
            self.session.cookies.set(
                cookie["name"],
                cookie["value"],
                domain=cookie.get("domain", "")
            )
        
        # User-Agent 동기화
        try:
            ua = self.driver.execute_script("return navigator.userAgent;")
            self.session.headers.update({"User-Agent": ua})
        except Exception:
            pass
    
    def _drain_alerts(self, wait_sec: float = 0.2):
        """화면에 뜬 모든 경고창(Alert)을 자동으로 닫음"""
        end_time = time.time() + wait_sec
        while True:
            try:
                alert = Alert(self.driver)
                text = alert.text
                alert.accept()
                self.logger.debug(f"[Alert] Dismissed: {text}")
            except Exception:
                pass
            
            if time.time() >= end_time:
                break
    
    # ======================================================================
    # 페이지 네비게이션 (Navigation)
    # ======================================================================
    
    def _navigate_to_home(self) -> bool:
        """홈페이지 접속 및 초기화"""
        try:
            self.driver.get(self.config["home_url"])
            WebDriverWait(self.driver, 12).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
            )
            self.logger.info(f"[Nav] Home URL accessed: {self.config['home_url']}")
            
            # 세션 정보 동기화
            self._selenium_to_requests_session()
            return True
        except Exception as e:
            self.logger.error(f"[Error] Failed to access home: {e}")
            return False
    
    def _open_period(self, period_name: str, cate_code: str) -> bool:
        """특정 연대 카테고리 페이지로 이동"""
        self.current_period = period_name
        self.current_cate_code = cate_code
        self.current_page = 1
        
        self.logger.info("============================================================")
        self.logger.info(f" Accessing Period: {period_name} ({cate_code})")
        self.logger.info("============================================================")
        
        # 1. 자바스크립트 함수 호출 방식 시도
        try:
            self.driver.execute_script(
                f"fn_bookSelectRequest(4, '{cate_code}', '');"
            )
            WebDriverWait(self.driver, 8).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.list_tbl ul"))
            )
            self.logger.info("  [Success] Period switched via JS.")
            return True
        except Exception:
            self.logger.warning("  [Warning] JS switching failed. Trying direct URL.")
        
        # 2. URL 직접 이동 방식 (Fallback)
        ts = int(time.time() * 1000)
        for mid in ["GSD", "GDS"]:
            url = self.config["list_url_template"].format(
                mid=mid, cate=cate_code, ts=ts
            )
            try:
                self.driver.get(url)
                WebDriverWait(self.driver, 8).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.list_tbl ul"))
                )
                self.logger.info(f"  [Success] Period switched via URL ({mid}).")
                return True
            except Exception as e:
                self.logger.debug(f"    [Fail] {mid} URL failed: {e}")
        
        self.logger.error("  [Error] Failed to switch period.")
        return False
    
    def _goto_page(self, page: int) -> bool:
        """특정 페이지 번호로 이동"""
        prev_signature = self._get_page_signature()
        
        try:
            self.driver.execute_script(f"fn_bookSearchResultNavi('{page}');")
        except Exception as e:
            self.logger.warning(f"  [Warning] Page nav JS failed: {e}")
            return False
        
        # 페이지 내용 변경 대기
        timeout = 8
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            self._drain_alerts(0.2)
            
            cur_page, _ = self._get_paging_info()
            if cur_page == page:
                cur_signature = self._get_page_signature()
                if cur_signature != prev_signature:
                    self.logger.info(f"  [Nav] Moved to page {page}")
                    self.current_page = page
                    return True
            
            time.sleep(0.3)
        
        self.logger.warning(f"  [Timeout] Failed to move to page {page}")
        return False
    
    def _get_paging_info(self) -> Tuple[int, int]:
        """현재 페이지 번호와 전체 페이지 수 파싱"""
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        
        current_page = 1
        on_element = soup.select_one("div.paging a.on")
        if on_element:
            try:
                current_page = int(on_element.get_text(strip=True))
            except ValueError:
                pass
        
        max_page = current_page
        for a in soup.select("div.paging a[onclick*='fn_bookSearchResultNavi']"):
            match = re.search(r"fn_bookSearchResultNavi\('(\d+)'\)", a.get("onclick", ""))
            if match:
                page_num = int(match.group(1))
                if page_num > max_page:
                    max_page = page_num
        
        return current_page, max_page
    
    def _get_page_signature(self) -> str:
        """현재 페이지 내용의 고유 시그니처 생성 (변경 감지용)"""
        try:
            ul_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.list_tbl ul")
            if not ul_elements:
                return ""
            
            soup = BeautifulSoup(ul_elements[0].get_attribute("outerHTML"), "html.parser")
            item_ids = []
            
            for strong in soup.select("strong[onclick*='fn_bookSearchResultView']"):
                onclick = strong.get("onclick", "")
                match = re.search(
                    r"fn_bookSearchResultView\(\s*'\d+'\s*,\s*'([A-Z0-9_]+)'\s*,\s*'master'",
                    onclick
                )
                if match:
                    item_ids.append(match.group(1))
            
            return ";".join(item_ids)
        except Exception:
            return ""
    
    # ======================================================================
    # 목록 및 메타데이터 파싱
    # ======================================================================
    
    def _list_items_on_page(self) -> List[Dict[str, str]]:
        """현재 페이지에 노출된 항목 리스트 추출"""
        items = []
        
        try:
            ul_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.list_tbl ul")
            if not ul_elements:
                return items
            
            soup = BeautifulSoup(ul_elements[0].get_attribute("outerHTML"), "html.parser")
            
            for strong in soup.select("strong[onclick*='fn_bookSearchResultView']"):
                onclick = strong.get("onclick", "")
                match = re.search(
                    r"fn_bookSearchResultView\(\s*'(\d+)'\s*,\s*'([A-Z0-9_]+)'\s*,\s*'master'",
                    onclick
                )
                
                # strong 태그 실패 시 하위 span 태그 확인
                if not match:
                    span = strong.find_next("span")
                    if span:
                        onclick2 = span.get("onclick", "")
                        match = re.search(
                            r"fn_bookSearchResultView\(\s*'(\d+)'\s*,\s*'([A-Z0-9_]+)'\s*,\s*'master'",
                            onclick2
                        )
                
                if match:
                    idx = match.group(1)
                    item_cd = match.group(2)
                    items.append({"idx": idx, "item_cd": item_cd})
        
        except Exception as e:
            self.logger.warning(f"  [Warning] Item list parsing error: {e}")
        
        return items
    
    def _open_item_detail(self, idx: str, item_cd: str) -> bool:
        """상세 페이지로 이동 (JS 실행)"""
        try:
            self.driver.execute_script(
                f"fn_bookSearchResultView('{idx}', '{item_cd}', 'master');"
            )
            time.sleep(0.8)
            self.logger.debug(f"    [Detail] Opened item {item_cd}")
            return True
        except Exception as e:
            self.logger.warning(f"    [Error] Failed to open detail: {e}")
            return False
    
    def _extract_metadata(self) -> List[Tuple[str, str]]:
        """상세 페이지의 테이블에서 메타데이터 추출"""
        try:
            container = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.detail_info table.zoom_area"))
            )
        except TimeoutException:
            self.logger.warning("    [Warning] Metadata table not found.")
            return []
        
        soup = BeautifulSoup(container.get_attribute("outerHTML"), "html.parser")
        rows = []
        
        for tr in soup.select("tbody tr"):
            th = tr.find("th")
            td = tr.find("td")
            
            if th and td:
                key = th.get_text(strip=True).replace("\xa0", " ")
                value = td.get_text("\n", strip=True).replace("\xa0", " ")
                rows.append((key, value))
        
        return rows
    
    def _parse_metadata_to_dict(self, rows: List[Tuple[str, str]]) -> Dict[str, str]:
        """메타데이터 리스트를 딕셔너리로 변환"""
        metadata = {
            "title": None,
            "period": None,
            "script_style": None
        }
        
        for key, value in rows:
            key_lower = key.lower()
            
            if "표제" in key or "title" in key_lower:
                metadata["title"] = value
            elif "시대" in key or "period" in key_lower:
                metadata["period"] = value
            elif "서체" in key or "script" in key_lower:
                metadata["script_style"] = value
        
        return metadata
    
    def _save_metadata_csv(self, item_cd: str, rows: List[Tuple[str, str]]) -> bool:
        """메타데이터를 개별 CSV 파일로 저장"""
        if not rows:
            return False
        
        period_dir = self.fh.get_source_dir("kyudb") / self.current_period
        item_dir = period_dir / item_cd
        self.fh.ensure_dir(item_dir)
        
        csv_path = item_dir / self.config["metadata_filename"]
        
        success = self.fh.save_metadata_csv(
            data=rows,
            savepath=csv_path,
            headers=["field", "value"]
        )
        
        if success:
            self.logger.debug(f"    [File] Metadata CSV saved ({len(rows)} rows)")
        
        return success
    
    # ======================================================================
    # 이미지 뷰어 및 다운로드 처리
    # ======================================================================
    
    def _open_viewer(self) -> bool:
        """원문 이미지 뷰어 팝업 열기"""
        candidates = [
            (By.CSS_SELECTOR, "div.btn_img_txt a[onclick*='fn_originalImg']"),
            (By.CSS_SELECTOR, "a[onclick*='fn_originalImg(']"),
            (By.XPATH, "//a[contains(.,'원문 이미지') or contains(.,'원문이미지')]"),
        ]
        
        for by, selector in candidates:
            try:
                element = WebDriverWait(self.driver, 6).until(
                    EC.element_to_be_clickable((by, selector))
                )
                self.driver.execute_script("arguments[0].click();", element)
                time.sleep(0.6)
                self._drain_alerts(0.2)
                self.logger.debug("    [Viewer] Opened successfully.")
                return True
            except Exception:
                continue
        
        self.logger.warning("    [Warning] Failed to open viewer.")
        return False
    
    def _get_viewer_handle(self) -> Optional[str]:
        """뷰어 윈도우 핸들 찾기"""
        try:
            current = self.driver.current_window_handle
        except WebDriverException:
            return None
        
        # 역순으로 탐색 (최신 창 우선)
        for handle in reversed(self.driver.window_handles):
            try:
                self.driver.switch_to.window(handle)
                # 뷰어 특정 요소 확인
                if self.driver.find_elements(
                    By.CSS_SELECTOR,
                    "span.img-down.btn-left-down, div.img-area img"
                ):
                    return handle
            except Exception:
                continue
        
        # 원래 창으로 복귀
        try:
            self.driver.switch_to.window(current)
            return current
        except Exception:
            return None
    
    def _parse_viewer_download_info(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """뷰어 페이지에서 다운로드 정보 및 다음 페이지 토큰 파싱"""
        html = self.driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        img_file_nm = None
        path = None
        next_token = None
        
        # 다운로드 버튼 파싱
        dl_span = soup.select_one("span.img-down.btn-left-down")
        if dl_span and dl_span.has_attr("onclick"):
            match = re.search(r"/ImageDown\.do\?([^']+)'", dl_span["onclick"])
            if match:
                from urllib.parse import parse_qs
                query = parse_qs(match.group(1))
                img_file_nm = (query.get("imgFileNm", [None])[0] or "").strip()
                path = (query.get("path", [None])[0] or "").strip()
        
        # 다음 페이지 버튼 파싱
        a_next = soup.select_one("a.btn_st.next")
        if a_next and a_next.has_attr("onclick"):
            match2 = re.search(
                r"fn_goPageJump\('([0-9a-z]{4})'\)",
                a_next["onclick"],
                re.IGNORECASE
            )
            if match2:
                next_token = match2.group(1)
        
        return img_file_nm, path, next_token
    
    def _download_viewer_image(
        self,
        img_file_nm: str,
        path: str,
        item_dir: Path,
        seen: set
    ) -> bool:
        """뷰어의 현재 이미지 다운로드"""
        filename = img_file_nm
        
        if filename in seen:
            self.logger.debug(f"      [Skip] Duplicate: {filename}")
            return False
        
        save_path = item_dir / filename
        
        if self.fh.file_exists(save_path):
            self.logger.debug(f"      [Skip] File exists: {filename}")
            seen.add(filename)
            return False
        
        # 다운로드 URL 구성
        query_str = urllib.parse.urlencode({"imgFileNm": img_file_nm, "path": path})
        url = f"{self.config['image_download_url']}?{query_str}"
        
        referer = self.driver.current_url
        headers = self.session.headers.copy()
        headers["Referer"] = referer
        
        try:
            response = self.session.get(url, headers=headers, timeout=25, stream=True)
            response.raise_for_status()
            
            # Content-Type 검증
            content_type = response.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/"):
                self.logger.warning(f"      [Warning] Invalid content-type: {content_type}")
                return False
            
            data = b"".join(response.iter_content(chunk_size=8192))
            
            if len(data) < 256:
                self.logger.warning(f"      [Warning] File too small: {len(data)}B")
                return False
            
            # 파일 저장
            with open(save_path, "wb") as f:
                f.write(data)
            
            seen.add(filename)
            self.stats["images_downloaded"] += 1
            self.logger.debug(f"      [Saved] {filename} ({len(data):,}B)")
            return True
        
        except Exception as e:
            self.logger.warning(f"      [Error] Download failed: {filename} - {e}")
            return False
    
    def _click_next_in_viewer(self, next_token: Optional[str]) -> bool:
        """뷰어에서 다음 페이지로 이동"""
        # 1. 버튼 클릭 시도
        try:
            btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn_st.next"))
            )
            self.driver.execute_script("arguments[0].click();", btn)
            return True
        except Exception:
            pass
        
        # 2. JS 함수 호출 시도
        if next_token:
            try:
                self.driver.execute_script(f"fn_goPageJump('{next_token}');")
                return True
            except Exception as e:
                self.logger.debug(f"      [Debug] JS nav failed: {e}")
        
        return False
    
    def _get_viewer_page_marker(self) -> str:
        """뷰어 페이지 변경 감지를 위한 마커 추출"""
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        
        # 다운로드 버튼에서 파일명 추출
        dl_span = soup.select_one("span.img-down.btn-left-down")
        if dl_span and dl_span.has_attr("onclick"):
            match = re.search(r"imgFileNm=([^&']+)", dl_span["onclick"])
            if match:
                fname = urllib.parse.unquote(match.group(1))
                token_match = re.search(r"_(\d{4}_[0-9a-z]{4})", fname, re.IGNORECASE)
                return token_match.group(1) if token_match else fname
        
        # 이미지 태그에서 src 추출
        img = soup.select_one("img#viewer_img, img#imgView, div.img-area img")
        if img and img.has_attr("src"):
            src = img["src"]
            token_match = re.search(r"_(\d{4}_[0-9a-z]{4})", src, re.IGNORECASE)
            return token_match.group(1) if token_match else src
        
        return str(int(time.time() * 1000))
    
    def _wait_viewer_page_change(self, prev_marker: str, timeout: int = 12) -> Tuple[bool, bool]:
        """페이지 변경 대기 및 마지막 페이지 여부 확인"""
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            # 알럿(마지막 페이지 안내) 체크
            try:
                alert = Alert(self.driver)
                text = alert.text
                alert.accept()
                self.logger.debug(f"      [Alert] {text}")
                if "마지막" in text:
                    return False, True
            except Exception:
                pass
            
            # 마커 변경 체크
            cur_marker = self._get_viewer_page_marker()
            if cur_marker and cur_marker != prev_marker:
                return True, False
            
            time.sleep(0.3)
        
        return False, False  # 타임아웃
    
    def _download_all_viewer_images(self, doc_id: str, item_cd: str) -> int:
        """뷰어 내의 모든 이미지를 순회하며 다운로드"""
        viewer_handle = self._get_viewer_handle()
        if not viewer_handle:
            self.logger.warning("    [Warning] Viewer handle not found.")
            return 0
        
        period_dir = self.fh.get_source_dir("kyudb") / self.current_period
        item_dir = period_dir / item_cd
        self.fh.ensure_dir(item_dir)
        
        seen = set()
        total = 0
        prev_marker = self._get_viewer_page_marker()
        
        while True:
            # 1. 현재 이미지 다운로드
            img_file_nm, path, next_token = self._parse_viewer_download_info()
            
            if img_file_nm and path:
                if self._download_viewer_image(img_file_nm, path, item_dir, seen):
                    total += 1
            else:
                self.logger.warning("      [Warning] Failed to parse image info.")
            
            # 2. 다음 페이지 이동
            if not self._click_next_in_viewer(next_token):
                self.logger.debug("      [Info] No next button. Finished.")
                break
            
            # 3. 페이지 변경 대기
            changed, is_last = self._wait_viewer_page_change(prev_marker, timeout=12)
            
            if is_last:
                self.logger.debug("      [Info] Reached last page.")
                break
            
            if not changed:
                self.logger.debug("      [Info] Page not changed. Finished.")
                break
            
            prev_marker = self._get_viewer_page_marker()
        
        # 뷰어 닫기
        try:
            self.driver.close()
        except Exception:
            pass
        
        # 메인 창으로 포커스 복귀
        try:
            if self.driver.window_handles:
                self.driver.switch_to.window(self.driver.window_handles[0])
        except Exception:
            pass
        
        return total
    
    def _return_to_list(self):
        """상세 페이지에서 목록으로 복귀"""
        # 팝업 정리
        try:
            handles = self.driver.window_handles
            if len(handles) > 1:
                for h in handles[1:]:
                    try:
                        self.driver.switch_to.window(h)
                        self.driver.close()
                    except Exception:
                        pass
                self.driver.switch_to.window(handles[0])
        except Exception:
            pass
        
        # 뒤로가기 실행
        try:
            cur_url = self.driver.current_url or ""
            if "book/view.do" in cur_url:
                self.driver.execute_script("window.history.back();")
                time.sleep(0.5)
        except Exception:
            pass
        
        # 목록 페이지 로드 대기
        try:
            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.list_tbl ul"))
            )
        except TimeoutException:
            # 실패 시 목록 재진입
            self._open_period(self.current_period, self.current_cate_code)
        
        # 페이지 번호 복구
        cur_page, _ = self._get_paging_info()
        if cur_page != self.current_page:
            self._goto_page(self.current_page)
    
    # ======================================================================
    # BaseCrawler 추상 메서드 구현
    # ======================================================================
    
    def fetch_item_list(self) -> List[Dict[str, Any]]:
        """전체 수집 항목 목록 생성"""
        all_items = []
        
        self.driver = self._create_driver()
        
        if not self._navigate_to_home():
            return all_items
        
        for period_name, cate_code in self.config["periods"]:
            if not self._open_period(period_name, cate_code):
                continue
            
            cur_page, max_page = self._get_paging_info()
            self.logger.info(f"  Total Pages: {max_page}")
            
            for page in range(1, max_page + 1):
                if page > 1:
                    if not self._goto_page(page):
                        continue
                
                items = self._list_items_on_page()
                self.logger.info(f"  [Page {page}/{max_page}] Found {len(items)} items")
                
                for item in items:
                    all_items.append({
                        "original_id": item["item_cd"],
                        "idx": item["idx"],
                        "period": period_name,
                        "cate_code": cate_code,
                        "page": page
                    })
        
        return all_items
    
    def scrape_item(self, item_info: Dict[str, Any]) -> bool:
        """단일 항목 스크래핑"""
        item_cd = item_info["original_id"]
        idx = item_info["idx"]
        
        doc_id = self.generate_doc_id(item_cd)
        
        if doc_id in self.done_items:
            self.logger.info(f"  [Skip] Already scraped: {item_cd}")
            self.stats["skipped"] += 1
            return True
        
        try:
            # 1. 상세 페이지 열기
            if not self._open_item_detail(idx, item_cd):
                self.logger.warning(f"  [Fail] Detail open failed: {item_cd}")
                self.update_status(doc_id, "error", "Detail open failed")
                return False
            
            # 2. 메타데이터 파싱
            metadata_rows = self._extract_metadata()
            metadata_dict = self._parse_metadata_to_dict(metadata_rows)
            
            # 3. CSV 저장
            self._save_metadata_csv(item_cd, metadata_rows)
            
            # 4. DB 저장
            self.save_document_metadata(
                doc_id=doc_id,
                original_id=item_cd,
                title=metadata_dict.get("title"),
                period=metadata_dict.get("period") or item_info.get("period"),
                script_style=metadata_dict.get("script_style"),
                metadata={"cate_code": item_info.get("cate_code")}
            )
            
            # 5. 이미지 다운로드
            image_count = 0
            if self._open_viewer():
                try:
                    image_count = self._download_all_viewer_images(doc_id, item_cd)
                    self.logger.info(f"  [Done] {item_cd} (Images: {image_count})")
                except Exception as e:
                    self.logger.error(f"  [Error] Image download error: {e}")
            else:
                self.logger.warning(f"  [Warning] Viewer open failed: {item_cd}")
            
            # 6. 목록 복귀
            self._return_to_list()
            
            # 7. 완료 처리
            self.update_status(doc_id, "completed")
            self.done_items.add(doc_id)
            return True
        
        except Exception as e:
            self.logger.error(f"  [Critical] Item scrape failed: {item_cd} - {e}", exc_info=True)
            self.update_status(doc_id, "error", str(e))
            
            # 복구 시도
            try:
                self._return_to_list()
            except Exception:
                pass
            
            return False
    
    def cleanup(self):
        """리소스 정리"""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("[Driver] Closed successfully.")
            except Exception:
                pass
        super().cleanup()

def main():
    crawler = KyudbCrawler()
    crawler.run()

if __name__ == "__main__":
    main()
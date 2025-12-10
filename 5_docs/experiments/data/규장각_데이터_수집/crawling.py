# -*- coding: utf-8 -*-
import os
import re
import csv
import io
import sys
import time
import urllib.parse
from datetime import datetime
from typing import Optional, List, Tuple

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# --------------------- 사이트 상수 ---------------------
BASE = "https://kyudb.snu.ac.kr"
HOME = f"{BASE}/main.do?mid=GSD&submain=Y"
LIST_URL = f"{BASE}/book/list.do"      # ?mid=GSD&book_cate=GSD0403
VIEW_URL = f"{BASE}/book/view.do"      # ?mid=GSD&item_cd=...
# 목록으로 복귀 시 더 안정적인 mid 조합 (네가 준 HTML 기준)
LIST_URL_BACK = f"{BASE}/book/list.do?mid=GDS&book_cate={{cate}}"

# ------------ 여기만 800년대로 제한 (필요시 바꿔서 실행) ------------
PERIODS = [
    ("800년대", "GSD0403"),
]

# --------------------- 유틸/로깅 ---------------------
class Progress:
    def __init__(self):
        self.period = None
        self.item = None
        self.img_ok = 0
        self.img_fail = 0

    def ts(self):
        return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    def log(self, msg):
        print(f"{self.ts()} {msg}")
        sys.stdout.flush()

    def set_period(self, p): self.period = p
    def set_item(self, i): self.item = i
    def inc_image_ok(self): self.img_ok += 1
    def inc_fail(self): self.img_fail += 1


def makedirs(p):
    os.makedirs(p, exist_ok=True)


def now_ms():
    return int(time.time() * 1000)


# --------------------- 드라이버/세션 ---------------------
def make_driver() -> webdriver.Chrome:
    opts = ChromeOptions()
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument("--lang=ko-KR,ko")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    prefs = {
        "download.prompt_for_download": False,
        "profile.default_content_setting_values.automatic_downloads": 1,
    }
    opts.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(30)
    return driver


def selenium_cookies_to_requests(driver) -> requests.Session:
    sess = requests.Session()
    for c in driver.get_cookies():
        sess.cookies.set(c['name'], c['value'], domain=c.get('domain', ''))
    # 사이트가 UA/Referer에 민감할 수 있으므로 맞춰줌
    ua = driver.execute_script("return navigator.userAgent;") or \
         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/141 Safari/537.36"
    sess.headers.update({
        "User-Agent": ua,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    })
    return sess


# --------------------- 다운로드 ---------------------
def _robust_download(sess: requests.Session, url: str, dst_path: str,
                     referer: Optional[str], reauth, max_retry=3, min_bytes=256) -> Tuple[bool, str]:
    last_err = ""
    for _ in range(max_retry):
        try:
            headers = {"Referer": referer or BASE}
            r = sess.get(url, headers=headers, timeout=25, stream=True)
            ctype = (r.headers.get("Content-Type") or "").lower()

            if (not ctype.startswith("image/")) and (not url.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))):
                last_err = f"bad ctype {ctype}"
                reauth()
                time.sleep(0.5)
                continue

            b = io.BytesIO()
            sz = 0
            for chunk in r.iter_content(8192):
                if not chunk:
                    continue
                b.write(chunk)
                sz += len(chunk)

            if sz < min_bytes:
                last_err = f"too small {sz}B"
                reauth()
                time.sleep(0.5)
                continue

            with open(dst_path, "wb") as f:
                f.write(b.getvalue())

            return True, f"{sz}B"
        except Exception as e:
            last_err = str(e)
            reauth()
            time.sleep(0.6)
    return False, last_err


# --------------------- 뷰어 헬퍼 ---------------------
def drain_alerts(driver, prog: Progress, wait_sec=0.0):
    end = time.time() + wait_sec
    while True:
        try:
            al = Alert(driver)
            txt = al.text
            al.accept()
            prog.log(f"[alert] {txt}")
        except Exception:
            pass
        if wait_sec == 0.0 or time.time() >= end:
            break


def get_viewer_handle(driver) -> Optional[str]:
    """이미지 뷰어 창(팝업) 핸들 탐색."""
    cur = driver.current_window_handle
    for h in reversed(driver.window_handles):
        try:
            driver.switch_to.window(h)
            if driver.find_elements(By.CSS_SELECTOR, "span.img-down.btn-left-down, div.img-area img, img#image, img#viewer_img"):
                return h
        except Exception:
            continue
    try:
        driver.switch_to.window(cur)
        return cur
    except Exception:
        return None


def get_download_onclick(driver) -> str:
    """'이미지다운로드' 버튼의 onclick 원문. (장 구분 시그니처)"""
    for sel in ["span.img-down.btn-left-down", "span.img-down", "span.btn-left-down"]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            onclick = el.get_attribute("onclick") or ""
            if onclick:
                return onclick
        except Exception:
            continue
    return ""


def parse_imagedown_from_onclick(onclick: str) -> Tuple[Optional[str], Optional[str]]:
    """
    onclick="location.href='/ImageDown.do?imgFileNm=XXX.jpg&path=/data01/...';"
    → (imgFileNm, path)
    """
    m = re.search(r"/ImageDown\.do\?([^']+)'", onclick)
    if not m:
        return None, None
    q = urllib.parse.parse_qs(m.group(1))
    imgFileNm = (q.get("imgFileNm", [None])[0] or "").strip()
    path = (q.get("path", [None])[0] or "").strip()
    return imgFileNm, path


def build_imagedown_url(imgFileNm: str, path: str) -> str:
    qs = urllib.parse.urlencode({"imgFileNm": imgFileNm, "path": path})
    return urllib.parse.urljoin(BASE, f"/ImageDown.do?{qs}")


def wait_page_flip_or_last(driver, prog: Progress, prev_onclick: str,
                           timeout=12, poll=0.25) -> Tuple[bool, bool]:
    """
    다운로드 버튼의 onclick 값이 달라질 때까지 대기.
    마지막 장 알림 시 (changed=False, is_last=True).
    """
    end = time.time() + timeout
    while time.time() < end:
        try:
            al = Alert(driver)
            txt = al.text
            al.accept()
            prog.log(f"  [alert] {txt}")
            if "마지막" in txt:
                return False, True
        except Exception:
            pass

        cur = get_download_onclick(driver)
        if cur and cur != prev_onclick:
            return True, False

        time.sleep(poll)

    return False, False


def click_next_in_viewer(driver, prog: Progress) -> bool:
    """다음 페이지 이동: 버튼 → JS 토큰 순서."""
    # 1) 버튼
    try:
        btn = WebDriverWait(driver, 4).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn_st.next, a.btn_next, a.next"))
        )
        driver.execute_script("arguments[0].click();", btn)
        return True
    except Exception:
        pass
    # 2) HTML에서 fn_goPageJump 토큰 파싱 후 JS 호출
    html = driver.page_source
    m = re.search(r"fn_goPageJump\('([0-9a-z]{4})'\)", html, flags=re.I)
    if m:
        try:
            driver.execute_script(f"fn_goPageJump('{m.group(1)}');")
            return True
        except Exception as e:
            prog.log(f"  [next] fn_goPageJump 실패: {e}")
    return False


def download_current_page(sess: requests.Session, driver, out_dir_item: str,
                          prog: Progress) -> int:
    """현재 장의 다운로드 버튼 onclick에서 URL을 만들어 저장."""
    onclick = get_download_onclick(driver)
    if not onclick:
        prog.log("  [warn] 다운로드 버튼 미탐")
        return 0

    imgFileNm, path = parse_imagedown_from_onclick(onclick)
    if not imgFileNm or not path:
        prog.log("  [warn] onclick 파싱 실패")
        return 0

    dst = os.path.join(out_dir_item, imgFileNm)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        prog.log(f"  [skip] 이미 존재: {imgFileNm}")
        return 0
    if os.path.exists(dst) and os.path.getsize(dst) == 0:
        try:
            os.remove(dst)
        except Exception:
            pass

    url = build_imagedown_url(imgFileNm, path)

    def reauth():
        new_sess = selenium_cookies_to_requests(driver)
        sess.cookies = new_sess.cookies

    ok, info = _robust_download(sess, url, dst, referer=driver.current_url, reauth=reauth, max_retry=3, min_bytes=256)
    if ok:
        prog.inc_image_ok()
        prog.log(f"  [ok] 이미지 저장: {imgFileNm} ({info})")
        return 1
    else:
        prog.inc_fail()
        prog.log(f"  [err] 이미지 실패: {imgFileNm} ({info})")
        return 0


def viewer_download_all_pages(driver, sess: requests.Session, out_dir_item: str,
                              prog: Progress) -> int:
    """뷰어에서 모든 장을 저장하고 창을 닫음."""
    vhandle = get_viewer_handle(driver)
    if not vhandle:
        prog.log("  [viewer] 핸들 없음 → 스킵")
        return 0
    driver.switch_to.window(vhandle)

    # 최초 시그니처
    prev_sig = get_download_onclick(driver)
    if not prev_sig:
        prog.log("  [viewer] 다운로드 버튼 없음 → 종료")
        try:
            driver.close()
        except Exception:
            pass
        return 0

    total = 0
    while True:
        # 1) 현재 장 저장
        total += download_current_page(sess, driver, out_dir_item, prog)

        # 2) 다음 이동
        moved = click_next_in_viewer(driver, prog)
        if not moved:
            prog.log("  [viewer] 다음 이동 실패 → 종료 추정")
            break

        # 3) 변화 대기(=onclick 변경) 또는 마지막 장 알림
        changed, is_last = wait_page_flip_or_last(driver, prog, prev_sig, timeout=12, poll=0.25)
        if is_last:
            prog.log("  [viewer] 마지막 장 → 닫기")
            break
        if not changed:
            prog.log("  [viewer] 장 변화 없음(타임아웃) → 종료 추정")
            break

        # 4) 다음 루프 대비 시그니처 갱신
        cur_sig = get_download_onclick(driver)
        prev_sig = cur_sig if cur_sig else prev_sig

    # 뷰어 닫고 메인으로
    try:
        driver.close()
    except Exception:
        pass
    try:
        if driver.window_handles:
            driver.switch_to.window(driver.window_handles[0])
    except Exception:
        pass

    return total


# --------------------- 상세 페이지/CSV ---------------------
def extract_table_csv_by_block(driver, css_block: str, out_csv: str, prog: Progress) -> int:
    """특정 블록(css_block) 아래의 table.zoom_area 에서 key/value 추출."""
    try:
        block = WebDriverWait(driver, 8).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, f"{css_block} table.zoom_area"))
        )
    except TimeoutException:
        prog.log(f"  [warn] {css_block} table.zoom_area 미발견 → 빈 CSV")
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(["field", "value"])
        return 0

    html = block.get_attribute("outerHTML")
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("tbody tr"):
        th = tr.find("th")
        td = tr.find("td")
        if not th or not td:
            continue
        key = th.get_text(strip=True).replace("\xa0", " ")
        val = td.get_text("\n", strip=True).replace("\xa0", " ")
        rows.append((key, val))

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "value"])
        for k, v in rows:
            w.writerow([k, v])

    prog.log(f"  [csv] 저장 → {out_csv} ({len(rows)} rows)")
    return len(rows)


def extract_detail_csvs(driver, out_dir_item: str, prog: Progress):
    """상세서지(detail_info) + info_table 두 개의 CSV를 모두 저장."""
    makedirs(out_dir_item)
    n1 = extract_table_csv_by_block(driver, "div.detail_info", os.path.join(out_dir_item, "detail_info.csv"), prog)
    n2 = extract_table_csv_by_block(driver, "div.info_table",   os.path.join(out_dir_item, "info_table.csv"), prog)
    return n1 + n2


# --------------------- 목록/상세 진입 ---------------------
def open_period(driver, cate_code: str, prog: Progress) -> bool:
    """연대 전환: JS → URL 순."""
    drain_alerts(driver, prog)
    try:
        driver.execute_script(f"fn_bookSelectRequest(4,'{cate_code}','');")
        WebDriverWait(driver, 8).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.list_tbl ul"))
        )
        prog.log("[period] 전환 성공(JS)")
        return True
    except Exception:
        prog.log("[period] JS호출 실패, URL로 시도")

    url = f"{LIST_URL}?mid=GSD&book_cate={cate_code}&_ts={now_ms()}"
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.list_tbl ul"))
        )
        prog.log("[period] 전환 성공(URL)")
        return True
    except Exception as e:
        prog.log(f"[period] 전환 실패: {e}")
        return False


def list_items_on_page(driver) -> List[Tuple[str, str]]:
    """
    목록 페이지에서 (item_cd, idx) 추출.
    strong/span 의 onclick="fn_bookSearchResultView('idx','ITEM','master')" 파싱.
    """
    items = []
    blocks = driver.find_elements(By.CSS_SELECTOR, "div.list_tbl ul")
    if not blocks:
        return items
    html = blocks[0].get_attribute("outerHTML")
    soup = BeautifulSoup(html, "html.parser")
    for strong in soup.select("strong[onclick*='fn_bookSearchResultView']"):
        on = strong.get("onclick") or ""
        m = re.search(r"fn_bookSearchResultView\(\s*'(\d+)'\s*,\s*'([A-Z0-9_]+)'\s*,\s*'master'", on)
        if not m:
            span = strong.find_next("span")
            if span:
                on2 = span.get("onclick") or ""
                m = re.search(r"fn_bookSearchResultView\(\s*'(\d+)'\s*,\s*'([A-Z0-9_]+)'\s*,\s*'master'", on2)
        if m:
            idx, item = m.group(1), m.group(2)
            items.append((item, idx))
    return items


def click_item_detail_inline(driver, idx: str, item_cd: str, prog: Progress) -> bool:
    """인라인 상세 열기(JS). 실패 시 False."""
    try:
        driver.execute_script(f"fn_bookSearchResultView('{idx}','{item_cd}','master');")
        time.sleep(0.7)
        # 상세 블록 존재 확인(너무 강하게 기다리면 느릴 수 있어 약하게 체크)
        return True
    except Exception as e:
        prog.log(f"[detail] 인라인 호출 실패: {e}")
        return False


def goto_item_view_page(driver, item_cd: str, prog: Progress) -> bool:
    """뷰 페이지로 직접 이동 (백업 경로)."""
    for mid in ("GSD", "GDS"):
        url = f"{VIEW_URL}?mid={mid}&item_cd={item_cd}"
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div#content"))
            )
            prog.log(f"[detail] 직접 이동 → {url}")
            return True
        except Exception as e:
            prog.log(f"[detail] 직접 이동 실패({mid}): {e}")
    return False


def open_viewer(driver, prog: Progress) -> bool:
    """
    상세에서 '원문이미지' 열기 (여러 셀렉터 후보 시도).
    """
    candidates = [
        (By.CSS_SELECTOR, "div.btn_img_txt a[onclick*='fn_originalImg']"),
        (By.CSS_SELECTOR, "a[onclick*='fn_originalImg']"),
        (By.XPATH, "//a[img[contains(@alt,'원문') or contains(@alt,'이미지')]]"),
        (By.XPATH, "//a[contains(.,'원문 이미지') or contains(.,'원문이미지')]"),
    ]
    for by, sel in candidates:
        try:
            el = WebDriverWait(driver, 6).until(EC.element_to_be_clickable((by, sel)))
            driver.execute_script("arguments[0].click();", el)
            time.sleep(0.6)
            prog.log("  [viewer] 뷰어 오픈")
            drain_alerts(driver, prog, 0.2)
            return True
        except Exception:
            continue
    return False


def next_list_page(driver, prog: Progress) -> bool:
    """목록 하단의 '다음' 버튼."""
    try:
        nxt = driver.find_element(By.CSS_SELECTOR, "a.btn_next, a.next")
        driver.execute_script("arguments[0].click();", nxt)
        time.sleep(0.6)
        # 새 목록 로드 대기
        WebDriverWait(driver, 8).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.list_tbl ul"))
        )
        return True
    except Exception:
        return False


def back_to_list(driver, cate_code: str, prog: Progress):
    """뷰어/상세 닫은 후 목록으로 복귀(안정 루트)."""
    url = LIST_URL_BACK.format(cate=cate_code)
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.list_tbl ul"))
        )
        prog.log(f"[list] 목록 복귀: {url}")
    except Exception as e:
        prog.log(f"[list] 목록 복귀 실패: {e}")


# --------------------- 메인 루프 ---------------------
def run(output_root="./kyudb_output_800s"):
    prog = Progress()
    print(f"{prog.ts()} ▶ 시작")
    makedirs(output_root)

    driver = make_driver()
    sess = None

    try:
        driver.get(HOME)
        WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
        sess = selenium_cookies_to_requests(driver)

        for period_name, cate_code in PERIODS:
            prog.set_period(period_name)
            print(f"{prog.ts()} === 시기 진입: {period_name} ===")

            if not open_period(driver, cate_code, prog):
                prog.log(f"[error] 연대 전환 실패 → {cate_code} 스킵")
                continue

            page_idx = 1
            while True:
                items = list_items_on_page(driver)
                prog.log(f"[info] 목록 {page_idx}페이지: 아이템 {len(items)}건")
                if not items:
                    break

                for (item_cd, idx) in items:
                    prog.set_item(item_cd)

                    # 상세 진입 (인라인 → 실패 시 직접 이동)
                    ok = click_item_detail_inline(driver, idx, item_cd, prog)
                    if not ok:
                        ok = goto_item_view_page(driver, item_cd, prog)
                        if not ok:
                            prog.log("  [warn] 상세 진입 실패 → 스킵")
                            continue

                    out_dir_item = os.path.join(output_root, period_name, item_cd)
                    makedirs(out_dir_item)

                    # CSV 두 종류 저장
                    try:
                        extract_detail_csvs(driver, out_dir_item, prog)
                    except Exception as e:
                        prog.log(f"  [warn] CSV 저장 오류: {e}")

                    # 뷰어 열고 모든 장 다운로드
                    if not open_viewer(driver, prog):
                        prog.log("  [warn] 뷰어 열기 실패 → 이미지 스킵")
                    else:
                        try:
                            img_count = viewer_download_all_pages(driver, sess, out_dir_item, prog)
                            prog.log(f"  [done] {period_name} / {item_cd} (images={img_count})")
                        except Exception as e:
                            prog.log(f"  [err] 이미지 저장 중 오류: {e}")

                    # 뷰어를 닫았어도 상세 보기가 남아 있을 수 있으니 목록으로 확실히 복귀
                    back_to_list(driver, cate_code, prog)

                    time.sleep(0.5)  # 서버 배려

                # 다음 목록 페이지
                if next_list_page(driver, prog):
                    page_idx += 1
                    continue
                else:
                    prog.log("→ 다음 목록 페이지 없음")
                    break

            print(f"{prog.ts()} === 시기 종료: {period_name} | images={prog.img_ok} ===")

    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    run(output_root="./kyudb_output_800s")

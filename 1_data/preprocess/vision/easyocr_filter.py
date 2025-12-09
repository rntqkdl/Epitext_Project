"""
EasyOCR Image Filtering
======================================================================
목적: 텍스트가 포함된 탁본 이미지만 선별하여 저장
작성자: Epitext Project Team
======================================================================
"""

import os
import sys
import shutil
import torch
import easyocr
from pathlib import Path

# 로컬 설정 임포트 시도
try:
    from config import Config
except ImportError:
    from .config import Config


# ======================================================================
# 유틸리티 함수
# ======================================================================
def detect_device():
    """사용할 디바이스 반환 (cuda 또는 cpu)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_log(log_path):
    """기존 로그 파일 읽기"""
    processed = {}

    if not log_path.exists() or log_path.stat().st_size == 0:
        print("[Info] No log file found. Creating new log.")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("filename,status\n")
        return processed

    print("[Info] Loading existing log...")
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("filename,") or "," not in line:
                continue
            fname, status = line.split(",", 1)
            processed[fname] = status

    return processed


def init_reader(device):
    """EasyOCR Reader 초기화"""
    use_gpu = device == "cuda"
    reader = easyocr.Reader(Config.LANGUAGES, gpu=use_gpu)
    print(f"[Info] EasyOCR initialized (GPU: {use_gpu})")
    return reader


def get_image_list(src_dir):
    """이미지 파일 리스트 반환"""
    if not src_dir.exists():
        print(f"[Error] Source directory not found: {src_dir}")
        return []
        
    images = sorted(list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")))
    print(f"[Info] Total images found: {len(images)}")
    return images


def print_progress(current, total, fname="", prev_status=None):
    """진행률 출력"""
    if total == 0:
        return

    percent = current / total
    bar_len = 30
    filled = int(percent * bar_len)
    bar = "=" * filled + "-" * (bar_len - filled)
    prev_txt = prev_status if prev_status else "NEW"

    line = f"[{bar}] {percent*100:5.1f}% ({current}/{total}) | {fname} (Status: {prev_txt})"
    sys.stdout.write("\r" + line[:120].ljust(120))
    sys.stdout.flush()


# ======================================================================
# 메인 처리 함수
# ======================================================================
def process_images():
    """EasyOCR 필터링 수행"""
    Config.print_config()
    
    # 디렉토리 생성
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    print(f"[Info] Using device: {device}")
    
    reader = init_reader(device)
    processed = load_log(Config.LOG_PATH)
    images = get_image_list(Config.SRC_DIR)
    
    if not images:
        print("[Error] No images to process.")
        return

    total_images = len(images)
    copied = 0
    skipped = 0
    already_done = 0
    errors = 0
    retried = 0
    recovered_keep = 0

    with open(Config.LOG_PATH, "a", encoding="utf-8") as log_f:
        for idx, img_path in enumerate(images, start=1):
            fname = img_path.name
            prev_status = processed.get(fname)
            dst = Config.OUTPUT_DIR / fname

            print_progress(idx, total_images, fname, prev_status)

            # 1. 이미 처리됨 (KEEP)
            if prev_status == "KEEP":
                if not dst.exists():
                    try:
                        shutil.copy2(img_path, dst)
                        recovered_keep += 1
                    except Exception as e:
                        errors += 1
                        status = f"ERROR:{e.__class__.__name__}"
                        log_f.write(f"{fname},{status}\n")
                        continue
                already_done += 1
                continue

            # 2. 이미 처리됨 (SKIP)
            if prev_status == "SKIP":
                already_done += 1
                continue

            # 3. 에러 재시도
            if prev_status and prev_status.startswith("ERROR:"):
                retried += 1

            # 4. OCR 수행
            try:
                result = reader.readtext(str(img_path))

                if len(result) > 0:
                    shutil.copy2(img_path, dst)
                    copied += 1
                    status = "KEEP"
                else:
                    skipped += 1
                    status = "SKIP"

            except Exception as e:
                errors += 1
                status = f"ERROR:{e.__class__.__name__}"

            log_f.write(f"{fname},{status}\n")
            log_f.flush()

    print("\n\n======================================================")
    print(" Processing Summary")
    print("======================================================")
    print(f" [Done] Already Processed: {already_done}")
    print(f" [New]  Kept (Text Found): {copied}")
    print(f" [New]  Skipped (No Text): {skipped}")
    print(f" [Err]  Errors:            {errors}")
    print(f" [Info] Retried:           {retried}")
    print(f" [Info] Recovered:         {recovered_keep}")
    print(f" [Total] Filtered Images:  {len(list(Config.OUTPUT_DIR.iterdir()))}")
    print("======================================================")


if __name__ == "__main__":
    process_images()
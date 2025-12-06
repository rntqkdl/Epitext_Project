"""
EasyOCR 기반 탁본 이미지 필터링
======================================================================
작성자: 4조 복원왕 김탁본
작성일: 2025-12-07
출처: 4주차 보고서
기능: 텍스트가 포함된 탁본 이미지만 선별
======================================================================
"""

import os
import sys
import shutil
from pathlib import Path

import torch
import easyocr


# ======================================================================
# 경로 설정
# ======================================================================
BASE_DIR = Path(__file__).parent.parent.parent / "raw_data"
IMAGE_FOLDER_NAME = "images"
OUTPUT_FOLDER_NAME = "filtered_takbon"
LOG_FILENAME = "filter_log.csv"
EASYOCR_LANGS = ["ch_tra"]


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
        print("[INFO] 로그 파일 없음 - 새로 생성")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("filename,status\n")
        return processed

    print("[INFO] 기존 로그 파일 발견 - 이어서 처리")
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
    reader = easyocr.Reader(EASYOCR_LANGS, gpu=use_gpu)
    print(f"[INFO] EasyOCR 초기화 완료 (GPU 사용: {use_gpu})")
    return reader


def get_image_list(src_dir):
    """이미지 파일 리스트 반환"""
    images = sorted(list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")))
    print(f"[INFO] 전체 이미지 개수: {len(images)}")
    return images


def print_progress(current, total, fname="", prev_status=None):
    """진행률 출력"""
    if total == 0:
        msg = "진행률: (처리할 이미지 없음)"
        sys.stdout.write("\r" + msg + " " * max(0, 80 - len(msg)))
        sys.stdout.flush()
        return

    percent = current / total
    bar_len = 30
    filled = int(percent * bar_len)
    bar = "=" * filled + "-" * (bar_len - filled)
    prev_txt = prev_status if prev_status else "NEW"

    line = f"[{bar}] {percent*100:5.1f}% ({current}/{total}) | {fname} (이전: {prev_txt})"
    sys.stdout.write("\r" + line[:200])
    sys.stdout.flush()


# ======================================================================
# 메인 처리 함수
# ======================================================================
def process_images():
    """EasyOCR 필터링 수행"""
    src_dir = BASE_DIR / IMAGE_FOLDER_NAME
    out_dir = BASE_DIR / OUTPUT_FOLDER_NAME
    log_path = BASE_DIR / LOG_FILENAME

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 원본 폴더: {src_dir}")
    print(f"[INFO] 결과 폴더: {out_dir}")
    print(f"[INFO] 로그 파일: {log_path}")

    device = detect_device()
    print(f"[INFO] 디바이스: {device}")
    reader = init_reader(device)

    processed = load_log(log_path)
    print(f"[INFO] 로그 기록 파일 수: {len(processed)}")

    images = get_image_list(src_dir)
    total_images = len(images)

    copied = 0
    skipped = 0
    already_done = 0
    errors = 0
    retried = 0
    recovered_keep = 0

    with open(log_path, "a", encoding="utf-8") as log_f:
        for idx, img_path in enumerate(images, start=1):
            fname = img_path.name
            prev_status = processed.get(fname)
            dst = out_dir / fname

            print_progress(idx, total_images, fname, prev_status)

            # 이미 KEEP된 파일
            if prev_status == "KEEP":
                if not dst.exists():
                    try:
                        shutil.copy2(img_path, dst)
                        recovered_keep += 1
                    except Exception as e:
                        errors += 1
                        status = f"ERROR:{e.__class__.__name__}"
                        log_f.write(f"{fname},{status}\n")
                        log_f.flush()
                        continue
                already_done += 1
                continue

            # 이미 SKIP된 파일
            if prev_status == "SKIP":
                already_done += 1
                continue

            # ERROR 재시도
            if prev_status and prev_status.startswith("ERROR:"):
                retried += 1

            # OCR 시도
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

    print()
    print("\n=== 처리 요약 ===")
    print(f"[OK] 이미 끝난 파일: {already_done}")
    print(f"[OK] 새로 KEEP: {copied}")
    print(f"[OK] 새로 SKIP: {skipped}")
    print(f"[ERR] 에러 발생: {errors}")
    print(f"[RETRY] 재시도: {retried}")
    print(f"[RECOVER] 복구 복사: {recovered_keep}")
    print(f"[TOTAL] 현재 필터링된 이미지 수: {len(list(out_dir.iterdir()))}")


if __name__ == "__main__":
    process_images()

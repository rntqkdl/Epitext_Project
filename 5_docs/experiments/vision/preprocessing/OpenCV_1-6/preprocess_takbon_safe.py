# OpenCV 이용한 이미지 전처리 1-6
# 목적: OpenCV를 활용하여 탁본 이미지의 대비 향상과 이진화 등 일련의 전처리 단계 수행
# 요약: 노이즈 억제, 조명 보정, 글자 극성 판별, 대비 강화, 언샤프, 이진화, 모폴로지, 배경 반전 등을 포함한 파이프라인 구현
# 작성일: 2025-12-10
import cv2
import numpy as np
import json
from pathlib import Path

def save_img(p, img):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)

def gentle_unsharp(gray, radius=3, amount=0.45):
    blur = cv2.GaussianBlur(gray, ((radius|1), (radius|1)), 0)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return sharp

def linear_stretch(gray, lo=1.0, hi=99.0):
    p1, p2 = np.percentile(gray, [lo, hi])
    if p2 <= p1 + 1e-6:
        return gray
    out = np.clip((gray - p1) * (255.0 / (p2 - p1)), 0, 255).astype(np.uint8)
    return out

def estimate_text_polarity(gray):
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    band = cv2.dilate(edges, k, iterations=1)
    edge_vals = gray[band > 0]
    bg_mask = cv2.erode((band == 0).astype(np.uint8), k, iterations=2)
    bg_vals = gray[bg_mask > 0]
    if len(edge_vals) < 100 or len(bg_vals) < 100:
        return (gray.mean() < 120)
    return (edge_vals.mean() > bg_vals.mean())

def preprocess_takbon_safe(image_path, out_dir="./out_safe2"):
    name = Path(image_path).stem
    outdir = Path(out_dir)/name
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"file": str(image_path)}
    src = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError(image_path)
    save_img(outdir/"00_src.png", src)
    den = cv2.medianBlur(src, 3)
    save_img(outdir/"01_denoise.png", den)
    k = 71 if min(src.shape) > 1200 else 41
    if k % 2 == 0: k += 1
    bg = cv2.medianBlur(den, k)
    norm = cv2.normalize((den.astype(np.float32) / (bg.astype(np.float32) + 1e-6)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_img(outdir/"02_illum_norm.png", norm)
    need_invert = estimate_text_polarity(norm)
    gray = cv2.bitwise_not(norm) if need_invert else norm
    meta["invert_applied"] = bool(need_invert)
    save_img(outdir/"03_gray_after_polarity.png", gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    c1 = clahe.apply(gray)
    c2 = linear_stretch(c1, 2.0, 98.0)
    save_img(outdir/"04_contrast.png", c2)
    sh = gentle_unsharp(c2, radius=3, amount=0.35)
    save_img(outdir/"05_sharp.png", sh)
    H, W = sh.shape
    win = int(max(25, (min(H, W)//48) | 1))
    bin_adp = cv2.adaptiveThreshold(sh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, win, 8)
    _, bin_otsu = cv2.threshold(sh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    def balance_score(b):
        p = float(np.mean(b == 255))
        return -abs(p - 0.5)
    bin_final = bin_adp if balance_score(bin_adp) >= balance_score(bin_otsu) else bin_otsu
    save_img(outdir/"06_bin_raw.png", bin_final)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    bin_clean = cv2.morphologyEx(bin_final, cv2.MORPH_OPEN, open_k, iterations=2)
    save_img(outdir/"07_bin_clean.png", bin_clean)
    white_ratio = float(np.mean(bin_clean == 255))
    if white_ratio < 0.5:
        bin_clean = cv2.bitwise_not(bin_clean)
    save_img(outdir/f"{name}_ocrprep.png", bin_clean)
    save_img(outdir/f"{name}_master.png", sh)
    meta.update({
        "illum_kernel": int(k),
        "adaptive_win": int(win),
        "white_ratio": white_ratio
    })
    with open(outdir/"params.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f" : {image_path} → {outdir}")
    return outdir

if __name__ == "__main__":
    image_name = "test3.jpg"
    preprocess_takbon_safe(image_name)

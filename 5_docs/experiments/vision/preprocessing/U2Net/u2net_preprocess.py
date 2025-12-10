# U²-Net을 이용한 이미지 전처리
# 목적: U²-Net 모델로 배경 제거와 pepper noise 제거를 수행
# 요약: rembg를 이용해 RGBA 컷아웃을 얻고 OpenCV로 노이즈를 제거한 후 흰 배경으로 합성
# 작성일: 2025-12-10
from rembg import remove, new_session
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

#   ( )
session = new_session("u2net")

def takbon_cutout_and_clean(src_path, dst_path=None, noise_kernel=3, min_area=20):
    src = Path(src_path)
    if dst_path is None:
        dst_path = src.with_name(src.stem + "_white_bg.png")
    im = Image.open(src).convert("RGBA")
    removed = remove(im, session=session)
    rgba = np.array(removed)
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((noise_kernel, noise_kernel), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean)
    mask_final = np.zeros_like(mask_clean)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            mask_final[labels == i] = 255
    white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
    mask_3ch = cv2.merge([mask_final]*3)
    result = np.where(mask_3ch == 255, rgb, white_bg)
    Image.fromarray(result).save(dst_path)
    print(f"   : {dst_path}")
    return str(dst_path)

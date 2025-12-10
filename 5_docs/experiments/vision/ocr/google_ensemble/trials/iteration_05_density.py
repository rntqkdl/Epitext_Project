# 5차 저장: MASK1(완전유실)/MASK2(흔적) 구분
import cv2, numpy as np
def calculate_pixel_density(binary_img, box):
    # ROI 추출 및 잉크 밀도 계산
    return 0.0
def classify_mask(density):
    if density < 0.05: return "MASK1"
    if density > 0.60: return "MASK1" # 오염
    return "MASK2"
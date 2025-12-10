# 이미지 전처리 시행착오 -6: 배경 노이즈 제거
import cv2, numpy as np
def clean_noise(img_path):
    # Adaptive Threshold -> Morphology -> Connected Components 필터링
    pass
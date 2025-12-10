# DBNet을 이용한 이미지 전처리
# 목적: PaddleOCR의 DBNet 모델을 사용하여 글자 영역을 검출하고 배경을 제거
# 요약: CLAHE 대비 강화 후 DBNet으로 텍스트 영역을 검출하고 마스크를 만들어 흰 배경으로 합성
# 작성일: 2025-12-10
from paddleocr import PaddleOCR
import cv2
import numpy as np
from pathlib import Path

ocr = PaddleOCR(use_angle_cls=False, use_gpu=False, det=True, rec=False, lang='ch')

def text_cutout_whitebg(src_path, dst_path=None, contrast_boost=True):
    src_path = Path(src_path)
    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(src_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if contrast_boost:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    tmp_path = src_path.with_name(src_path.stem + "_tmp_for_ocr.png")
    cv2.imwrite(str(tmp_path), gray)
    res = ocr.ocr(str(tmp_path), det=True, rec=False)
    if not res or res[0] is None or len(res[0]) == 0:
        print("    .      .")
        return None
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for line in res[0]:
        if line is None:
            continue
        points = np.array(line[0]).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    mask_3 = cv2.merge([mask]*3)
    result = np.where(mask_3 == 255, img, white_bg)
    if dst_path is None:
        dst_path = src_path.with_name(src_path.stem + "_text_cutout.png")
    cv2.imwrite(str(dst_path), result)
    print(f"  : {dst_path}")
    tmp_path.unlink(missing_ok=True)
    return str(dst_path)

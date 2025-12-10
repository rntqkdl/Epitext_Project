# -*- coding: utf-8 -*-
"""Unified Image Preprocessing Module for Epitext AI Project
========================================================

이 모듈은 한자 이미지의 Swin Gray (3채널) 전처리와 OCR (1채널) 전처리를
한 번의 함수 호출로 수행합니다. OpenCV와 NumPy를 사용하여 탁본 영역
검출, 텍스트 영역 검출, 밝은/흰 배경 보장, 크롭 및 여백 조정 등을
처리합니다.

`UnifiedImagePreprocessor` 클래스를 사용하거나 `preprocess_image_unified`
편의 함수를 통해 즉시 사용할 수 있습니다. 설정 값은 JSON 파일로
제공할 수 있으며, 기본 값은 코드 내부에 정의되어 있습니다.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 기본 설정값
DEFAULT_MARGIN = 10
DEFAULT_BRIGHTNESS_THRESHOLD = 127
DEFAULT_RUBBING_MIN_AREA_RATIO = 0.1
DEFAULT_TEXT_MIN_AREA = 16
DEFAULT_TEXT_AREA_RATIO = 0.00005
DEFAULT_MORPHOLOGY_KERNEL_SIZE = (2, 2)
DEFAULT_MORPHOLOGY_CLOSE_ITERATIONS = 3
DEFAULT_MORPHOLOGY_OPEN_ITERATIONS = 2
DEFAULT_RUBBING_KERNEL_SIZE = (5, 5)
DEFAULT_RUBBING_CLOSE_ITERATIONS = 10
DEFAULT_RUBBING_OPEN_ITERATIONS = 5

class UnifiedImagePreprocessor:
    """통합 이미지 전처리 클래스 (Swin + OCR).

    한 번의 처리로 Swin Gray와 OCR용 이미지를 모두 생성합니다.
    """
    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = self._load_config(config_path)
        logger.info("[INIT] UnifiedImagePreprocessor 초기화 완료")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """설정 파일을 로드합니다."""
        default_config = {
            "margin": DEFAULT_MARGIN,
            "brightness_threshold": DEFAULT_BRIGHTNESS_THRESHOLD,
            "rubbing_min_area_ratio": DEFAULT_RUBBING_MIN_AREA_RATIO,
            "text_min_area": DEFAULT_TEXT_MIN_AREA,
            "text_area_ratio": DEFAULT_TEXT_AREA_RATIO,
            "morphology_kernel_size": DEFAULT_MORPHOLOGY_KERNEL_SIZE,
            "morphology_close_iterations": DEFAULT_MORPHOLOGY_CLOSE_ITERATIONS,
            "morphology_open_iterations": DEFAULT_MORPHOLOGY_OPEN_ITERATIONS,
            "rubbing_kernel_size": DEFAULT_RUBBING_KERNEL_SIZE,
            "rubbing_close_iterations": DEFAULT_RUBBING_CLOSE_ITERATIONS,
            "rubbing_open_iterations": DEFAULT_RUBBING_OPEN_ITERATIONS,
        }
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    logger.info(f"[CONFIG] 설정 파일 로드: {config_path}")
            except Exception as e:
                logger.warning(f"[CONFIG] 설정 파일 로드 실패: {e} - 기본 설정 사용")
        return default_config

    def _find_rubbing_bbox(self, gray_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """탁본 영역을 검출합니다 (큰 어두운 사각형)."""
        H_img, W_img = gray_image.shape
        _, dark_mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        kernel_rub = np.ones(self.config["rubbing_kernel_size"], np.uint8)
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_CLOSE, kernel_rub,
            iterations=self.config["rubbing_close_iterations"]
        )
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_OPEN, kernel_rub,
            iterations=self.config["rubbing_open_iterations"]
        )
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        min_area = (H_img * W_img) * self.config["rubbing_min_area_ratio"]
        if area < min_area:
            return None
        return cv2.boundingRect(largest)

    def _find_text_bbox(self, gray_image: np.ndarray) -> Tuple[int, int, int, int]:
        """텍스트 영역을 검출합니다."""
        H_img, W_img = gray_image.shape
        _, binary = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel_morph = np.ones(self.config["morphology_kernel_size"], np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel_morph,
            iterations=self.config["morphology_close_iterations"]
        )
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel_morph,
            iterations=self.config["morphology_open_iterations"]
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(
            self.config["text_min_area"],
            int((H_img * W_img) * self.config["text_area_ratio"])
        )
        valid_contours = [
            cnt for cnt in contours
            if cv2.contourArea(cv2.boundingRect(cnt)) >= min_area
        ]
        if valid_contours:
            all_points = np.vstack(valid_contours)
            return cv2.boundingRect(all_points)
        return (0, 0, W_img, H_img)

    def _apply_margin(
        self,
        bbox: Tuple[int, int, int, int],
        gray_image: np.ndarray,
        margin_val: int,
    ) -> Tuple[int, int, int, int]:
        """여백을 추가합니다."""
        x, y, w, h = bbox
        H_img, W_img = gray_image.shape
        x_new = max(0, x - margin_val)
        y_new = max(0, y - margin_val)
        w_new = min(W_img - x_new, w + 2 * margin_val)
        h_new = min(H_img - y_new, h + 2 * margin_val)
        return (x_new, y_new, w_new, h_new)

    def _ensure_bright_background(
        self,
        gray_cropped: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """밝은 배경을 보장합니다 (Swin용)."""
        mean_brightness = np.mean(gray_cropped)
        is_inverted = False
        if mean_brightness < self.config["brightness_threshold"]:
            gray_bright = cv2.bitwise_not(gray_cropped)
            is_inverted = True
        else:
            gray_bright = gray_cropped.copy()
        final_brightness = np.mean(gray_bright)
        if final_brightness < self.config["brightness_threshold"]:
            gray_bright = cv2.bitwise_not(gray_bright)
            is_inverted = not is_inverted
            final_brightness = np.mean(gray_bright)
        return gray_bright, {
            "mean_brightness_before": float(mean_brightness),
            "mean_brightness_after": float(final_brightness),
            "is_inverted": is_inverted,
            "is_bright_bg": final_brightness >= self.config["brightness_threshold"],
        }

    def _ensure_white_background(
        self,
        gray_cropped: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """흰 배경을 보장합니다 (OCR용)."""
        _, binary = cv2.threshold(
            gray_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mean_brightness = np.mean(binary)
        if mean_brightness < self.config["brightness_threshold"]:
            binary_final = cv2.bitwise_not(binary)
            polarity = "inverted"
        else:
            binary_final = binary
            polarity = "normal"
        final_brightness = np.mean(binary_final)
        return binary_final, {
            "mean_brightness_before": float(mean_brightness),
            "mean_brightness_after": float(final_brightness),
            "polarity": polarity,
            "is_white_bg": final_brightness > self.config["brightness_threshold"],
        }

    def preprocess_unified(
        self,
        input_image_path: str,
        output_swin_path: str,
        output_ocr_path: str,
        margin: Optional[int] = None,
        use_rubbing: bool = False,
        metadata_json_path: Optional[str] = None,
    ) -> Dict:
        """Swin Gray와 OCR 이미지를 동시에 생성하는 통합 전처리를 수행합니다."""
        margin_val = margin or self.config["margin"]
        try:
            img_bgr = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"이미지 로드 실패: {input_image_path}")
            original_shape = img_bgr.shape
            logger.info(f"[LOAD] 이미지 로드: {input_image_path} {original_shape}")
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if use_rubbing:
                detected_bbox = self._find_rubbing_bbox(gray)
                region_type = "rubbing"
                logger.info("[DETECT] 탁본 영역 검출 모드")
            else:
                detected_bbox = None
                region_type = "text"
                logger.info("[DETECT] 텍스트 영역 검출 모드")
            H_img, W_img = gray.shape
            if detected_bbox is not None:
                bbox_final = self._apply_margin(detected_bbox, gray, margin_val)
                logger.info(f"[DETECT] {region_type} 영역 검출: {bbox_final}")
            else:
                if use_rubbing:
                    bbox_final = (0, 0, W_img, H_img)
                    logger.warning("[DETECT] 탁본 미검출 - 전체 이미지 사용")
                else:
                    bbox_text = self._find_text_bbox(gray)
                    bbox_final = self._apply_margin(bbox_text, gray, margin_val)
                    logger.info(f"[DETECT] 텍스트 영역 검출: {bbox_final}")
            x, y, w, h = bbox_final
            gray_cropped = gray[y : y + h, x : x + w]
            logger.info(f"[CROP] 크롭 완료: {gray_cropped.shape}")
            gray_bright, info_swin = self._ensure_bright_background(gray_cropped)
            swin_output_3ch = cv2.cvtColor(gray_bright, cv2.COLOR_GRAY2BGR)
            binary_final, info_ocr = self._ensure_white_background(gray_cropped)
            out_swin = Path(output_swin_path)
            out_swin.parent.mkdir(parents=True, exist_ok=True)
            swin_ok = cv2.imwrite(str(out_swin), swin_output_3ch)
            out_ocr = Path(output_ocr_path)
            out_ocr.parent.mkdir(parents=True, exist_ok=True)
            ocr_ok = cv2.imwrite(str(out_ocr), binary_final)
            if not swin_ok or not ocr_ok:
                raise ValueError("이미지 저장 실패")
            logger.info(f"[SAVE] Swin 저장: {out_swin}")
            logger.info(f"[SAVE] OCR 저장: {out_ocr}")
            summary = {
                "success": True,
                "version": "Unified Swin Gray + OCR (v1.0.0)",
                "original_shape": original_shape,
                "bbox": bbox_final,
                "region_type": region_type,
                "region_detected": detected_bbox is not None,
                "swin": {
                    "output_path": out_swin.as_posix(),
                    "output_shape": swin_output_3ch.shape,
                    "color_type": "Grayscale 3채널 (B=G=R)",
                    "is_inverted": info_swin["is_inverted"],
                    "mean_brightness_before": info_swin["mean_brightness_before"],
                    "mean_brightness_after": info_swin["mean_brightness_after"],
                    "is_bright_bg": info_swin["is_bright_bg"],
                },
                "ocr": {
                    "output_path": out_ocr.as_posix(),
                    "output_shape": binary_final.shape,
                    "polarity": info_ocr["polarity"],
                    "mean_brightness_before": info_ocr["mean_brightness_before"],
                    "mean_brightness_after": info_ocr["mean_brightness_after"],
                    "is_white_bg": info_ocr["is_white_bg"],
                },
                "message": "통합 전처리 완료 (Swin + OCR)",
            }
            meta_path = Path(metadata_json_path) if metadata_json_path else out_swin.with_name("preprocess_metadata.json")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            summary["metadata_json"] = meta_path.as_posix()
            logger.info(f"[SAVE] 메타데이터 저장: {meta_path}")
            return summary
        except Exception as e:
            logger.error(f"[ERROR] 통합 전처리 실패: {e}")
            return {"success": False, "message": str(e)}

_global_preprocessor: Optional[UnifiedImagePreprocessor] = None

def get_preprocessor(config_path: Optional[str] = None) -> UnifiedImagePreprocessor:
    """전역 전처리기 인스턴스를 반환합니다."""
    global _global_preprocessor
    if _global_preprocessor is None:
        _global_preprocessor = UnifiedImagePreprocessor(config_path)
    return _global_preprocessor

def preprocess_image_unified(
    input_path: str,
    output_swin_path: str,
    output_ocr_path: str,
    margin: Optional[int] = None,
    use_rubbing: bool = False,
    metadata_json_path: Optional[str] = None,
) -> Dict:
    """편의 함수: 통합 이미지 전처리를 수행합니다."""
    prep = get_preprocessor()
    return prep.preprocess_unified(
        input_path,
        output_swin_path,
        output_ocr_path,
        margin,
        use_rubbing,
        metadata_json_path,
    )

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("[TEST] Unified Image Preprocessor 테스트 시작")
    logger.info("=" * 80)
    try:
        prep = UnifiedImagePreprocessor()
        result = prep.preprocess_unified(
            "test_input.jpg",
            "test_swin.jpg",
            "test_ocr.png",
        )
        if result["success"]:
            logger.info("[TEST] 통합 전처리 성공")
            logger.info(f"[TEST] Swin 결과: {result['swin']['output_path']}")
            logger.info(f"[TEST] OCR 결과:  {result['ocr']['output_path']}")
        else:
            logger.error(f"[TEST] 실패: {result['message']}")
    except Exception as e:
        logger.error(f"[TEST] 예외 발생: {e}")
    logger.info("=" * 80)

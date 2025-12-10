"""
Swin Transformer Mask Restoration Pipeline
==================================================================
목적:
- OCR 결과(MASK2 영역) 기반 손상된 문자 이미지 복원
- Swin Transformer V2 모델을 사용하여 Top-K 후보 추론

작성자: Epitext Project Team
업데이트: 2025.12.10
==================================================================
"""

import json
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms

# -------------------------------------------------
# 로깅 설정
# -------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =================================================
# 설정 관리 (Configuration)
# =================================================

@dataclass
class PipelineConfig:
    """Swin 복원 파이프라인 설정"""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[4]
    )

    input_dir: str = "sample_data/ocr_output"
    output_dir: str = "sample_data/swin_output"

    json_name: str = "v1_ocr_18.json"
    image_name: str = "v1_ocr_18.png"
    output_name: str = "v1_swin_18.json"

    checkpoint_path: str = "3_model/saved_models/swinv2_best.pth"

    model_name: str = "swinv2_small_window16_256"
    img_size: int = 256
    top_k: int = 10

    normalize_mean: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    normalize_std: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def input_json(self) -> Path:
        return self.project_root / self.input_dir / self.json_name

    @property
    def input_image(self) -> Path:
        return self.project_root / self.input_dir / self.image_name

    @property
    def output_json(self) -> Path:
        return self.project_root / self.output_dir / self.output_name

    @property
    def model_ckpt(self) -> Path:
        return self.project_root / self.checkpoint_path


# =================================================
# Swin 추론 엔진
# =================================================

class SwinInferenceEngine:
    """Swin Transformer 추론 담당 클래스"""

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.model = None
        self.idx2char = {}
        self.transform = None
        self._load_model()

    def _load_model(self):
        logger.info("Swin 모델 로딩 중...")

        ckpt = torch.load(self.cfg.model_ckpt, map_location=self.cfg.device)
        self.idx2char = {
            int(k): v for k, v in ckpt["char_mapping"]["idx2char"].items()
        }

        self.model = timm.create_model(
            self.cfg.model_name,
            pretrained=False,
            num_classes=ckpt["num_classes"],
            img_size=self.cfg.img_size,
        ).to(self.cfg.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.cfg.normalize_mean, self.cfg.normalize_std
                ),
            ]
        )

        logger.info("Swin 모델 로딩 완료")

    def predict(self, image: np.ndarray) -> List[Dict]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        tensor = self.transform(pil_img).unsqueeze(0).to(self.cfg.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_p, top_idx = torch.topk(probs, self.cfg.top_k)
        results = []

        for p, idx in zip(top_p, top_idx):
            results.append(
                {
                    "token": self.idx2char.get(int(idx), "?"),
                    "probability": float(p),
                }
            )
        return results


# =================================================
# 실행 엔트리
# =================================================

def run_pipeline():
    cfg = PipelineConfig()
    logger.info("Swin MASK 복원 파이프라인 시작")

    if not cfg.input_json.exists() or not cfg.input_image.exists():
        logger.error("입력 파일 누락")
        return

    image = cv2.imread(str(cfg.input_image))
    data = json.loads(cfg.input_json.read_text(encoding="utf-8"))

    engine = SwinInferenceEngine(cfg)
    results = []

    for item in data.get("results", []):
        if "MASK2" not in item.get("type", ""):
            continue

        x1, y1, x2, y2 = map(int, item["box"])
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        preds = engine.predict(crop)
        results.append({"order": item["order"], "top_k": preds})

    cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Swin MASK 복원 완료")


if __name__ == "__main__":
    run_pipeline()

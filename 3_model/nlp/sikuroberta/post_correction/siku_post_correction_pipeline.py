"""
SikuRoBERTa Post-Correction Pipeline
==================================================================
목적:
- OCR 결과 텍스트 후처리
- 구두점 복원 + MASK 문자 예측 (MLM 기반)

작성자: Epitext Project Team
업데이트: 2025.12.10
==================================================================
"""

import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# -------------------------------------------------
# 로깅 설정
# -------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =================================================
# 설정 관리
# =================================================

@dataclass
class PipelineConfig:
    project_root: Path = Path(__file__).resolve().parents[4]

    input_dir: str = "sample_data/ocr_output"
    output_dir: str = "sample_data/siku_output"

    input_txt: str = "v1_ocr.txt"
    output_json: str = "v1_siku.json"

    model_name: str = "jhangyejin/epitext-sikuroberta"
    top_k: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =================================================
# 후처리 엔진
# =================================================

class MaskRestorer:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(cfg.model_name).to(cfg.device)
        self.model.eval()

    def predict(self, text: str) -> List[List[Dict]]:
        inputs = self.tokenizer(
            text, return_tensors="pt"
        ).to(self.cfg.device)

        mask_idx = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero()
        with torch.no_grad():
            logits = self.model(**inputs).logits

        results = []
        for _, pos in mask_idx:
            probs = torch.softmax(logits[0, pos], dim=-1)
            top_p, top_idx = torch.topk(probs, self.cfg.top_k)
            results.append(
                [
                    {
                        "token": self.tokenizer.decode([i]),
                        "probability": float(p),
                    }
                    for p, i in zip(top_p, top_idx)
                ]
            )
        return results


# =================================================
# 실행 엔트리
# =================================================

def run_pipeline():
    cfg = PipelineConfig()
    logger.info("SikuRoBERTa Post-Correction 시작")

    input_path = cfg.project_root / cfg.input_dir / cfg.input_txt
    output_path = cfg.project_root / cfg.output_dir / cfg.output_json

    if not input_path.exists():
        logger.error("입력 TXT 파일 누락")
        return

    text = input_path.read_text(encoding="utf-8")
    text = re.sub(r"\[MASK\d+\]", "[MASK]", text)

    restorer = MaskRestorer(cfg)
    predictions = restorer.predict(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("SikuRoBERTa Post-Correction 완료")


if __name__ == "__main__":
    run_pipeline()

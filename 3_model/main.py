"""
Epitext 모델 파이프라인 메인 엔트리 포인트

- SikuRoBERTa (Train / Post-Correction)
- SwinV2 (Train / Mask Restoration Inference)
- OCR 평가
- Gemini 번역/평가

각 모델은 독립적으로 실행 가능하며,
--task 인자를 통해 선택적으로 실행한다.

실행 예시:
python -m 3_model.main --task sikuroberta_train
python -m 3_model.main --task swin_restore
python -m 3_model.main --task siku_post_correct
"""

import argparse


# =========================
# SikuRoBERTa
# =========================

def run_sikuroberta_train() -> None:
    print("[MODEL] SikuRoBERTa training started")
    from .nlp.sikuroberta.train.train_task import train_main
    train_main()
    print("[MODEL] SikuRoBERTa training finished")


def run_siku_post_correction() -> None:
    """OCR 결과 텍스트 후처리 (구두점 + MASK 복원)"""
    print("[MODEL] SikuRoBERTa post-correction started")
    from .nlp.sikuroberta.post_correction.siku_post_correction_pipeline import run_pipeline
    run_pipeline()
    print("[MODEL] SikuRoBERTa post-correction finished")


# =========================
# SwinV2
# =========================

def run_swin_train() -> None:
    print("[MODEL] SwinV2 training started")
    from .vision.swin_experiment.train import main as swin_train_main
    swin_train_main()
    print("[MODEL] SwinV2 training finished")


def run_swin_restore() -> None:
    """MASK2 이미지 복원 추론"""
    print("[MODEL] Swin Mask Restoration started")
    from .vision.swin_experiment.inference.swin_mask_restore_pipeline import run_pipeline
    run_pipeline()
    print("[MODEL] Swin Mask Restoration finished")


# =========================
# OCR
# =========================

def run_ocr_eval() -> None:
    print("[MODEL] OCR evaluation started")
    from .vision.ocr_experiment.evaluate import main as ocr_eval_main
    ocr_eval_main()
    print("[MODEL] OCR evaluation finished")


# =========================
# Gemini
# =========================

def run_gemini_eval() -> None:
    print("[MODEL] Gemini evaluation started")
    from .nlp.gemini_experiment.eval_task import eval_main
    eval_main()
    print("[MODEL] Gemini evaluation finished")


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser("Epitext Model Pipeline")
    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "sikuroberta_train",
            "siku_post_correct",
            "swin_train",
            "swin_restore",
            "ocr_eval",
            "gemini_eval",
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == "sikuroberta_train":
        run_sikuroberta_train()
    elif args.task == "siku_post_correct":
        run_siku_post_correction()
    elif args.task == "swin_train":
        run_swin_train()
    elif args.task == "swin_restore":
        run_swin_restore()
    elif args.task == "ocr_eval":
        run_ocr_eval()
    elif args.task == "gemini_eval":
        run_gemini_eval()
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()

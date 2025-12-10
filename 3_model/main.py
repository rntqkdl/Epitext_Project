"""
Epitext 모델 파이프라인 메인 엔트리 포인트

- SikuRoBERTa (NLP)
- SwinV2 (Vision)
- OCR 평가
- Gemini 번역/평가

각 모델은 독립적으로 실행 가능하며,
--task 인자를 통해 선택적으로 실행한다.

실행 예시:
python -m 3_model.main --task sikuroberta_train
python -m 3_model.main --task swin_train
python -m 3_model.main --task ocr_eval
"""

import argparse


# =========================
# SikuRoBERTa
# =========================

def run_sikuroberta_train() -> None:
    """SikuRoBERTa 학습"""
    print("[MODEL] SikuRoBERTa training started")

    from .nlp.sikuroberta.train.train_task import train_main
    train_main()

    print("[MODEL] SikuRoBERTa training finished")


# =========================
# SwinV2
# =========================

def run_swin_train() -> None:
    """SwinV2 학습"""
    print("[MODEL] SwinV2 training started")

    from .vision.swin_experiment.train import main as swin_train_main
    swin_train_main()

    print("[MODEL] SwinV2 training finished")


def run_swin_eval() -> None:
    """SwinV2 평가"""
    print("[MODEL] SwinV2 evaluation started")

    from .vision.swin_experiment.evaluation.evaluate import main as swin_eval_main
    swin_eval_main()

    print("[MODEL] SwinV2 evaluation finished")


# =========================
# OCR
# =========================

def run_ocr_eval() -> None:
    """OCR 평가"""
    print("[MODEL] OCR evaluation started")

    from .vision.ocr_experiment.evaluate import main as ocr_eval_main
    ocr_eval_main()

    print("[MODEL] OCR evaluation finished")


# =========================
# Gemini
# =========================

def run_gemini_eval() -> None:
    """Gemini 번역 / 평가"""
    print("[MODEL] Gemini evaluation started")

    from .nlp.gemini_experiment.eval_task import eval_main
    eval_main()

    print("[MODEL] Gemini evaluation finished")


# =========================
# Composite Tasks
# =========================

def run_all_train() -> None:
    """모든 학습 task 실행"""
    run_sikuroberta_train()
    run_swin_train()


def run_all_eval() -> None:
    """모든 평가 task 실행"""
    run_ocr_eval()
    run_gemini_eval()
    run_swin_eval()


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Epitext 모델 파이프라인 실행기"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "sikuroberta_train",
            "swin_train",
            "swin_eval",
            "ocr_eval",
            "gemini_eval",
            "all_train",
            "all_eval",
        ],
        help="실행할 모델 task 이름",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.task == "sikuroberta_train":
        run_sikuroberta_train()
    elif args.task == "swin_train":
        run_swin_train()
    elif args.task == "swin_eval":
        run_swin_eval()
    elif args.task == "ocr_eval":
        run_ocr_eval()
    elif args.task == "gemini_eval":
        run_gemini_eval()
    elif args.task == "all_train":
        run_all_train()
    elif args.task == "all_eval":
        run_all_eval()
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()

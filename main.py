import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))


def run_data_pipeline(step: str) -> None:
    """데이터 파이프라인 실행"""
    data_dir = ROOT_DIR / "1_data"
    sys.path.insert(0, str(data_dir))
    import main as data_main  # 1_data/main.py

    sys.argv = [sys.argv[0], "--step", step]
    data_main.main()


def run_model_pipeline(task: str) -> None:
    """모델 학습/평가 실행"""
    model_dir = ROOT_DIR / "3_model"
    sys.path.insert(0, str(model_dir))
    import main as model_main  # 3_model/main.py

    sys.argv = [sys.argv[0], "--task", task]
    model_main.main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Epitext 연구 파이프라인 통합 실행기"
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["data", "model", "all"],
        default="all",
        help="실행 단계 선택 (기본값: all)",
    )

    parser.add_argument(
        "--step",
        type=str,
        choices=["crawl", "preprocess", "eda", "all"],
        default="all",
        help="데이터 파이프라인 단계 (phase=data일 때)",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "sikuroberta_train",
            "sikuroberta_eval",
            "gemini_eval",
            "swin_train",
            "swin_eval",
            "ocr_eval",
            "all_train",
            "all_eval",
        ],
        default="sikuroberta_train",
        help="모델 작업 선택 (phase=model일 때)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("Epitext Project 연구 파이프라인")
    print("=" * 60)

    if args.phase in ("data", "all"):
        print(f"\n[PHASE 1] 데이터 파이프라인 (단계: {args.step})")
        print("-" * 60)
        run_data_pipeline(step=args.step)

    if args.phase in ("model", "all"):
        print(f"\n[PHASE 2] 모델 학습/평가 (작업: {args.task})")
        print("-" * 60)
        run_model_pipeline(task=args.task)

    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

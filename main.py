"""
Epitext Project Main Controller
======================================================================
목적: 프로젝트의 모든 파이프라인(Data, Model)을 통합 실행하는 진입점
수정사항: 
  1. OMP: Error #15 해결 (KMP_DUPLICATE_LIB_OK=TRUE)
  2. 에러가 발생해도 멈추지 않고 다음 단계 시도 (Try-Except 강화)
======================================================================
"""
import os

# [중요] OMP 라이브러리 충돌 방지 (가장 먼저 실행되어야 함)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import logging
import importlib
from pathlib import Path
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 프로젝트 루트 경로 자동 추가
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Main")

def run_data_pipeline(step):
    logger.info(f"🚀 [Data Phase] Step: {step}")
    
    # ------------------------------------------------------------------
    # 1. Vision 전처리 (EasyOCR Filter)
    # ------------------------------------------------------------------
    if step in ["preprocess", "all"]:
        logger.info("\n>>> [1/2] Vision Preprocessing (EasyOCR Filter) 시도...")
        try:
            mod = importlib.import_module("1_data.preprocess.vision.easyocr_filter.easyocr_filter")
            mod.process_images()
            logger.info("✅ Vision Preprocessing 완료")
        except FileNotFoundError:
            logger.warning("⚠️ 데이터가 없어 Vision 처리를 건너뜁니다 (정상).")
        except Exception as e:
            logger.error(f"❌ Vision 모듈 실행 중 오류: {e}")

    # ------------------------------------------------------------------
    # 2. NLP 전처리 (Text Clean)
    # ------------------------------------------------------------------
    if step in ["preprocess", "all"]:
        logger.info("\n>>> [2/2] NLP Preprocessing (Text Cleaning) 시도...")
        try:
            mod = importlib.import_module("1_data.preprocess.nlp.text_clean")
            if hasattr(mod, 'clean_corpus'):
                mod.clean_corpus()
            else:
                logger.warning("⚠️ clean_corpus 함수를 찾을 수 없습니다. (모듈은 로드됨)")
            logger.info("✅ NLP Preprocessing 완료")
        except FileNotFoundError:
            logger.warning("⚠️ 데이터가 없어 NLP 처리를 건너뜁니다 (정상).")
        except ImportError:
            logger.warning("⚠️ NLP 모듈을 찾을 수 없습니다.")
        except Exception as e:
            logger.error(f"❌ NLP 모듈 실행 중 오류: {e}")

    if step == "eda":
        logger.info(">> EDA 실행 (준비중)")

def run_model_pipeline(task):
    logger.info(f"🧠 [Model Phase] Task: {task}")

    # 공통 실행 함수 (오류가 나도 죽지 않게 처리)
    def try_run(module_path, func_name="main", desc=""):
        logger.info(f"\n>>> {desc} 시작...")
        try:
            mod = importlib.import_module(module_path)
            func = getattr(mod, func_name)
            func()
            logger.info(f"✅ {desc} 완료")
        except FileNotFoundError:
            logger.warning(f"⚠️ 학습/평가 데이터가 없어 중단됨: {desc} (경로 확인 필요)")
        except ImportError as e:
            logger.error(f"❌ 모듈 로드 실패 ({module_path}): {e}")
        except Exception as e:
            logger.error(f"❌ 실행 중 치명적 오류 ({desc}): {e}")

    # ------------------------------------------------------------------
    # Task 분기
    # ------------------------------------------------------------------
    if task == "sikuroberta_train":
        try_run("3_model.nlp.sikuroberta.train.train_task", "main", "SikuRoBERTa 학습")

    elif task == "sikuroberta_eval":
        try_run("3_model.nlp.sikuroberta.evaluation.evaluate_task", "main", "SikuRoBERTa 평가")

    elif task == "swin_train":
        try_run("3_model.vision.swin_experiment.train", "main", "Swin Transformer 학습")

    elif task == "swin_eval":
        try_run("3_model.vision.swin_experiment.evaluate", "main", "Swin Transformer 평가")

    elif task == "gemini_eval":
        try_run("3_model.nlp.gemini_experiment.run_evaluation", "main", "Gemini 번역 실험")
    
    elif task == "all":
        # 전체 테스트 모드 (순차 실행)
        try_run("3_model.nlp.sikuroberta.train.train_task", "main", "[ALL] SikuRoBERTa 학습")
        try_run("3_model.nlp.sikuroberta.evaluation.evaluate_task", "main", "[ALL] SikuRoBERTa 평가")
        try_run("3_model.vision.swin_experiment.train", "main", "[ALL] Swin 학습")
    
    else:
        logger.error(f"알 수 없는 Task입니다: {task}")

def main():
    parser = argparse.ArgumentParser(description="Epitext Project Controller")
    parser.add_argument("--phase", type=str, choices=["data", "model"], required=True, help="실행 단계")
    parser.add_argument("--step", type=str, default="preprocess", help="[Data] 세부 단계")
    parser.add_argument("--task", type=str, help="[Model] 모델 작업명 (예: sikuroberta_train, all)")

    args = parser.parse_args()

    if args.phase == "data":
        run_data_pipeline(args.step)
    elif args.phase == "model":
        if not args.task:
            print("❌ 오류: Model Phase는 --task 인자가 필수입니다.")
            sys.exit(1)
        run_model_pipeline(args.task)

if __name__ == "__main__":
    main()
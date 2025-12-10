# -*- coding: utf-8 -*-
"""
연구 파이프라인 메인 스크립트
======================================================================
목적: Epitext 연구 저장소의 데이터 수집, 전처리, 모델 학습, 평가,
      산출물 생성을 순차적으로 실행합니다. 이 스크립트 하나로 전체
      워크플로우를 관리할 수 있습니다.

특이사항: 최상위 폴더가 숫자로 시작하여 파이썬 패키지로 직접 import
          할 수 없기 때문에, ``importlib``을 이용하여 파일 경로를
          기반으로 모듈을 동적으로 로딩합니다. 검증이나 시스템화와
          같은 새로운 단계를 추가하고 싶다면 적절한 모듈에 함수를
          정의한 뒤 ``run_pipeline()``에 호출을 추가하면 됩니다.

주의: 전체 파이프라인을 실행하기 전에 각 모듈이 단독으로 잘 동작하는지
      각각의 ``main.py``를 개별적으로 실행해 보며 테스트하는 것을 추천합니다.

작성자: Epitext Project Team
작성일: 2025-12-10
======================================================================
"""

from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
from typing import Callable, List


def _load_module(module_path: pathlib.Path, function_name: str) -> Callable:
    """주어진 파일 경로에서 모듈을 로드하고 원하는 함수를 반환합니다.

    매 단계에서 해당 모듈의 ``main`` 함수 또는 특정 함수를 호출하기 위해
    사용되는 유틸리티입니다. 파일이 존재하지 않거나 함수가 없으면 예외를
    발생시켜 추후 파이프라인 단계에서 오류를 명확히 확인할 수 있습니다.

    Args:
        module_path (pathlib.Path): 로드할 파이썬 파일의 절대 경로.
        function_name (str): 모듈 내에서 호출하고자 하는 함수명.

    Returns:
        Callable: 지정한 함수를 반환합니다.

    Raises:
        FileNotFoundError: 모듈 파일이 존재하지 않을 때.
        AttributeError: 지정한 함수가 모듈에 없을 때.
    """
    if not module_path.exists():
        raise FileNotFoundError(f"모듈 파일이 존재하지 않습니다: {module_path}")

    # 파일 경로로부터 모듈 사양(spec) 생성
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    # 모듈 실행하여 동적으로 로드
    spec.loader.exec_module(module)  # type: ignore
    try:
        return getattr(module, function_name)
    except AttributeError as e:
        raise AttributeError(
            f"모듈 '{module_path}'에 함수 '{function_name}'가 없습니다."
        ) from e


def run_pipeline() -> None:
    """전체 데이터 및 모델 파이프라인을 실행합니다.

    파이프라인은 다음 단계로 구성됩니다:

    1. 데이터 수집 (크롤러 실행)
    2. NLP 전처리 및 비전 전처리
    3. NLP 및 비전 모델 학습
    4. NLP 및 비전 모델 평가
    5. 산출물 생성

    각 단계는 해당 파일의 ``main`` 함수를 동적으로 로드하여 실행됩니다.
    어느 단계에서든 예외가 발생하면 파이프라인이 중단되며, 로깅을 통해
    처리 상태를 확인할 수 있습니다.
    """
    project_root = pathlib.Path(__file__).resolve().parent

    # 1. 데이터 수집
    logging.info("1단계: 데이터 수집 시작…")
    data_collection_path = project_root / "1_data" / "crawlers" / "pdf_processing" / "main.py"
    data_collection_main = _load_module(data_collection_path, "main")
    data_collection_main()

    # 2. NLP 전처리
    logging.info("2단계: NLP 전처리 수행…")
    nlp_preprocess_path = project_root / "1_data" / "preprocess" / "nlp" / "text_clean.py"
    nlp_preprocess_main = _load_module(nlp_preprocess_path, "main")
    nlp_preprocess_main()

    # 2. 비전 전처리
    logging.info("2단계: 비전 전처리 수행…")
    vision_preprocess_path = project_root / "1_data" / "preprocess" / "vision" / "easyocr_filter.py"
    vision_preprocess_main = _load_module(vision_preprocess_path, "main")
    vision_preprocess_main()

    # 3. NLP 모델 학습
    logging.info("3단계: NLP 모델 학습…")
    nlp_train_path = project_root / "3_model" / "nlp" / "sikuroberta" / "train" / "train_task.py"
    nlp_train_main = _load_module(nlp_train_path, "main")
    nlp_train_main()

    # 3. 비전 모델 학습
    logging.info("3단계: 비전 모델 학습…")
    vision_train_path = project_root / "3_model" / "vision" / "swin_experiment" / "train.py"
    vision_train_main = _load_module(vision_train_path, "main")
    vision_train_main()

    # 4. NLP 모델 평가
    logging.info("4단계: NLP 모델 평가…")
    nlp_eval_path = project_root / "3_model" / "nlp" / "sikuroberta" / "evaluation" / "evaluate_task.py"
    nlp_eval_main = _load_module(nlp_eval_path, "main")
    nlp_eval_main()

    # 4. 비전 모델 평가
    logging.info("4단계: 비전 모델 평가…")
    vision_eval_path = project_root / "3_model" / "vision" / "swin_experiment" / "evaluation" / "evaluate.py"
    vision_eval_main = _load_module(vision_eval_path, "main")
    vision_eval_main()

    # 5. 산출물 생성
    logging.info("5단계: 산출물 생성…")
    generate_deliverables(project_root)

    logging.info("파이프라인 실행이 완료되었습니다.")


def generate_deliverables(project_root: pathlib.Path) -> None:
    """파이프라인 실행 결과를 정리한 산출물을 생성합니다.

    기본적으로 ``outputs/summary.txt`` 파일을 생성하여 파이프라인이
    정상적으로 완료되었음을 기록합니다. 필요에 따라 이 함수를 확장하여
    모델 성능 지표를 통합하거나 학습된 모델 파일을 별도 위치로 복사하는
    등의 작업을 수행할 수 있습니다.

    Args:
        project_root (pathlib.Path): 저장소 루트 디렉터리 경로.
    """
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    summary_file = outputs_dir / "summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("Epitext 파이프라인이 성공적으로 완료되었습니다.\n")
        f.write("모든 단계가 오류 없이 실행되었습니다.\n\n")
        f.write("이 요약 파일은 필요에 따라 지표, 모델 성능 등을 추가하여\n")
        f.write("확장할 수 있습니다.\n")


if __name__ == "__main__":
    # 기본 로깅 설정: 시간, 레벨, 메시지 출력
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        run_pipeline()
    except Exception:
        # 예외가 발생하면 스택 트레이스를 포함한 메시지를 기록
        logging.exception("예기치 않은 오류로 인해 파이프라인이 실패했습니다.")
        sys.exit(1)
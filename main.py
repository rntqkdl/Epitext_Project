"""Epitext Project entrypoint and module launcher.

This script no longer runs the full data/model pipeline automatically.
Instead, it provides a guided interface that lists per-module entrypoints
and allows the user to execute them selectively. All paths are resolved
relative to the project root discovered via :func:`find_project_root`.

Execution model and conventions
- Individual `main.py` scripts remain the canonical way to run each module.
- Use this interface to list available modules and launch one at a time
  (common test interface).
- Paths are kept relative to the project root; avoid manual ``sys.path``
  tinkering unless debugging.
- Every module should log to console and may write artifacts under
  ``result/`` or ``log/`` folders near the project root as needed.
- ``4_test_main`` is intended for integration-level runs; module-specific
  tests can live beside their own code under dedicated test directories.
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path
from typing import Dict

from project_paths import find_project_root

MODULES: Dict[str, Dict[str, str]] = {
    "crawl_historydb": {
        "path": "1_data/crawlers/01_crawling_historydb.py",
        "description": "역사 데이터 수집 모듈",
    },
    "crawl_nrich": {
        "path": "1_data/crawlers/01_crawling_nrich.py",
        "description": "NRICH 데이터 수집 모듈",
    },
    "crawl_kyu": {
        "path": "1_data/crawlers/01_crawling_kyu.py",
        "description": "규장각 데이터 수집 모듈",
    },
    "pdf_processing": {
        "path": "1_data/crawlers/pdf_processing/main.py",
        "description": "PDF 파서 및 정규화",
    },
    "preprocess_vision_unified": {
        "path": "1_data/preprocess/vision/unified_preprocessor/main.py",
        "description": "Swin/OCR 통합 이미지 전처리",
    },
    "preprocess_nlp": {
        "path": "1_data/preprocess/nlp/text_clean.py",
        "description": "NLP 텍스트 클린업",
    },
    "punctuation_restoration": {
        "path": "3_model/nlp/punctuation_restoration/main.py",
        "description": "한자 구두점 복원",
    },
    "swin_train": {
        "path": "3_model/vision/swin_experiment/train.py",
        "description": "Swin 기반 OCR 학습",
    },
}


def _resolve_module_path(rel_path: str) -> Path:
    root = find_project_root()
    module_path = root / rel_path
    if not module_path.exists():
        raise FileNotFoundError(f"모듈 경로를 찾을 수 없습니다: {module_path}")
    return module_path


def list_modules() -> None:
    print("실행 가능한 모듈 목록 (common test interface):")
    for key, meta in MODULES.items():
        print(f"- {key}: {meta['description']} -> {meta['path']}")
    print("\n사용 예시: python main.py --run punctuation_restoration")


def run_module(name: str) -> None:
    if name not in MODULES:
        raise ValueError(f"알 수 없는 모듈: {name}")
    module_path = _resolve_module_path(MODULES[name]["path"])
    print(f"[RUN] {name} ({module_path})")
    runpy.run_path(str(module_path), run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Epitext 프로젝트 모듈 실행 가이드",
        epilog=(
            "경로는 find_project_root()를 통해 자동 계산됩니다. "
            "로그/결과는 필요 시 project_root/result 또는 project_root/log 에 저장하세요."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="등록된 모듈 목록만 출력",
    )
    parser.add_argument(
        "--run",
        choices=sorted(MODULES.keys()),
        help="선택한 모듈의 main 스크립트 실행",
    )
    args = parser.parse_args(argv)

    if args.list or not args.run:
        list_modules()
        return

    run_module(args.run)


if __name__ == "__main__":
    main()

"""
데이터 파이프라인 진입점.

이 스크립트는 데이터 수집(크롤링), 전처리, EDA 단계를 조정합니다.
각 단계는 개별 함수로 구현되어 있어 독립적으로 실행하거나 순차적으로 실행할 수 있습니다.
실제 크롤링, 전처리, EDA 로직은 각 함수 내에 구현되어야 합니다.

사용 예시:

    # 전체 파이프라인 실행
    python 1_data/main.py --step all

    # 크롤링 단계만 실행
    python 1_data/main.py --step crawl

    # 전처리 단계만 실행
    python 1_data/main.py --step preprocess

    # EDA 단계만 실행
    python 1_data/main.py --step eda
"""

import argparse
import sys
from pathlib import Path

# "1_data" 패키지 import 문제를 해결하기 위해 프로젝트 루트를 sys.path에 추가
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# 주의: 폴더명이 숫자로 시작하면 일반적인 import가 안 될 수 있습니다.
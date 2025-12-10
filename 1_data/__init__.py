"""
데이터 파이프라인 패키지 초기화.

이 모듈은 프로젝트 루트를 기준으로 상대 경로 상수를 정의합니다.
이를 통해 다른 모듈에서 절대 경로를 하드코딩하지 않고 파일을 로드하거나 저장할 수 있습니다.
README에 설명된 디렉토리 구조를 따릅니다:

    1_data/
      raw_data/      # CSV 및 다운로드한 원본 데이터 저장
      processed/     # 전처리 완료된 데이터 (예: 토큰화된 데이터, .npz)
      eda_outputs/   # EDA 과정에서 생성된 보고서 및 시각화 자료

사용자는 코드를 실행하기 전에 제공된 Google Drive 링크에서 원본 데이터를 다운로드하여
``raw_data`` 폴더에 위치시켜야 합니다.
"""

from pathlib import Path

# 프로젝트 루트 경로 (현재 파일 기준 상위 2단계가 아닌, 통상적인 구조인 1단계 위로 가정하고 수정함)
# 만약 패키지 구조가 깊다면 parents[2]로 변경하세요.
ROOT_DIR = Path(__file__).resolve().parents[1]

# 데이터 관련 기본 디렉토리
DATA_DIR = ROOT_DIR / "1_data"

# 원본 데이터 경로 (Drive에서 다운로드)
RAW_DIR = DATA_DIR / "raw_data"

# 전처리 파이프라인 출력 경로
PROCESSED_DIR = DATA_DIR / "processed"

# EDA 보고서 및 임시 파일 저장 경로
EDA_DIR = DATA_DIR / "eda_outputs"

__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "EDA_DIR",
]
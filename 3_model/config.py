
"""

모델 파이프라인 설정.



이 모듈은 랜덤 시드, 장치 선택, 출력 디렉토리, 데이터셋 위치 등

모델 학습 및 평가를 위한 설정을 중앙에서 관리합니다.



경로는 ``3_model.__init__`` 및 ``1_data.__init__``에서 임포트되어

프로젝트 루트를 기준으로 계산됩니다.

"""



from pathlib import Path

from . import SAVED_MODELS_DIR, ROOT_DIR



# PyTorch, NumPy 등에서 결정론적 동작을 보장하기 위한 시드값

RANDOM_SEED: int = 42



# 모델을 실행할 장치; GPU가 사용 가능한 경우 "cuda"로 설정

DEVICE: str = "cuda"



###############################################################################

# 모델별 출력 디렉토리

###############################################################################



SIKUROBERTA_SAVE_DIR = SAVED_MODELS_DIR / "sikuroberta"

SWIN_SAVE_DIR = SAVED_MODELS_DIR / "swin"

OCR_SAVE_DIR = SAVED_MODELS_DIR / "ocr"

GEMINI_SAVE_DIR = SAVED_MODELS_DIR / "gemini"



###############################################################################

# 데이터셋 디렉토리

###############################################################################



# Swin 모델용 전처리된 데이터가 포함된 디렉토리 (.npz 파일)

SWIN_DATA_DIR = ROOT_DIR / "1_data" / "processed" / "swin_data"



# 토큰화된 SikuRoBERTa 데이터셋 디렉토리.

# Drive에서 ``tokenized_sikuroberta_simple_128_split.zip``을 다운로드하고

# ``1_data/raw_data/split_dataset/`` 경로에 압축을 해제하면 아래 디렉토리가 존재해야 합니다.

SIKU_TRAIN_DIR = ROOT_DIR / "1_data" / "raw_data" / "split_dataset" / "tokenized_sikuroberta_simple_128_split" / "train"

SIKU_VALID_DIR = ROOT_DIR / "1_data" / "raw_data" / "split_dataset" / "tokenized_sikuroberta_simple_128_split" / "validation"

SIKU_TEST_DIR = ROOT_DIR / "1_data" / "raw_data" / "split_dataset" / "tokenized_sikuroberta_simple_128_split" / "test"



__all__ = [

    "RANDOM_SEED",

    "DEVICE",

    "SIKUROBERTA_SAVE_DIR",

    "SWIN_SAVE_DIR",

    "OCR_SAVE_DIR",

    "GEMINI_SAVE_DIR",

    "SWIN_DATA_DIR",

    "SIKU_TRAIN_DIR",

    "SIKU_VALID_DIR",

    "SIKU_TEST_DIR",

]

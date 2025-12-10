
"""

모델 파이프라인 패키지 초기화.



프로젝트 루트를 기준으로 자주 사용되는 디렉토리 경로를 정의합니다.

모델 학습 스크립트는 이 상수들을 임포트하여 경로를 하드코딩하지 않고도

체크포인트를 찾거나 아티팩트를 저장할 수 있습니다.

"""



from pathlib import Path



# 프로젝트 루트 경로 결정 (이 파일 기준 두 단계 상위)

ROOT_DIR = Path(__file__).resolve().parents[2]



# 모델 관련 파일의 기본 디렉토리

MODEL_DIR = ROOT_DIR / "3_model"



# 학습된 모델 체크포인트를 저장할 디렉토리 (Git 추적 제외 권장)

SAVED_MODELS_DIR = MODEL_DIR / "saved_models"



__all__ = [

    "ROOT_DIR",

    "MODEL_DIR",

    "SAVED_MODELS_DIR",

]

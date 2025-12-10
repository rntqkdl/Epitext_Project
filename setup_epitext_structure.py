#!/usr/bin/env python3
"""
Epitext Project 구조 자동 생성 스크립트

사용법:
    python setup_epitext_structure.py
    
이 스크립트는 다음 파일들을 자동으로 생성합니다:
    - 1_data/__init__.py, config.py, main.py, README.md
    - 3_model/__init__.py, config.py, main.py, README.md
    - main.py (루트)
    - README.md (업데이트)
"""

import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent


def colored_text(text: str, color: str) -> str:
    """터미널 색상 코드"""
    colors = {
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "magenta": "\033[35m",
        "red": "\033[31m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def create_file(filepath: Path, content: str) -> bool:
    """파일 생성"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        print(colored_text(f"✗ 오류: {filepath} - {e}", "red"))
        return False


def print_header(text: str) -> None:
    """헤더 출력"""
    print(colored_text("\n" + "=" * 60, "cyan"))
    print(colored_text(text, "green"))
    print(colored_text("=" * 60, "cyan"))


# ============================================================================
# 1_DATA 모듈 파일들
# ============================================================================

DATA_INIT = '''from pathlib import Path

# 프로젝트 루트 경로
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "1_data"
RAW_DIR = DATA_DIR / "sample_data"
PROCESSED_DIR = DATA_DIR / "processed"
EDA_DIR = DATA_DIR / "eda_outputs"

__all__ = ["ROOT_DIR", "DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "EDA_DIR"]
'''

DATA_CONFIG = '''from pathlib import Path
from . import ROOT_DIR, RAW_DIR, PROCESSED_DIR, EDA_DIR

# 실행 단계 플래그
RUN_CRAWL = True
RUN_PREPROCESS = True
RUN_EDA = True

# 크롤링 설정
CRAWL_OUTPUT_DIR = RAW_DIR

# 전처리 설정
PREPROCESS_INPUT_DIR = RAW_DIR
PREPROCESS_OUTPUT_DIR = PROCESSED_DIR

# EDA 설정
EDA_INPUT_DIR = PROCESSED_DIR
EDA_REPORT_DIR = EDA_DIR

__all__ = ["RUN_CRAWL", "RUN_PREPROCESS", "RUN_EDA"]
'''

DATA_MAIN = '''"""
1_data/main.py
데이터 파이프라인 실행 오케스트레이터

사용법:
    python 1_data/main.py --step all
    python 1_data/main.py --step crawl
    python 1_data/main.py --step preprocess
    python 1_data/main.py --step eda
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 1_data import config


def run_crawl() -> None:
    """크롤링 실행"""
    print("[데이터] 크롤링 시작...")
    # from crawlers import base_crawler
    # base_crawler.main()
    print("[데이터] 크롤링 완료")


def run_preprocess() -> None:
    """전처리 실행"""
    print("[데이터] 전처리 시작...")
    # from preprocess.nlp import text_clean
    # from preprocess.vision import easyocr_filter
    # text_clean.main()
    # easyocr_filter.main()
    print("[데이터] 전처리 완료")


def run_eda() -> None:
    """EDA 실행"""
    print("[데이터] EDA 시작...")
    # from eda.nlp import text_stats
    # from eda.vision import quality_analysis
    # text_stats.main()
    # quality_analysis.main()
    print("[데이터] EDA 완료")


def run_all() -> None:
    """전체 파이프라인 실행"""
    if config.RUN_CRAWL:
        run_crawl()
    if config.RUN_PREPROCESS:
        run_preprocess()
    if config.RUN_EDA:
        run_eda()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Epitext 데이터 파이프라인 실행"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["crawl", "preprocess", "eda", "all"],
        default="all",
        help="실행할 단계 선택 (기본값: all)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("Epitext 데이터 파이프라인")
    print(f"단계: {args.step}")
    print("=" * 50)

    if args.step == "crawl":
        run_crawl()
    elif args.step == "preprocess":
        run_preprocess()
    elif args.step == "eda":
        run_eda()
    else:
        run_all()

    print("=" * 50)
    print("완료")
    print("=" * 50)


if __name__ == "__main__":
    main()
'''

DATA_README = '''# 1_data - 데이터 파이프라인

데이터 수집, 전처리, EDA를 관리하는 모듈입니다.

## 구조

- `crawlers/`: 데이터 크롤링 코드
- `preprocess/`: 데이터 전처리 (NLP, Vision)
- `eda/`: 탐색적 데이터 분석
- `utils/`: 공통 유틸리티
- `sample_data/`: 샘플 데이터

## 실행 예시

모든 명령은 `Epitext_Project` 루트에서 실행합니다.

### 전체 파이프라인

```bash
python 1_data/main.py --step all
```

### 단계별 실행

```bash
# 크롤링만
python 1_data/main.py --step crawl

# 전처리만
python 1_data/main.py --step preprocess

# EDA만
python 1_data/main.py --step eda
```

### 상세 로그 출력

```bash
python 1_data/main.py --step all -v
```

## 프로젝트 구조

```
1_data/
  __init__.py          # 경로 설정
  config.py            # 실행 설정
  main.py              # 오케스트레이터
  README.md            # 이 파일
  crawlers/            # 크롤링 코드
  preprocess/          # 전처리 코드
  eda/                 # EDA 코드
  utils/               # 공통 유틸리티
  sample_data/         # 샘플 데이터
```

## 참고사항

- 각 단계의 실제 로직은 `config.py`의 플래그로 제어됩니다.
- 크롤링/전처리/EDA 모듈이 준비되면 main.py의 주석을 해제하면 됩니다.
'''

# ============================================================================
# 3_MODEL 모듈 파일들
# ============================================================================

MODEL_INIT = '''from pathlib import Path

# 프로젝트 루트 경로
ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT_DIR / "3_model"
SAVED_MODELS_DIR = MODEL_DIR / "saved_models"

__all__ = ["ROOT_DIR", "MODEL_DIR", "SAVED_MODELS_DIR"]
'''

MODEL_CONFIG = '''from pathlib import Path
from . import SAVED_MODELS_DIR

# 공통 설정
RANDOM_SEED = 42
DEVICE = "cuda"  # "cuda" 또는 "cpu"

# 모델별 저장 경로
SIKUROBERTA_SAVE_DIR = SAVED_MODELS_DIR / "sikuroberta"
SWIN_SAVE_DIR = SAVED_MODELS_DIR / "swin"
OCR_SAVE_DIR = SAVED_MODELS_DIR / "ocr"
GEMINI_SAVE_DIR = SAVED_MODELS_DIR / "gemini"

# 데이터 경로
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "1_data" / "processed"

__all__ = [
    "RANDOM_SEED",
    "DEVICE",
    "SIKUROBERTA_SAVE_DIR",
    "SWIN_SAVE_DIR",
    "OCR_SAVE_DIR",
    "GEMINI_SAVE_DIR",
    "DATA_DIR"
]
'''

MODEL_MAIN = '''"""
3_model/main.py
모델 학습/평가 실행 오케스트레이터

사용법:
    python 3_model/main.py --task sikuroberta_train
    python 3_model/main.py --task swin_train
    python 3_model/main.py --task all_eval
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 3_model import config


def run_sikuroberta_train() -> None:
    """SikuRoBERTa 학습"""
    print("[모델] SikuRoBERTa 학습 시작...")
    # from nlp.sikuroberta.train import train_task
    # train_task.main()
    print("[모델] SikuRoBERTa 학습 완료")


def run_sikuroberta_eval() -> None:
    """SikuRoBERTa 평가"""
    print("[모델] SikuRoBERTa 평가 시작...")
    # from nlp.sikuroberta.evaluation import evaluate_task
    # evaluate_task.main()
    print("[모델] SikuRoBERTa 평가 완료")


def run_gemini_eval() -> None:
    """Gemini 평가"""
    print("[모델] Gemini 평가 시작...")
    # from nlp.gemini_experiment import run_evaluation
    # run_evaluation.main()
    print("[모델] Gemini 평가 완료")


def run_swin_train() -> None:
    """SwinV2 학습"""
    print("[모델] SwinV2 학습 시작...")
    # from vision.swin_experiment.train import main as swin_train_main
    # swin_train_main()
    print("[모델] SwinV2 학습 완료")


def run_swin_eval() -> None:
    """SwinV2 평가"""
    print("[모델] SwinV2 평가 시작...")
    # from vision.swin_experiment.evaluation import evaluate
    # evaluate()
    print("[모델] SwinV2 평가 완료")


def run_ocr_eval() -> None:
    """OCR 평가"""
    print("[모델] OCR 평가 시작...")
    # from vision.ocr_experiment import evaluate
    # evaluate()
    print("[모델] OCR 평가 완료")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Epitext 모델 학습/평가 실행"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
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
        help="실행할 작업 선택",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("Epitext 모델 학습/평가")
    print(f"작업: {args.task}")
    print("=" * 50)

    if args.task == "sikuroberta_train":
        run_sikuroberta_train()
    elif args.task == "sikuroberta_eval":
        run_sikuroberta_eval()
    elif args.task == "gemini_eval":
        run_gemini_eval()
    elif args.task == "swin_train":
        run_swin_train()
    elif args.task == "swin_eval":
        run_swin_eval()
    elif args.task == "ocr_eval":
        run_ocr_eval()
    elif args.task == "all_train":
        run_sikuroberta_train()
        run_swin_train()
    elif args.task == "all_eval":
        run_sikuroberta_eval()
        run_gemini_eval()
        run_swin_eval()
        run_ocr_eval()

    print("=" * 50)
    print("완료")
    print("=" * 50)


if __name__ == "__main__":
    main()
'''

MODEL_README = '''# 3_model - 모델 학습/평가

NLP 및 Vision 모델 학습과 평가를 관리하는 모듈입니다.

## 구조

- `nlp/`
  - `sikuroberta/`: 한자 인식 모델 (SikuRoBERTa)
  - `gemini_experiment/`: Gemini 기반 평가
- `vision/`
  - `swin_experiment/`: Vision Transformer (SwinV2)
  - `ocr_experiment/`: OCR 모델 평가
- `saved_models/`: 학습된 모델 저장 경로

## 실행 예시

모든 명령은 `Epitext_Project` 루트에서 실행합니다.

### 개별 모델 학습

```bash
# SikuRoBERTa 학습
python 3_model/main.py --task sikuroberta_train

# SwinV2 학습
python 3_model/main.py --task swin_train
```

### 개별 모델 평가

```bash
# SikuRoBERTa 평가
python 3_model/main.py --task sikuroberta_eval

# Gemini 평가
python 3_model/main.py --task gemini_eval

# SwinV2 평가
python 3_model/main.py --task swin_eval

# OCR 평가
python 3_model/main.py --task ocr_eval
```

### 일괄 실행

```bash
# 모든 모델 학습
python 3_model/main.py --task all_train

# 모든 모델 평가
python 3_model/main.py --task all_eval
```

### 상세 로그 출력

```bash
python 3_model/main.py --task sikuroberta_train -v
```

## 설정

`config.py`에서 다음을 설정할 수 있습니다:

- `RANDOM_SEED`: 재현성 보장 (기본값: 42)
- `DEVICE`: "cuda" 또는 "cpu"
- 각 모델의 저장 경로

## 참고사항

- 모델 학습 코드는 각 폴더의 `train/main.py`에 위치합니다.
- 평가 코드는 각 폴더의 `evaluation/main.py`에 위치합니다.
'''

# ============================================================================
# ROOT main.py
# ============================================================================

ROOT_MAIN = '''"""
Epitext Project 최상위 실행 스크립트

사용법:
    python main.py --phase data --step all
    python main.py --phase model --task sikuroberta_train
    python main.py --phase all
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))


def run_data_pipeline(step: str) -> None:
    """데이터 파이프라인 실행"""
    from 1_data import main as data_main
    sys.argv = [sys.argv[0], "--step", step]
    data_main.main()


def run_model_pipeline(task: str) -> None:
    """모델 학습/평가 실행"""
    from 3_model import main as model_main
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
        "-v", "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\\n" + "=" * 60)
    print("Epitext Project 연구 파이프라인")
    print("=" * 60)

    try:
        if args.phase in ("data", "all"):
            print(f"\\n[PHASE 1] 데이터 파이프라인 (단계: {args.step})")
            print("-" * 60)
            run_data_pipeline(step=args.step)

        if args.phase in ("model", "all"):
            print(f"\\n[PHASE 2] 모델 학습/평가 (작업: {args.task})")
            print("-" * 60)
            run_model_pipeline(task=args.task)

        print("\\n" + "=" * 60)
        print("모든 작업 완료!")
        print("=" * 60 + "\\n")

    except Exception as e:
        print(f"\\n오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
'''

# ============================================================================
# ROOT README.md (더 자세한 버전)
# ============================================================================

ROOT_README = '''# Epitext Project – 연구 및 실험 저장소

한자 탁본 자동 복원 AI 시스템 연구 및 실험 저장소입니다.

## 개요

이 저장소는 **연구 및 실험 코드**를 포함하며, 다음과 같은 업무를 수행하는 데 사용됩니다:

- 데이터 수집 및 전처리 파이프라인 구축
- NLP/비전 모델 학습 및 평가 실험
- 연구 결과 및 문서 관리

서비스용 백엔드/프론트엔드 코드는 별도 저장소(`rntqkdl/Epitext_Service`)에서 관리합니다.

## 프로젝트 구조

```
Epitext_Project/
├── 1_data/                 # 데이터 파이프라인
│   ├── __init__.py        # 경로 설정
│   ├── config.py          # 실행 설정
│   ├── main.py            # 오케스트레이터
│   ├── README.md          # 상세 설명서
│   ├── crawlers/          # 데이터 크롤러
│   ├── preprocess/        # 데이터 전처리
│   ├── eda/               # 탐색적 데이터 분석
│   ├── utils/             # 공통 유틸리티
│   └── sample_data/       # 샘플 데이터
│
├── 2_notebooks/           # Jupyter 노트북 (EDA, 실험)
│
├── 3_model/               # 모델 학습/평가
│   ├── __init__.py        # 경로 설정
│   ├── config.py          # 모델 설정
│   ├── main.py            # 오케스트레이터
│   ├── README.md          # 상세 설명서
│   ├── nlp/               # NLP 모델
│   │   ├── sikuroberta/   # SikuRoBERTa 실험
│   │   └── gemini_experiment/  # Gemini 실험
│   ├── vision/            # Vision 모델
│   │   ├── swin_experiment/    # SwinV2 실험
│   │   └── ocr_experiment/     # OCR 실험
│   └── saved_models/      # 학습된 모델 저장
│
├── 5_docs/                # 문서 및 리포트
│
├── main.py                # 통합 실행 진입점
├── requirements.txt       # Python 의존성
├── README.md              # 이 파일
└── test.py                # 테스트 스크립트
```

## 빠른 시작

### 설치

```bash
# 저장소 복제
git clone https://github.com/rntqkdl/Epitext_Project.git
cd Epitext_Project

# 가상환경 생성 및 활성화
conda create -n epitext python=3.10
conda activate epitext

# 의존성 설치
pip install -r requirements.txt
```

## 실행 방법

### 방법 1: 통합 진입점 (권장)

`main.py`를 사용하면 데이터와 모델을 한 번에 제어할 수 있습니다.

#### 전체 파이프라인

```bash
# 데이터 처리 + 모든 모델 학습
python main.py --phase all --task all_train
```

#### 데이터만 처리

```bash
python main.py --phase data --step all
python main.py --phase data --step crawl
python main.py --phase data --step preprocess
python main.py --phase data --step eda
```

#### 모델만 실행

```bash
python main.py --phase model --task sikuroberta_train
python main.py --phase model --task all_eval
```

### 방법 2: 모듈별 직접 실행

더 세밀한 제어가 필요하면 각 모듈의 `main.py`를 직접 실행합니다.

#### 데이터 파이프라인 (1_data/main.py)

```bash
# 전체 데이터 처리
python 1_data/main.py --step all

# 크롤링만
python 1_data/main.py --step crawl

# 전처리만
python 1_data/main.py --step preprocess

# EDA만
python 1_data/main.py --step eda
```

자세한 내용: [1_data/README.md](./1_data/README.md)

#### 모델 파이프라인 (3_model/main.py)

```bash
# SikuRoBERTa 학습
python 3_model/main.py --task sikuroberta_train

# SwinV2 학습
python 3_model/main.py --task swin_train

# 모든 모델 학습
python 3_model/main.py --task all_train

# 모든 모델 평가
python 3_model/main.py --task all_eval
```

자세한 내용: [3_model/README.md](./3_model/README.md)

## 실행 옵션

### 공통 옵션

```bash
-v, --verbose     상세 로그 출력
-h, --help        도움말 표시
```

### main.py 옵션

```bash
--phase {data|model|all}              실행 단계 (기본값: all)
--step {crawl|preprocess|eda|all}    데이터 단계 (phase=data일 때, 기본값: all)
--task {TASK_NAME}                    모델 작업 (phase=model일 때, 기본값: sikuroberta_train)
```

## 모델 목록

### NLP 모델

| 모델 | 설명 | 경로 | 실행 명령어 |
|------|------|------|----------|
| SikuRoBERTa | 한자 인식 | `3_model/nlp/sikuroberta/` | `python 3_model/main.py --task sikuroberta_train` |
| Gemini | 멀티모달 평가 | `3_model/nlp/gemini_experiment/` | `python 3_model/main.py --task gemini_eval` |

### Vision 모델

| 모델 | 설명 | 경로 | 실행 명령어 |
|------|------|------|----------|
| SwinV2 | 이미지 복원 | `3_model/vision/swin_experiment/` | `python 3_model/main.py --task swin_train` |
| OCR | 문자 인식 | `3_model/vision/ocr_experiment/` | `python 3_model/main.py --task ocr_eval` |

## 설정 파일

### 1_data/config.py

데이터 파이프라인 제어:

```python
RUN_CRAWL = True          # 크롤링 실행 여부
RUN_PREPROCESS = True     # 전처리 실행 여부
RUN_EDA = True            # EDA 실행 여부
```

### 3_model/config.py

모델 학습/평가 제어:

```python
RANDOM_SEED = 42          # 재현성 보장
DEVICE = "cuda"           # "cuda" 또는 "cpu"
```

## 개발 가이드

### 새 모델 추가

1. `3_model/nlp/` 또는 `3_model/vision/` 아래 새 폴더 생성
2. `train/main.py`와 `evaluation/main.py` 작성
3. `3_model/main.py`에 실행 함수와 task 추가

### 새 전처리 파이프라인 추가

1. `1_data/preprocess/` 아래 새 모듈 생성
2. `main()` 함수 구현
3. `1_data/main.py`의 `run_preprocess()`에 호출 추가

## 주요 의존성

- **PyTorch**: `torch`, `torchvision`
- **NLP**: `transformers`, `tokenizers`
- **Vision**: `opencv-python`, `torchvision`
- **Data**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`

자세한 내용: [requirements.txt](./requirements.txt)

## 참고 사항

- 모든 모델은 `3_model/saved_models/`에 저장됩니다.
- 처리된 데이터는 `1_data/processed/`에 저장됩니다.
- 각 모듈의 상세 문서는 해당 폴더의 README.md를 참고하세요.

## 라이센스

MIT License

## 연락처

- GitHub: [@rntqkdl](https://github.com/rntqkdl)
'''


def main() -> None:
    """메인 실행 함수"""
    
    print_header("Epitext Project 구조 자동 생성 시작")
    print(colored_text(f"프로젝트 경로: {PROJECT_ROOT}", "yellow"))

    files_created = 0

    # 1_data 파일들
    print(colored_text("\n[1/4] 1_data 폴더 구조 생성 중...", "magenta"))
    
    data_files: Dict[str, str] = {
        "1_data/__init__.py": DATA_INIT,
        "1_data/config.py": DATA_CONFIG,
        "1_data/main.py": DATA_MAIN,
        "1_data/README.md": DATA_README,
    }
    
    for filename, content in data_files.items():
        filepath = PROJECT_ROOT / filename
        if create_file(filepath, content):
            print(colored_text(f"  ✓ {filename} 생성됨", "green"))
            files_created += 1

    # 3_model 파일들
    print(colored_text("\n[2/4] 3_model 폴더 구조 생성 중...", "magenta"))
    
    model_files: Dict[str, str] = {
        "3_model/__init__.py": MODEL_INIT,
        "3_model/config.py": MODEL_CONFIG,
        "3_model/main.py": MODEL_MAIN,
        "3_model/README.md": MODEL_README,
    }
    
    for filename, content in model_files.items():
        filepath = PROJECT_ROOT / filename
        if create_file(filepath, content):
            print(colored_text(f"  ✓ {filename} 생성됨", "green"))
            files_created += 1

    # 루트 main.py
    print(colored_text("\n[3/4] 루트 main.py 생성 중...", "magenta"))
    filepath = PROJECT_ROOT / "main.py"
    if create_file(filepath, ROOT_MAIN):
        print(colored_text(f"  ✓ main.py 생성됨", "green"))
        files_created += 1

    # 루트 README 업데이트
    print(colored_text("\n[4/4] 루트 README.md 업데이트 중...", "magenta"))
    filepath = PROJECT_ROOT / "README.md"
    if create_file(filepath, ROOT_README):
        print(colored_text(f"  ✓ README.md 업데이트됨", "green"))
        files_created += 1

    # 완료 메시지
    print_header(f"완료! {files_created}개 파일 생성/업데이트됨")

    print(colored_text("\n생성된 파일들:", "yellow"))
    print(colored_text("  1_data/__init__.py", "gray"))
    print(colored_text("  1_data/config.py", "gray"))
    print(colored_text("  1_data/main.py", "gray"))
    print(colored_text("  1_data/README.md", "gray"))
    print(colored_text("  3_model/__init__.py", "gray"))
    print(colored_text("  3_model/config.py", "gray"))
    print(colored_text("  3_model/main.py", "gray"))
    print(colored_text("  3_model/README.md", "gray"))
    print(colored_text("  main.py (루트)", "gray"))
    print(colored_text("  README.md (업데이트)", "gray"))

    print(colored_text("\n다음 명령어로 테스트하세요:", "yellow"))
    print(colored_text("  python main.py --phase data --step all", "cyan"))
    print(colored_text("  python main.py --phase model --task sikuroberta_train", "cyan"))
    print(colored_text("  python 1_data/main.py --step crawl", "cyan"))

    print(colored_text("\nVSCode에서 새로고침 (Ctrl+Shift+P → reload) 후 파일 확인", "magenta"))
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(colored_text("\n작업 취소됨", "red"))
        sys.exit(1)
    except Exception as e:
        print(colored_text(f"\n오류 발생: {e}", "red"))
        sys.exit(1)

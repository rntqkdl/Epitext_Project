#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Epitext 연구 저장소 모듈 설치 스크립트
=====================================

이 스크립트는 연구용 저장소에 두 가지 모듈을 설치합니다.

1. **구두점 복원 모델** – `3_model/nlp/punctuation_restoration` 폴더에
   배치됩니다. 이 모듈은 한문 텍스트에서 누락된 구두점을 SikuRoBERTa
   기반 모델로 복원하는 코드를 포함하며, 데이터 전체를 처리하는
   스크립트(`main.py`)와 설정 파일(`config.py`), 초기화 파일,
   README를 자동으로 생성합니다.
2. **통합 이미지 전처리기** – `1_data/preprocess/vision/unified_preprocessor`에
   배치됩니다. 하나의 입력 이미지로부터 Swin Gray(밝은 배경 3채널)
   이미지와 OCR(흰 배경 1채널) 이미지를 동시에 생성하는 모듈입니다.
   기본 파라미터를 담은 JSON 설정 파일과 함께 제공됩니다.

본 스크립트를 실행한 후 변경 사항을 Git에 커밋하여 원격 저장소에
반영하십시오. 스크립트 실행 전에 `setup_import_aliases.py`를 실행하면
`epitext_data`와 `epitext_model` alias 패키지가 생성되어 모듈을 더
직관적으로 임포트할 수 있습니다.
"""

import json
from pathlib import Path
from typing import Dict, Optional


def write_file(path: Path, content: str) -> None:
    """지정된 경로에 UTF-8 인코딩으로 파일을 기록합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def create_punctuation_module(root: Path) -> None:
    """구두점 복원 모델 모듈을 생성합니다 (3_model/nlp)."""
    module_dir = root / "3_model" / "nlp" / "punctuation_restoration"
    module_dir.mkdir(parents=True, exist_ok=True)

    # __init__.py: 패키지 설명과 사용 예시를 포함합니다.
    init_py = """# -*- coding: utf-8 -*-
'''구두점 복원 모델 패키지 초기화 모듈.

이 패키지는 한문 텍스트에서 누락된 구두점을 복원하기 위한 도구를
제공합니다. `main.py`의 `main()` 함수를 실행하면 CSV 데이터셋 전체의
구두점을 복원하고, `restore_punctuation_sliding` 함수를 사용하면 단일
문장의 구두점을 복원할 수 있습니다.

예제:

    from epitext_model.nlp.punctuation_restoration import config, main as punc_main
    from epitext_model.nlp.punctuation_restoration.main import restore_punctuation_sliding

    # 모델 다운로드 및 로드
    punc_main.download_model(config.MODEL_TAG, config.MODEL_CACHE_DIR)
    model_info = punc_main.load_model(config.MODEL_CACHE_DIR, device="cpu")

    # 단일 문장 처리
    cleaned = punc_main.remove_punctuation("예시 문장입니다")
    restored = restore_punctuation_sliding(cleaned, model_info)
    print(restored)

    # CSV 전체 처리
    punc_main.main()

위 예제에서 `epitext_model`은 `setup_import_aliases.py`를 실행하여
생성되는 alias 패키지입니다. alias를 사용하지 않는 경우에는
`importlib`을 통해 경로 기반으로 모듈을 로드할 수 있습니다.
'''
"""
    write_file(module_dir / "__init__.py", init_py)

    # config.py: 모델 태그, 입력/출력 파일 등 설정값 정의
    config_py = """# -*- coding: utf-8 -*-
'''구두점 복원 모델 설정 파일.

이 모듈은 구두점 복원에 필요한 경로와 모델 설정을 정의합니다. 경로는
프로젝트 루트 기준으로 상대적으로 설정되며, 필요시 수정하여 사용할
수 있습니다.
'''

from pathlib import Path

# 프로젝트 루트 경로 계산 (현재 파일에서 세 단계 위가 프로젝트 루트)
BASE_DIR = Path(__file__).resolve().parents[3]

# 입력 CSV: 구두점이 제거된 텍스트가 저장된 파일
INPUT_CSV = BASE_DIR / "1_data" / "raw_data" / "doc_id_transcript_dataset_processed.csv"

# 출력 CSV: 구두점 복원 결과를 저장할 파일
OUTPUT_CSV = BASE_DIR / "1_data" / "raw_data" / "doc_id_transcript_with_punctuation.csv"

# Hugging Face 모델 태그 (SikuRoBERTa 기반)
MODEL_TAG = "seyoungsong/SikuRoBERTa-PUNC-AJD-KLC"

# 모델 캐시 디렉터리 (다운로드된 모델을 저장)
MODEL_CACHE_DIR = BASE_DIR / "models" / "punctuation"

# 최대 시퀀스 길이
MAX_LENGTH = 512

# 슬라이딩 윈도우 크기 및 오버랩 크기 (문자 단위)
WINDOW_SIZE_CHARS = 400
OVERLAP_CHARS = 100

# 배치 크기 (진행 상태 표시용)
DISPLAY_BATCH_SIZE = 32
"""
    write_file(module_dir / "config.py", config_py)

    # main.py: 구두점 복원 실행 스크립트
    main_py = """# -*- coding: utf-8 -*-
'''구두점 복원 메인 모듈.

이 스크립트는 SikuRoBERTa 기반 토큰 분류 모델을 사용하여
한문 텍스트의 구두점을 자동으로 복원합니다. 긴 텍스트에 대해서는
슬라이딩 윈도우 방식으로 처리하며, 전체 데이터셋을 대상으로 결과를
생성합니다.
'''

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertForTokenClassification,
    pipeline,
)

from . import config


def download_model(model_tag: str, model_path: Path) -> None:
    '''Hugging Face Hub에서 모델을 다운로드합니다.'''
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and any(model_path.iterdir()):
        print(f"모델이 이미 존재함: {model_path}")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub 패키지가 설치되어 있지 않습니다. requirements.txt를 확인하세요.")
    print(f"모델 다운로드 시작: {model_tag}")
    snapshot_download(
        repo_id=model_tag,
        repo_type="model",
        local_dir=model_path,
        local_dir_use_symlinks=False,
    )
    print("모델 다운로드 완료")


def load_model(model_path: Path, device: str = "cpu") -> Dict:
    '''모델과 토크나이저, 파이프라인을 로드합니다.'''
    torch_dtype = torch.float16 if "cuda" in device else torch.float32
    model_files = sorted(model_path.rglob("*.safetensors"))
    if not model_files:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    hface_path = model_files[0].parent
    tokenizer = AutoTokenizer.from_pretrained(
        hface_path,
        model_max_length=config.MAX_LENGTH,
    )
    model: BertForTokenClassification = AutoModelForTokenClassification.from_pretrained(
        hface_path,
        device_map=device,
        torch_dtype=torch_dtype,
    )
    model.eval()
    pipe = pipeline(task="ner", model=model, tokenizer=tokenizer)
    label2id_path = hface_path / "label2id.json"
    if not label2id_path.exists():
        label2id_path = hface_path.parent / "label2id.json"
    if not label2id_path.exists():
        raise FileNotFoundError(f"label2id.json 파일을 찾을 수 없습니다: {hface_path}")
    label2id = json.loads(label2id_path.read_text(encoding="utf-8"))
    return {
        "model": model,
        "tokenizer": tokenizer,
        "pipe": pipe,
        "label2id": label2id,
    }


def remove_punctuation(text: str) -> str:
    '''문자열에서 구두점과 공백을 제거합니다.'''
    return ''.join(c for c in text if unicodedata.category(c)[0] not in 'PZ')


def _reduce_punctuation(punc: str) -> str:
    '''여러 구두점을 하나의 기호로 축약합니다.'''
    reduce_map = {
        ",": ",", "-": ",", "/": ",", ":": ",", "|": ",",
        "·": ",", "、": ",",
        "?": "?",
        "!": "。", ".": "。", ";": "。", "。": "。",
    }
    reduced = ''.join([reduce_map.get(c, "") for c in punc])
    priority_order = "?。,"  # 우선순위: 물음표, 마침표, 쉼표
    if not set(reduced).intersection(priority_order):
        return ""
    counts = {c: reduced.count(c) for c in priority_order}
    max_count = max(counts.values())
    max_keys = {k for k, v in counts.items() if v == max_count}
    if len(max_keys) == 1:
        return max_keys.pop()
    for c in priority_order:
        if c in max_keys:
            return c
    return ""


def _insert_space_after(text: str, chars: str) -> str:
    '''특정 문자 뒤에 공백을 삽입합니다.'''
    result = ""
    for char in text:
        result += char
        if char in chars:
            result += " "
    return result


def build_label2punc(
    label2id: Dict[str, int],
    add_space: bool = True,
    reduce: bool = True,
) -> Dict[str, str]:
    '''레이블을 구두점으로 매핑하는 딕셔너리를 생성합니다.

    `label2id` 사전에서 라벨 이름을 키로 사용하고 구두점을 값으로 하는
    새 사전을 만들어 돌려줍니다. 이 함수의 설명에는 Unicode 화살표
    기호 대신 ASCII 화살표(`->`)를 사용하여 Windows 환경에서도 오류가
    발생하지 않도록 합니다.
    '''
    label2punc = {f"B-{v}": k for k, v in label2id.items()}
    label2punc["O"] = ""
    if reduce:
        new_map = {}
        for lbl, punc in label2punc.items():
            if lbl == "O":
                new_map[lbl] = ""
            else:
                new_map[lbl] = _reduce_punctuation(punc)
        label2punc = new_map
    if add_space:
        special_puncs = "!:,?。"
        label2punc = {k: _insert_space_after(v, special_puncs) for k, v in label2punc.items()}
        label2punc["O"] = ""
    return label2punc


def align_predictions(
    text: str,
    predictions: List[dict],
) -> Tuple[List[str], List[str]]:
    '''토큰 단위 예측 결과를 문자 단위로 정렬합니다.'''
    words = list(text)
    labels = ["O"] * len(words)
    for pred in predictions:
        idx = pred["end"] - 1
        if 0 <= idx < len(labels):
            labels[idx] = pred["entity"]
    return words, labels


def predict_labels_sliding(
    text: str,
    model_info: Dict,
    window_size: int = config.WINDOW_SIZE_CHARS,
    overlap: int = config.OVERLAP_CHARS,
) -> List[str]:
    '''슬라이딩 윈도우 방식으로 전체 텍스트의 레이블을 예측합니다.'''
    pipe = model_info["pipe"]
    n = len(text)
    if n == 0:
        return []
    labels_per_pos: List[List[str]] = [[] for _ in range(n)]
    stride = max(1, window_size - overlap)
    start = 0
    while start < n:
        end = min(start + window_size, n)
        sub_text = text[start:end]
        sub_preds = pipe(sub_text)
        _, sub_labels = align_predictions(sub_text, sub_preds)
        for i, lbl in enumerate(sub_labels):
            global_idx = start + i
            if global_idx >= n:
                break
            if lbl != "O":
                labels_per_pos[global_idx].append(lbl)
        if end == n:
            break
        start += stride
    final_labels: List[str] = []
    for candidates in labels_per_pos:
        if not candidates:
            final_labels.append("O")
        else:
            counter = Counter(candidates)
            most_common_label, _ = counter.most_common(1)[0]
            final_labels.append(most_common_label)
    return final_labels


def restore_punctuation_sliding(
    text: str,
    model_info: Dict,
    window_size: int = config.WINDOW_SIZE_CHARS,
    overlap: int = config.OVERLAP_CHARS,
    add_space: bool = True,
    reduce: bool = True,
) -> str:
    '''슬라이딩 윈도우 방식으로 구두점을 복원합니다.'''
    if not text.strip():
        return ""
    label2id = model_info["label2id"]
    label2punc = build_label2punc(label2id, add_space=add_space, reduce=reduce)
    labels = predict_labels_sliding(
        text,
        model_info=model_info,
        window_size=window_size,
        overlap=overlap,
    )
    # 길이 차이 보정
    if len(labels) < len(text):
        labels += ["O"] * (len(text) - len(labels))
    elif len(labels) > len(text):
        labels = labels[: len(text)]
    result = ""
    for ch, lbl in zip(text, labels):
        result += ch
        punc = label2punc.get(lbl, "")
        result += punc
    return result.strip()


def run_dataset() -> None:
    '''CSV 파일을 읽어 구두점을 복원하고 결과를 저장합니다.'''
    if not config.INPUT_CSV.exists():
        print(f"입력 CSV 파일이 없습니다: {config.INPUT_CSV}")
        return
    print(f"데이터 로드: {config.INPUT_CSV}")
    df = pd.read_csv(config.INPUT_CSV)
    if 'preprocess' not in df.columns:
        print("'preprocess' 컬럼이 존재하지 않습니다.")
        return
    print(f"총 {len(df):,}개 문서")
    print(f"\n모델 준비: {config.MODEL_TAG}")
    download_model(config.MODEL_TAG, config.MODEL_CACHE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")
    model_info = load_model(config.MODEL_CACHE_DIR, device=device)
    print("\n텍스트 전처리 (구두점 제거)")
    texts = df['preprocess'].fillna("").astype(str).tolist()
    cleaned_texts = [remove_punctuation(t) for t in tqdm(texts, desc="Cleaning")]
    print(f"\n구두점 복원 시작 (윈도우={config.WINDOW_SIZE_CHARS}, 오버랩={config.OVERLAP_CHARS})")
    all_predictions = []
    for i in tqdm(range(0, len(cleaned_texts), config.DISPLAY_BATCH_SIZE), desc="Restoring"):
        batch = cleaned_texts[i: i + config.DISPLAY_BATCH_SIZE]
        batch_results = []
        for t in batch:
            if t.strip():
                restored = restore_punctuation_sliding(
                    t,
                    model_info=model_info,
                    window_size=config.WINDOW_SIZE_CHARS,
                    overlap=config.OVERLAP_CHARS,
                    add_space=True,
                    reduce=True,
                )
            else:
                restored = ""
            batch_results.append(restored)
        all_predictions.extend(batch_results)
    df['sentence'] = all_predictions
    print("\n결과 샘플 (상위 3개)")
    print(df[['doc_id', 'preprocess', 'sentence']].head(3).to_string())
    config.OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n저장 완료: {config.OUTPUT_CSV}")
    print("추가된 컬럼: 'sentence' (구두점 복원)")


def main() -> None:
    '''메인 함수: 데이터셋 구두점 복원 실행'''
    print("=" * 60)
    print("구두점 복원 작업 시작")
    print("=" * 60)
    try:
        run_dataset()
    except Exception as e:
        print(f"오류 발생: {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
    write_file(module_dir / "main.py", main_py)

    # README.md: 모듈 설명
    readme_md = """# 구두점 복원 모델

이 모듈은 한문 텍스트에서 누락된 구두점을 자동으로 복원합니다. SikuRoBERTa
토큰 분류 모델을 사용하여 문자별로 구두점을 예측하고, 슬라이딩 윈도우
방식을 적용하여 긴 텍스트도 효율적으로 처리합니다. 전체 데이터셋을
처리하려면 `main.py`를 실행하고, 단일 문자열을 복원하려면
`restore_punctuation_sliding()` 함수를 이용하세요.

## 파일 구성

- **config.py** – 모델 태그, 입력/출력 경로, 윈도우 크기 등 설정을 정의합니다.
- **main.py** – CSV 파일을 읽어 구두점을 복원하고 결과를 저장하는 실행 스크립트입니다.
- **\_\_init\_\_.py** – 패키지 초기화 모듈로, 사용 예시를 제공합니다.

## 사용법

### 1. 스크립트 실행

프로젝트 루트에서 다음 명령을 실행하여 데이터셋 전체의 구두점을 복원할 수 있습니다.

```bash
cd 3_model/nlp/punctuation_restoration
python main.py
```

실행 결과는 `1_data/raw_data/doc_id_transcript_with_punctuation.csv`에 저장됩니다.

### 2. 모듈 임포트 및 함수 사용

다른 파이썬 코드에서 구두점 복원 기능을 사용하려면 다음과 같이 임포트할 수 있습니다.

```python
from epitext_model.nlp.punctuation_restoration.main import (
    download_model,
    load_model,
    remove_punctuation,
    restore_punctuation_sliding,
)
from epitext_model.nlp.punctuation_restoration import config

# 모델 다운로드 및 로드
download_model(config.MODEL_TAG, config.MODEL_CACHE_DIR)
model_info = load_model(config.MODEL_CACHE_DIR, device="cpu")

# 단일 문장 처리 예시
cleaned = remove_punctuation("예시 문장입니다")
restored = restore_punctuation_sliding(cleaned, model_info)
print(restored)
```

`epitext_model` 패키지는 `setup_import_aliases.py`를 실행하여 생성되는
alias입니다. alias를 사용하지 않는 경우에는 `importlib`로 직접 로드할 수도
있습니다.

## 요구 사항

- transformers
- huggingface_hub
- pandas
- tqdm
- torch

이 모듈은 연구용 코드이며, 프로덕션 환경에서는 적절한 예외 처리와 로깅을
추가하여 사용하세요.
"""
    write_file(module_dir / "README.md", readme_md)


def create_unified_preprocessor_module(root: Path) -> None:
    """통합 이미지 전처리 모듈을 생성합니다 (1_data/preprocess/vision)."""
    module_dir = root / "1_data" / "preprocess" / "vision" / "unified_preprocessor"
    module_dir.mkdir(parents=True, exist_ok=True)

    # __init__.py: 패키지 안내와 예시
    init_py = """# -*- coding: utf-8 -*-
'''통합 이미지 전처리 패키지 초기화 모듈.

이 패키지는 하나의 입력 이미지로부터 Swin Gray(3채널)와 OCR(1채널) 이미지를
동시에 생성하는 기능을 제공합니다. `UnifiedImagePreprocessor` 클래스를 사용하거나
`preprocess_image_unified` 편의 함수를 이용할 수 있습니다.

예제:

    from epitext_data.preprocess.vision.unified_preprocessor.unified_preprocessor import (
        preprocess_image_unified, UnifiedImagePreprocessor
    )

    # 편의 함수 사용
    result = preprocess_image_unified(
        "input.jpg", "swin.jpg", "ocr.png", margin=10, use_rubbing=False
    )
    print(result)

    # 클래스 직접 사용
    prep = UnifiedImagePreprocessor(config_path="preprocessor_config.json")
    prep.preprocess_unified(
        "input.jpg", "swin.jpg", "ocr.png", margin=10, use_rubbing=False
    )

`epitext_data` alias 패키지는 `setup_import_aliases.py`를 실행하면 생성됩니다.
'''
"""
    write_file(module_dir / "__init__.py", init_py)

    # unified_preprocessor.py: 본문 코드 작성 (길어서 별도 변수로 저장)
    unified_code = '''# -*- coding: utf-8 -*-
"""Unified Image Preprocessing Module for Epitext AI Project
========================================================

이 모듈은 한자 이미지의 Swin Gray (3채널) 전처리와 OCR (1채널) 전처리를
한 번의 함수 호출로 수행합니다. OpenCV와 NumPy를 사용하여 탁본 영역
검출, 텍스트 영역 검출, 밝은/흰 배경 보장, 크롭 및 여백 조정 등을
처리합니다.

`UnifiedImagePreprocessor` 클래스를 사용하거나 `preprocess_image_unified`
편의 함수를 통해 즉시 사용할 수 있습니다. 설정 값은 JSON 파일로
제공할 수 있으며, 기본 값은 코드 내부에 정의되어 있습니다.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 기본 설정값
DEFAULT_MARGIN = 10
DEFAULT_BRIGHTNESS_THRESHOLD = 127
DEFAULT_RUBBING_MIN_AREA_RATIO = 0.1
DEFAULT_TEXT_MIN_AREA = 16
DEFAULT_TEXT_AREA_RATIO = 0.00005
DEFAULT_MORPHOLOGY_KERNEL_SIZE = (2, 2)
DEFAULT_MORPHOLOGY_CLOSE_ITERATIONS = 3
DEFAULT_MORPHOLOGY_OPEN_ITERATIONS = 2
DEFAULT_RUBBING_KERNEL_SIZE = (5, 5)
DEFAULT_RUBBING_CLOSE_ITERATIONS = 10
DEFAULT_RUBBING_OPEN_ITERATIONS = 5

class UnifiedImagePreprocessor:
    """통합 이미지 전처리 클래스 (Swin + OCR).

    한 번의 처리로 Swin Gray와 OCR용 이미지를 모두 생성합니다.
    """
    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = self._load_config(config_path)
        logger.info("[INIT] UnifiedImagePreprocessor 초기화 완료")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """설정 파일을 로드합니다."""
        default_config = {
            "margin": DEFAULT_MARGIN,
            "brightness_threshold": DEFAULT_BRIGHTNESS_THRESHOLD,
            "rubbing_min_area_ratio": DEFAULT_RUBBING_MIN_AREA_RATIO,
            "text_min_area": DEFAULT_TEXT_MIN_AREA,
            "text_area_ratio": DEFAULT_TEXT_AREA_RATIO,
            "morphology_kernel_size": DEFAULT_MORPHOLOGY_KERNEL_SIZE,
            "morphology_close_iterations": DEFAULT_MORPHOLOGY_CLOSE_ITERATIONS,
            "morphology_open_iterations": DEFAULT_MORPHOLOGY_OPEN_ITERATIONS,
            "rubbing_kernel_size": DEFAULT_RUBBING_KERNEL_SIZE,
            "rubbing_close_iterations": DEFAULT_RUBBING_CLOSE_ITERATIONS,
            "rubbing_open_iterations": DEFAULT_RUBBING_OPEN_ITERATIONS,
        }
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    logger.info(f"[CONFIG] 설정 파일 로드: {config_path}")
            except Exception as e:
                logger.warning(f"[CONFIG] 설정 파일 로드 실패: {e} - 기본 설정 사용")
        return default_config

    def _find_rubbing_bbox(self, gray_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """탁본 영역을 검출합니다 (큰 어두운 사각형)."""
        H_img, W_img = gray_image.shape
        _, dark_mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        kernel_rub = np.ones(self.config["rubbing_kernel_size"], np.uint8)
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_CLOSE, kernel_rub,
            iterations=self.config["rubbing_close_iterations"]
        )
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_OPEN, kernel_rub,
            iterations=self.config["rubbing_open_iterations"]
        )
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        min_area = (H_img * W_img) * self.config["rubbing_min_area_ratio"]
        if area < min_area:
            return None
        return cv2.boundingRect(largest)

    def _find_text_bbox(self, gray_image: np.ndarray) -> Tuple[int, int, int, int]:
        """텍스트 영역을 검출합니다."""
        H_img, W_img = gray_image.shape
        _, binary = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel_morph = np.ones(self.config["morphology_kernel_size"], np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel_morph,
            iterations=self.config["morphology_close_iterations"]
        )
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel_morph,
            iterations=self.config["morphology_open_iterations"]
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(
            self.config["text_min_area"],
            int((H_img * W_img) * self.config["text_area_ratio"])
        )
        valid_contours = [
            cnt for cnt in contours
            if cv2.contourArea(cv2.boundingRect(cnt)) >= min_area
        ]
        if valid_contours:
            all_points = np.vstack(valid_contours)
            return cv2.boundingRect(all_points)
        return (0, 0, W_img, H_img)

    def _apply_margin(
        self,
        bbox: Tuple[int, int, int, int],
        gray_image: np.ndarray,
        margin_val: int,
    ) -> Tuple[int, int, int, int]:
        """여백을 추가합니다."""
        x, y, w, h = bbox
        H_img, W_img = gray_image.shape
        x_new = max(0, x - margin_val)
        y_new = max(0, y - margin_val)
        w_new = min(W_img - x_new, w + 2 * margin_val)
        h_new = min(H_img - y_new, h + 2 * margin_val)
        return (x_new, y_new, w_new, h_new)

    def _ensure_bright_background(
        self,
        gray_cropped: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """밝은 배경을 보장합니다 (Swin용)."""
        mean_brightness = np.mean(gray_cropped)
        is_inverted = False
        if mean_brightness < self.config["brightness_threshold"]:
            gray_bright = cv2.bitwise_not(gray_cropped)
            is_inverted = True
        else:
            gray_bright = gray_cropped.copy()
        final_brightness = np.mean(gray_bright)
        if final_brightness < self.config["brightness_threshold"]:
            gray_bright = cv2.bitwise_not(gray_bright)
            is_inverted = not is_inverted
            final_brightness = np.mean(gray_bright)
        return gray_bright, {
            "mean_brightness_before": float(mean_brightness),
            "mean_brightness_after": float(final_brightness),
            "is_inverted": is_inverted,
            "is_bright_bg": final_brightness >= self.config["brightness_threshold"],
        }

    def _ensure_white_background(
        self,
        gray_cropped: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """흰 배경을 보장합니다 (OCR용)."""
        _, binary = cv2.threshold(
            gray_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mean_brightness = np.mean(binary)
        if mean_brightness < self.config["brightness_threshold"]:
            binary_final = cv2.bitwise_not(binary)
            polarity = "inverted"
        else:
            binary_final = binary
            polarity = "normal"
        final_brightness = np.mean(binary_final)
        return binary_final, {
            "mean_brightness_before": float(mean_brightness),
            "mean_brightness_after": float(final_brightness),
            "polarity": polarity,
            "is_white_bg": final_brightness > self.config["brightness_threshold"],
        }

    def preprocess_unified(
        self,
        input_image_path: str,
        output_swin_path: str,
        output_ocr_path: str,
        margin: Optional[int] = None,
        use_rubbing: bool = False,
    ) -> Dict:
        """Swin Gray와 OCR 이미지를 동시에 생성하는 통합 전처리를 수행합니다."""
        margin_val = margin or self.config["margin"]
        try:
            img_bgr = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"이미지 로드 실패: {input_image_path}")
            original_shape = img_bgr.shape
            logger.info(f"[LOAD] 이미지 로드: {input_image_path} {original_shape}")
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if use_rubbing:
                detected_bbox = self._find_rubbing_bbox(gray)
                region_type = "rubbing"
                logger.info("[DETECT] 탁본 영역 검출 모드")
            else:
                detected_bbox = None
                region_type = "text"
                logger.info("[DETECT] 텍스트 영역 검출 모드")
            H_img, W_img = gray.shape
            if detected_bbox is not None:
                bbox_final = self._apply_margin(detected_bbox, gray, margin_val)
                logger.info(f"[DETECT] {region_type} 영역 검출: {bbox_final}")
            else:
                if use_rubbing:
                    bbox_final = (0, 0, W_img, H_img)
                    logger.warning("[DETECT] 탁본 미검출 - 전체 이미지 사용")
                else:
                    bbox_text = self._find_text_bbox(gray)
                    bbox_final = self._apply_margin(bbox_text, gray, margin_val)
                    logger.info(f"[DETECT] 텍스트 영역 검출: {bbox_final}")
            x, y, w, h = bbox_final
            gray_cropped = gray[y : y + h, x : x + w]
            logger.info(f"[CROP] 크롭 완료: {gray_cropped.shape}")
            gray_bright, info_swin = self._ensure_bright_background(gray_cropped)
            swin_output_3ch = cv2.cvtColor(gray_bright, cv2.COLOR_GRAY2BGR)
            binary_final, info_ocr = self._ensure_white_background(gray_cropped)
            out_swin = Path(output_swin_path)
            out_swin.parent.mkdir(parents=True, exist_ok=True)
            swin_ok = cv2.imwrite(str(out_swin), swin_output_3ch)
            out_ocr = Path(output_ocr_path)
            out_ocr.parent.mkdir(parents=True, exist_ok=True)
            ocr_ok = cv2.imwrite(str(out_ocr), binary_final)
            if not swin_ok or not ocr_ok:
                raise ValueError("이미지 저장 실패")
            logger.info(f"[SAVE] Swin 저장: {out_swin}")
            logger.info(f"[SAVE] OCR 저장: {out_ocr}")
            return {
                "success": True,
                "version": "Unified Swin Gray + OCR (v1.0.0)",
                "original_shape": original_shape,
                "bbox": bbox_final,
                "region_type": region_type,
                "region_detected": detected_bbox is not None,
                "swin": {
                    "output_path": str(out_swin).replace("\\", "/"),
                    "output_shape": swin_output_3ch.shape,
                    "color_type": "Grayscale 3채널 (B=G=R)",
                    "is_inverted": info_swin["is_inverted"],
                    "mean_brightness_before": info_swin["mean_brightness_before"],
                    "mean_brightness_after": info_swin["mean_brightness_after"],
                    "is_bright_bg": info_swin["is_bright_bg"],
                },
                "ocr": {
                    "output_path": str(out_ocr).replace("\\", "/"),
                    "output_shape": binary_final.shape,
                    "polarity": info_ocr["polarity"],
                    "mean_brightness_before": info_ocr["mean_brightness_before"],
                    "mean_brightness_after": info_ocr["mean_brightness_after"],
                    "is_white_bg": info_ocr["is_white_bg"],
                },
                "message": "통합 전처리 완료 (Swin + OCR)",
            }
        except Exception as e:
            logger.error(f"[ERROR] 통합 전처리 실패: {e}")
            return {"success": False, "message": str(e)}

_global_preprocessor: Optional[UnifiedImagePreprocessor] = None

def get_preprocessor(config_path: Optional[str] = None) -> UnifiedImagePreprocessor:
    """전역 전처리기 인스턴스를 반환합니다."""
    global _global_preprocessor
    if _global_preprocessor is None:
        _global_preprocessor = UnifiedImagePreprocessor(config_path)
    return _global_preprocessor

def preprocess_image_unified(
    input_path: str,
    output_swin_path: str,
    output_ocr_path: str,
    margin: Optional[int] = None,
    use_rubbing: bool = False,
) -> Dict:
    """편의 함수: 통합 이미지 전처리를 수행합니다."""
    prep = get_preprocessor()
    return prep.preprocess_unified(
        input_path,
        output_swin_path,
        output_ocr_path,
        margin,
        use_rubbing,
    )

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("[TEST] Unified Image Preprocessor 테스트 시작")
    logger.info("=" * 80)
    try:
        prep = UnifiedImagePreprocessor()
        result = prep.preprocess_unified(
            "test_input.jpg",
            "test_swin.jpg",
            "test_ocr.png",
        )
        if result["success"]:
            logger.info("[TEST] 통합 전처리 성공")
            logger.info(f"[TEST] Swin 결과: {result['swin']['output_path']}")
            logger.info(f"[TEST] OCR 결과:  {result['ocr']['output_path']}")
        else:
            logger.error(f"[TEST] 실패: {result['message']}")
    except Exception as e:
        logger.error(f"[TEST] 예외 발생: {e}")
    logger.info("=" * 80)
'''
    write_file(module_dir / "unified_preprocessor.py", unified_code)

    # preprocessor_config.json: 기본 설정 및 주석 포함 JSON
    config_json: Dict = {
        "module_info": {
            "name": "preprocessor",
            "version": "1.0.0",
            "description": "탁본 이미지 전처리 모듈",
            "status": "Production Ready",
            "created": "2025-12-10",
        },
        "preprocessing_pipeline": {
            "enabled_steps": ["denoise", "enhance_contrast", "binarize", "morphological_operations"],
            "description": "1단계: 노이즈 제거, 2단계: 대비 향상, 3단계: 이진화, 4단계: 형태학적 연산 순서대로 실행됩니다.",
        },
        "denoise_config": {
            "method": "bilateral",
            "bilateral_d": 9,
            "bilateral_sigma_color": 75,
            "bilateral_sigma_space": 75,
        },
        "contrast_enhancement": {
            "method": "clahe",
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": [8, 8],
        },
        "binarization": {
            "method": "otsu",
            "otsu_offset": 0,
        },
        "morphological_operations": {
            "kernel_size": [3, 3],
            "operations": [
                {"type": "close", "iterations": 1},
                {"type": "open", "iterations": 1},
            ],
        },
        "output": {
            "save_intermediate": False,
            "intermediate_dir": "./preprocessing_steps/",
            "output_format": "png",
        },
        "advanced": {
            "enable_background_removal": False,
            "background_removal_method": "grabcut",
            "enable_skew_correction": False,
            "skew_correction_threshold": 0.5,
            "enable_border_removal": False,
            "border_margin": 10,
        },
        "performance": {
            "resize_before_processing": False,
            "target_width": 1024,
            "use_gpu": False,
            "max_image_size": [4096, 4096],
        },
    }
    write_file(module_dir / "preprocessor_config.json", json.dumps(config_json, ensure_ascii=False, indent=2))

    # README.md: 전처리 모듈 설명
    readme_md = """# 통합 이미지 전처리 모듈

이 모듈은 하나의 입력 이미지에서 Swin Gray(밝은 배경 3채널)와 OCR(흰 배경 1채널) 이미지를
동시에 생성합니다. 탁본 검출, 텍스트 검출, 밝기/배경 보정, 크롭 등
다양한 처리를 포함하여 단일 호출로 최종 결과를 얻을 수 있습니다.

## 파일 구성

- **unified_preprocessor.py** – 통합 전처리 클래스(`UnifiedImagePreprocessor`)와 편의 함수(`preprocess_image_unified`)를 정의합니다.
- **preprocessor_config.json** – 전처리 파라미터와 파이프라인 설정을 담은 JSON 파일입니다.
- **\_\_init\_\_.py** – 패키지 초기화 모듈로 사용 예시를 제공합니다.

## 사용법

### 1. 편의 함수 사용

```python
from epitext_data.preprocess.vision.unified_preprocessor.unified_preprocessor import preprocess_image_unified

result = preprocess_image_unified(
    "input.jpg",
    "output_swin.jpg",
    "output_ocr.png",
    margin=10,
    use_rubbing=False,
)
if result["success"]:
    print("출력 경로:", result["swin"]["output_path"], result["ocr"]["output_path"])
else:
    print("오류:", result["message"])
```

### 2. 클래스 직접 사용

```python
from epitext_data.preprocess.vision.unified_preprocessor.unified_preprocessor import UnifiedImagePreprocessor

prep = UnifiedImagePreprocessor(config_path="preprocessor_config.json")
prep.preprocess_unified(
    "input.jpg",
    "swin.jpg",
    "ocr.png",
    margin=10,
    use_rubbing=False,
)
```

### 3. 설정 파일 편집

`preprocessor_config.json` 파일을 열어 `denoise_config`, `contrast_enhancement` 등
필요한 파라미터를 수정할 수 있습니다. 설명용 항목은 실제 설정에
영향을 주지 않습니다.

## 요구 사항

- opencv-python>=4.8.0
- numpy>=1.24.0

이 모듈은 연구용 코드이며, 서비스 환경에서는 적절한 예외 처리와
로깅을 추가하여 사용하세요.
"""
    write_file(module_dir / "README.md", readme_md)


def main() -> None:
    root = Path(__file__).resolve().parent
    create_punctuation_module(root)
    create_unified_preprocessor_module(root)
    print("모듈 설치가 완료되었습니다. 변경 사항을 확인한 후 Git에 커밋하세요.")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
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
    print(f"
모델 준비: {config.MODEL_TAG}")
    download_model(config.MODEL_TAG, config.MODEL_CACHE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")
    model_info = load_model(config.MODEL_CACHE_DIR, device=device)
    print("
텍스트 전처리 (구두점 제거)")
    texts = df['preprocess'].fillna("").astype(str).tolist()
    cleaned_texts = [remove_punctuation(t) for t in tqdm(texts, desc="Cleaning")]
    print(f"
구두점 복원 시작 (윈도우={config.WINDOW_SIZE_CHARS}, 오버랩={config.OVERLAP_CHARS})")
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
    print("
결과 샘플 (상위 3개)")
    print(df[['doc_id', 'preprocess', 'sentence']].head(3).to_string())
    config.OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"
저장 완료: {config.OUTPUT_CSV}")
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

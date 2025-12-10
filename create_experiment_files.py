import os
import re

# Base path for experiments
BASE_PATH = r"C:\hanja_data\Epitext_Project\5_docs\experiments"

# List of experiments with their README info and code files.
# For brevity, only the major experiments are populated with full code; others include placeholders.
experiments = [
    {
        "path": "nlp/sikuroberta/MLM_성능_비교",
        "title": "MLM 성능 비교",
        "purpose": "BERT 기반 한문 MLM 모델들의 성능을 비교 평가",
        "summary": "모델별 Top-K 정확도 및 질적 비교를 통해 성능 차이를 분석",
        "files": {
            "main.py": r'''# -*- coding: utf-8 -*-
"""
Historical BERT MLM Evaluation Framework
========================================
Description:
    한문 및 고전 문헌 처리에 특화된 3가지 BERT 모델의 
    Masked Language Modeling (MLM) 성능을 비교 평가하는 자동화 스크립트입니다.

    [비교 대상 모델]
    1. SillokBERT (HuggingFace): 조선왕조실록 기반
    2. SikuRoBERTa (HuggingFace): 사고전서 기반
    3. HUE (Local): 한문 고전 문헌 모델 (직접 다운로드 필요)

Features:
    - 상대 경로 지원: Git Clone 후 즉시 실행 가능
    - 자동 데이터셋 생성: 한자(Hanja) 식별 및 무작위 마스킹
    - 이중 평가: 정량적(Top-K Acc) 및 정성적(Side-by-side) 비교

Author: [Your Name]
Date: 2025-12-09
Version: 1.2.0 (Relative Path Support)
"""

import os
import re
import time
import random
import logging
from typing import List, Dict, Optional, Any

import torch
from transformers import pipeline, Pipeline

# -----------------------------------------------------------------------------
# 1. Logger Setup (로깅 설정)
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. Configuration (환경 설정)
# -----------------------------------------------------------------------------
class Config:
    """프로젝트 실행을 위한 전역 설정 클래스"""
    # [Project Root] 현재 실행 파일(main.py)의 위치를 기준으로 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 모델 경로
    SILLOK_MODEL_PATH: str = "ddokbaro/SillokBert"
    SIKU_MODEL_PATH: str = "SIKU-BERT/sikuroberta"
    HUE_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "HUE")
    TEST_DATA_PATH: str = os.path.join(BASE_DIR, "data", "sillok_test.txt")
    TOP_K: int = 5
    NUM_SAMPLES: int = 1000
    DEVICE: int = 0 if torch.cuda.is_available() else -1

# -----------------------------------------------------------------------------
# 3. Model Handler
# -----------------------------------------------------------------------------
class ModelHandler:
    @staticmethod
    def load_pipeline(model_path: str, name: str, top_k: int = 5) -> Optional[Pipeline]:
        """모델 경로를 받아 fill-mask 파이프라인을 생성합니다."""
        is_local = os.path.isabs(model_path) or os.path.sep in model_path
        if is_local:
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                logger.warning(f"⚠️  [Skip] '{name}' 모델을 찾을 수 없습니다.")
                logger.warning(f"    경로: {model_path}")
                return None
        try:
            logger.info(f"⏳ [{name}] 모델 로딩 중...")
            pipe = pipeline(
                "fill-mask",
                model=model_path,
                tokenizer=model_path,
                device=Config.DEVICE,
                top_k=top_k
            )
            logger.info(f"✅ [{name}] 로딩 완료")
            return pipe
        except Exception as e:
            logger.error(f"❌ [{name}] 로딩 실패: {e}")
            return None

# -----------------------------------------------------------------------------
# 4. Data Processor
# -----------------------------------------------------------------------------
class DataProcessor:
    @staticmethod
    def load_and_clean(file_path: str) -> List[str]:
        """텍스트 파일을 읽고 XML 태그 및 특수문자를 제거합니다."""
        if not os.path.exists(file_path):
            logger.error(f"❌ 데이터 파일을 찾을 수 없습니다: {file_path}")
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"❌ 파일 읽기 오류: {e}")
            return []
        clean_lines = []
        patterns = [
            re.compile(r'<[^>]+>'),
            re.compile(r'\([^)]+\)'),
            re.compile(r'\[[^\]]+\]')
        ]
        for line in lines:
            for pat in patterns:
                line = pat.sub('', line)
            line = line.strip()
            if line:
                clean_lines.append(line)
        logger.info(f"📂 데이터 로드 완료: {len(clean_lines)} 문장")
        return clean_lines

    @staticmethod
    def create_masked_dataset(lines: List[str], num_samples: int) -> List[Dict[str, str]]:
        """문장 내 한자(Hanja)를 식별하여 무작위로 [MASK] 토큰을 삽입합니다."""
        if not lines:
            return []
        target_lines = lines
        if 0 < num_samples < len(lines):
            target_lines = random.sample(lines, num_samples)
        dataset = []
        for line in target_lines:
            try:
                hanja_indices = [i for i, char in enumerate(line) if '\u4e00' <= char <= '\u9fff']
                if not hanja_indices:
                    continue
                mask_idx = random.choice(hanja_indices)
                answer = line[mask_idx]
                masked_text = line[:mask_idx] + "[MASK]" + line[mask_idx+1:]
                dataset.append({"masked_text": masked_text, "answer": answer})
            except Exception:
                continue
        logger.info(f"🛠️  마스킹 데이터셋 생성 완료: 유효 샘플 {len(dataset)}개")
        return dataset

# -----------------------------------------------------------------------------
# 5. Evaluator
# -----------------------------------------------------------------------------
class Evaluator:
    @staticmethod
    def evaluate_quantitative(name: str, pipe: Pipeline, dataset: List[Dict], top_k: int) -> Dict[str, Any]:
        hits_top1 = 0
        hits_topk = 0
        valid_count = 0
        max_len = getattr(pipe.tokenizer, 'model_max_length', 512)
        logger.info(f"📊 [{name}] 평가 시작 (샘플 {len(dataset)}개)...")
        start_time = time.time()
        for item in dataset:
            text = item["masked_text"]
            answer = item["answer"]
            if len(text) > max_len:
                continue
            try:
                results = pipe(text, top_k=top_k)
                if results and isinstance(results[0], list):
                    results = results[0]
                preds = [res['token_str'].strip() for res in results]
                if not preds:
                    continue
                if preds[0] == answer:
                    hits_top1 += 1
                if answer in preds:
                    hits_topk += 1
                valid_count += 1
            except:
                continue
        duration = time.time() - start_time
        if valid_count == 0:
            return {"name": name, "acc_top1": 0.0, "acc_topk": 0.0, "count": 0}
        return {
            "name": name,
            "acc_top1": (hits_top1 / valid_count) * 100,
            "acc_topk": (hits_topk / valid_count) * 100,
            "count": valid_count,
            "duration": duration
        }

    @staticmethod
    def compare_qualitative(sample: Dict, pipes: Dict[str, Pipeline], top_k: int):
        text = sample["masked_text"]
        answer = sample["answer"]
        print("\n" + "="*80)
        print(f"🧐 [Qualitative Analysis] 모델별 예측 결과 비교")
        print(f"• Input Context: {text}")
        print(f"• Correct Answer: [{answer}]")
        print("-" * 80)
        model_names = list(pipes.keys())
        col_width = 20
        header = f"| Rank |"
        for name in model_names:
            display_name = name[:col_width]
            header += f" {display_name:<{col_width}} |"
        print(header)
        print("|:----:|" + ("-" * (col_width+2) + "|") * len(model_names))
        results_map = {}
        for name, pipe in pipes.items():
            try:
                res = pipe(text, top_k=top_k)
                if res and isinstance(res[0], list):
                    res = res[0]
                results_map[name] = res
            except:
                results_map[name] = []
        for i in range(top_k):
            row_str = f"| {i+1:<4} |"
            for name in model_names:
                res_list = results_map.get(name, [])
                if i < len(res_list):
                    token = res_list[i]['token_str'].strip()
                    score = res_list[i]['score']
                    display = f"{token} ({score:.3f})"
                else:
                    display = "-"
                row_str += f" {display:<{col_width}} |"
            print(row_str)
        print("="*80 + "\n")

# -----------------------------------------------------------------------------
# 6. Main Execution
# -----------------------------------------------------------------------------
def main():
    print("🚀 [Start] Historical BERT Comparison Framework")
    print(f"📂 Project Root: {Config.BASE_DIR}")
    pipelines = {}
    p1 = ModelHandler.load_pipeline(Config.SILLOK_MODEL_PATH, "SillokBERT", Config.TOP_K)
    if p1: pipelines["SillokBERT"] = p1
    p2 = ModelHandler.load_pipeline(Config.SIKU_MODEL_PATH, "SikuRoBERTa", Config.TOP_K)
    if p2: pipelines["SikuRoBERTa"] = p2
    p3 = ModelHandler.load_pipeline(Config.HUE_MODEL_PATH, "HUE (Local)", Config.TOP_K)
    if p3: pipelines["HUE (Local)"] = p3
    if not pipelines:
        logger.error("❌ 로드된 모델이 하나도 없습니다. 프로그램을 종료합니다.")
        return
    raw_lines = DataProcessor.load_and_clean(Config.TEST_DATA_PATH)
    if not raw_lines:
        return
    test_dataset = DataProcessor.create_masked_dataset(raw_lines, Config.NUM_SAMPLES)
    if not test_dataset:
        logger.error("❌ 테스트 데이터셋 생성 실패.")
        return
    print("\n" + "#"*60)
    print("📊 정량 평가 결과 (Quantitative Evaluation)")
    print("#"*60)
    for name, pipe in pipelines.items():
        res = Evaluator.evaluate_quantitative(name, pipe, test_dataset, Config.TOP_K)
        print(f"\n🏷️  Model: {name}")
        print(f"   - Top-1 Accuracy: {res['acc_top1']:.2f}%")
        print(f"   - Top-{Config.TOP_K} Accuracy: {res['acc_topk']:.2f}%")
        print(f"   - Valid Samples : {res['count']}")
    if test_dataset:
        Evaluator.compare_qualitative(test_dataset[0], pipelines, Config.TOP_K)
    print("\n🎉 모든 평가가 완료되었습니다.")

if __name__ == "__main__":
    main()
'''
        }
    },
    {
        "path": "nlp/sikuroberta/판독문_전체_텍스트_중복_탁본_제외",
        "title": "판독문 전체 텍스트 중복 탁본 제외",
        "purpose": "조선왕조실록 판독문 전체 텍스트에서 중복된 탁본을 제외한 데이터셋 구축",
        "summary": "토크나이저 Vocab 확장 후 토큰화와 모델 학습을 수행하여 중복 제거 시 성능을 평가",
        "files": {
            "preprocess.py": r'''import os
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset

def expand_vocab(tokenizer, data_path):
    """
    데이터셋에 존재하는 모든 문자를 확인하고, 기존 토크나이저 vocab에 없는 토큰을 추가합니다.
    """
    print(f"[*] Vocab 확장을 위한 데이터 스캔 중: {data_path}")
    unique_chars_in_data = set()
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                unique_chars_in_data.update(list(line.strip()))
    except FileNotFoundError:
        raise FileNotFoundError(f"오류: {data_path} 파일을 찾을 수 없습니다.")
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = list(unique_chars_in_data - existing_vocab)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"[+] Vocab 확장 완료. 추가된 토큰 수: {len(new_tokens)}")
        print(f"[+] 최종 Vocab 크기: {len(tokenizer)}")
    else:
        print("[=] 모든 문자가 이미 Vocab에 존재합니다.")
    return tokenizer

def preprocess_data(args):
    print(f"[*] 토크나이저 로드 중: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer = expand_vocab(tokenizer, args.input_file)
    print("[*] 데이터셋 로드 중...")
    dataset = load_dataset('text', data_files={'train': args.input_file})
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding='max_length',
            return_special_tokens_mask=True
        )
    print("[*] 토크나이징 진행 중...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    os.makedirs(args.output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(args.output_dir)
    print(f"[+] 전처리 완료. 데이터셋 저장 경로: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SikuRoBERTa Data Preprocessing")
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikuroberta", help="HuggingFace 모델명")
    parser.add_argument("--input_file", type=str, default="./data/preprocess_txt.txt", help="학습할 텍스트 파일 경로")
    parser.add_argument("--output_dir", type=str, default="./data/tokenized_dataset", help="토큰화된 데이터셋 저장 경로")
    parser.add_argument("--max_length", type=int, default=512, help="시퀀스 최대 길이")
    args = parser.parse_args()
    preprocess_data(args)
''',
            "train.py": r'''import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, 
    BertForMaskedLM, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
from datasets import load_from_disk, DatasetDict

def load_tokenizer_and_model(model_name, data_file_path):
    print(f"[*] 토크나이저 로드 (use_fast=False): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    unique_chars_in_data = set()
    with open(data_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            unique_chars_in_data.update(list(line.strip()))
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = list(unique_chars_in_data - existing_vocab)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"[+] Vocab 확장 적용 완료 ({len(new_tokens)} 개 추가)")
    print("[*] MaskedLM 모델 로드 중...")
    model = BertForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def plot_loss_graph(log_history, save_path):
    logs_df = pd.DataFrame(log_history)
    train_logs = logs_df[logs_df["loss"].notna()]
    eval_logs = logs_df[logs_df["eval_loss"].notna()]
    plt.figure(figsize=(12, 6))
    plt.plot(train_logs["step"], train_logs["loss"], label="Training Loss", color='blue', alpha=0.6)
    plt.plot(eval_logs["step"], eval_logs["eval_loss"], label="Validation Loss", color='red', marker='o', linestyle='--')
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss vs Validation Loss")
    plt.legend(); plt.grid(True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[+] 학습 그래프 저장 완료: {save_path}")

def train_model(args):
    tokenizer, model = load_tokenizer_and_model(args.model_name, args.raw_data_path)
    try:
        loaded_dataset = load_from_disk(args.dataset_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {args.dataset_path}. preprocess.py를 먼저 실행하세요.")
    if 'validation' not in loaded_dataset.keys():
        print("[*] 데이터셋 분할 시작 (8:1:1)...")
        train_testval = loaded_dataset['train'].train_test_split(test_size=0.2, seed=42)
        test_val = train_testval['test'].train_test_split(test_size=0.5, seed=42)
        tokenized_dataset = DatasetDict({
            'train': train_testval['train'],
            'validation': test_val['train'],
            'test': test_val['test']
        })
    else:
        tokenized_dataset = loaded_dataset
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        fp16=True,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    print("[*] 학습 시작...")
    trainer.train()
    os.makedirs(os.path.dirname(args.graph_path), exist_ok=True)
    plot_loss_graph(trainer.state.log_history, args.graph_path)
    print("[*] 최종 Test Set 평가 진행...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    print(f"[Result] Test Set 결과: {test_results}")
    trainer.save_model(args.final_model_path)
    tokenizer.save_pretrained(args.final_model_path)
    print(f"[+] 최종 모델 저장 완료: {args.final_model_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="SikuRoBERTa Fine-Tuning")
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikuroberta")
    parser.add_argument("--raw_data_path", type=str, default="./data/preprocess_txt.txt", help="Vocab 확장을 위한 원본 텍스트")
    parser.add_argument("--dataset_path", type=str, default="./data/tokenized_dataset", help="전처리된 데이터셋 경로")
    parser.add_argument("--output_dir", type=str, default="./output/checkpoints")
    parser.add_argument("--final_model_path", type=str, default="./output/final_model")
    parser.add_argument("--graph_path", type=str, default="./output/loss_graph.png")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
'''
        }
    },
    {
        "path": "nlp/sikuroberta/판독문_전체_텍스트_중복_탁본_포함",
        "title": "판독문 전체 텍스트 중복 탁본 포함",
        "purpose": "중복 탁본을 포함한 전체 판독문 데이터셋으로 MLM 학습",
        "summary": "중복 데이터가 포함된 텍스트를 블록 단위로 그룹화하여 토크나이징 후 MLM 학습을 진행",
        "files": {
            "preprocess.py": r'''import os
import argparse
from itertools import chain
from transformers import AutoTokenizer
from datasets import load_dataset

def expand_vocab(tokenizer, data_path):
    print(f"[*] Scanning data for vocabulary expansion: {data_path}")
    unique_chars_in_data = set()
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                unique_chars_in_data.update(list(line.strip()))
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File {data_path} not found.")
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = list(unique_chars_in_data - existing_vocab)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"[+] Added {len(new_tokens)} new tokens to vocabulary.")
    else:
        print("[=] No new tokens needed. Vocabulary is complete.")
    return tokenizer

def preprocess_and_group(args):
    print(f"[*] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer = expand_vocab(tokenizer, args.input_file)
    print("[*] Loading text dataset...")
    dataset = load_dataset('text', data_files={'train': args.input_file})
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=True
        )
    print("[*] Tokenizing raw text...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= args.block_size:
            total_length = (total_length // args.block_size) * args.block_size
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    print(f"[*] Grouping text into chunks of {args.block_size} tokens...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc
    )
    print(f"[+] Original count: {len(dataset['train'])}")
    print(f"[+] Grouped count : {len(lm_datasets['train'])}")
    lm_datasets.save_to_disk(args.output_dir)
    print(f"[+] Saved processed dataset to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SikuRoBERTa Data Grouping & Preprocessing")
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikuroberta", help="HuggingFace model name")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw .txt file")
    parser.add_argument("--output_dir", type=str, default="./data/processed_dataset", help="Path to save tokenized data")
    parser.add_argument("--block_size", type=int, default=256, help="Token block size (chunk size)")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of CPU processes for mapping")
    args = parser.parse_args()
    preprocess_and_group(args)
''',
            "train.py": r'''import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, 
    BertForMaskedLM, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
from datasets import load_from_disk, DatasetDict

def load_resources(args):
    print(f"[*] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    unique_chars_in_data = set()
    if os.path.exists(args.raw_data_path):
        with open(args.raw_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                unique_chars_in_data.update(list(line.strip()))
        existing_vocab = set(tokenizer.get_vocab().keys())
        new_tokens = list(unique_chars_in_data - existing_vocab)
        if new_tokens:
            tokenizer.add_tokens(new_tokens)
            print(f"[+] Vocab expanded by {len(new_tokens)} tokens.")
    else:
        print("[!] Warning: Raw data path not found. Skipping Vocab check.")
    print("[*] Loading Model (BertForMaskedLM)...")
    model = BertForMaskedLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def prepare_dataset(dataset_path):
    print(f"[*] Loading dataset from: {dataset_path}")
    try:
        loaded_dataset = load_from_disk(dataset_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run preprocess.py first.")
    if 'validation' not in loaded_dataset.keys():
        print("[*] Splitting dataset (80% Train, 10% Valid, 10% Test)...")
        train_testval = loaded_dataset['train'].train_test_split(test_size=0.2, seed=42)
        test_val = train_testval['test'].train_test_split(test_size=0.5, seed=42)
        return DatasetDict({
            'train': train_testval['train'],
            'validation': test_val['train'],
            'test': test_val['test']
        })
    return loaded_dataset

def plot_training_history(log_history, save_path):
    logs_df = pd.DataFrame(log_history)
    train_logs = logs_df[logs_df["loss"].notna()]
    eval_logs = logs_df[logs_df["eval_loss"].notna()]
    plt.figure(figsize=(12, 6))
    plt.plot(train_logs["step"], train_logs["loss"], label="Training Loss", color='blue', alpha=0.6)
    plt.plot(eval_logs["step"], eval_logs["eval_loss"], label="Validation Loss", color='red', marker='o', linestyle='--')
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss vs Validation Loss")
    plt.legend(); plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[+] Loss graph saved to: {save_path}")

def main(args):
    tokenizer, model = load_resources(args)
    tokenized_dataset = prepare_dataset(args.dataset_path)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        fp16=True,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    print("[*] Starting training...")
    trainer.train()
    plot_training_history(trainer.state.log_history, args.graph_path)
    print("[*] Evaluating on Test Set...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    print(f"[Result] {test_results}")
    trainer.save_model(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    print(f"[+] Final model saved to: {args.final_model_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="SikuRoBERTa Training Script")
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikuroberta")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to raw txt for vocab check")
    parser.add_argument("--dataset_path", type=str, default="./data/processed_dataset", help="Path to processed (grouped) dataset")
    parser.add_argument("--output_dir", type=str, default="./output/checkpoints")
    parser.add_argument("--final_model_dir", type=str, default="./output/final_model")
    parser.add_argument("--graph_path", type=str, default="./output/loss_graph.png")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
'''
        }
    },
    {
        "path": "nlp/sikuroberta/문장_학습",
        "title": "문장 학습 코드 백업",
        "purpose": "문장 단위의 한문 말뭉치로 SikuRoBERTa를 학습하는 실험 코드 보관",
        "summary": "Dynamic Padding 전략을 사용하여 문장별 길이에 맞춘 학습을 수행",
        "files": {
            "utils.py": r'''import os
from transformers import AutoTokenizer

def load_tokenizer_and_expand_vocab(model_name, data_path, use_fast=False):
    print(f"[*] 토크나이저 로드 중: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    print(f"[*] Vocab 확장을 위한 데이터 스캔 중: {data_path}")
    unique_chars_in_data = set()
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                unique_chars_in_data.update(list(line.strip()))
    except FileNotFoundError:
        raise FileNotFoundError(f"오류: 데이터 파일 {data_path}을 찾을 수 없습니다.")
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = list(unique_chars_in_data - existing_vocab)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"[+] Vocab 확장 완료. 추가된 토큰 수: {len(new_tokens)}")
    else:
        print("[=] 모든 문자가 이미 Vocab에 존재합니다.")
    return tokenizer
''',
            "preprocess.py": r'''import os
import argparse
from datasets import load_dataset
from utils import load_tokenizer_and_expand_vocab

def preprocess_dynamic(args):
    tokenizer = load_tokenizer_and_expand_vocab(args.model_name, args.input_file)
    print("[*] 데이터셋 로드 중...")
    dataset = load_dataset('text', data_files={'train': args.input_file})
    print(f"[Info] 필터링 전 데이터 개수: {len(dataset['train'])}")
    dataset = dataset.filter(lambda example: len(example['text']) >= 10)
    print(f"[Info] 필터링 후(10자 이상) 데이터 개수: {len(dataset['train'])}")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_special_tokens_mask=True
        )
    print("[*] 토크나이징 진행 (padding=False 적용)...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    print("\n[Check] 샘플 데이터 길이 확인 (길이가 제각각이어야 정상):")
    for i in range(min(3, len(tokenized_dataset['train']))):
        print(f" - Sample {i} length: {len(tokenized_dataset['train'][i]['input_ids'])}")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(args.output_dir)
    print(f"[+] 전처리 완료. 저장 경로: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SikuRoBERTa Dynamic Padding Preprocessing")
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikuroberta")
    parser.add_argument("--input_file", type=str, required=True, help="원본 텍스트 파일 경로")
    parser.add_argument("--output_dir", type=str, default="./data/processed_dynamic", help="저장할 데이터셋 경로")
    parser.add_argument("--max_length", type=int, default=128, help="시퀀스 최대 길이")
    args = parser.parse_args()
    preprocess_dynamic(args)
''',
            "train.py": r'''import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    BertForMaskedLM, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk, DatasetDict
from utils import load_tokenizer_and_expand_vocab

def train_model(args):
    tokenizer = load_tokenizer_and_expand_vocab(args.model_name, args.raw_data_path)
    print("[*] 모델 로드 및 임베딩 리사이징...")
    model = BertForMaskedLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    print(f"[*] 처리된 데이터셋 로드: {args.dataset_path}")
    try:
        loaded_dataset = load_from_disk(args.dataset_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"데이터셋 없음: {args.dataset_path}. preprocess.py 먼저 실행하세요.")
    if 'validation' not in loaded_dataset.keys():
        print("[*] 데이터 분할 시작 (Train 80% / Valid 10% / Test 10%)...")
        train_testval = loaded_dataset['train'].train_test_split(test_size=0.2, seed=42)
        test_val = train_testval['test'].train_test_split(test_size=0.5, seed=42)
        tokenized_dataset = DatasetDict({
            'train': train_testval['train'],
            'validation': test_val['train'],
            'test': test_val['test']
        })
    else:
        tokenized_dataset = loaded_dataset
    if args.save_split_dataset:
        split_save_path = args.dataset_path + "_split"
        tokenized_dataset.save_to_disk(split_save_path)
        print(f"[Info] 분할된 데이터셋 별도 저장됨: {split_save_path}")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        group_by_length=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        fp16=True,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    print("[*] 학습 시작...")
    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        print(f"[Resume] 체크포인트 발견: {last_checkpoint} 에서 재개합니다.")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    def plot_loss_graph(log_history, save_path):
        logs_df = pd.DataFrame(log_history)
        train_logs = logs_df[logs_df["loss"].notna()]
        eval_logs = logs_df[logs_df["eval_loss"].notna()]
        plt.figure(figsize=(10, 6))
        plt.plot(train_logs["epoch"], train_logs["loss"], label="Training Loss", color='blue', alpha=0.6)
        plt.plot(eval_logs["epoch"], eval_logs["eval_loss"], label="Validation Loss", color='red', marker='o', linestyle='--')
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss (Dynamic Padding)")
        plt.legend(); plt.grid(True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[+] 그래프 저장 완료: {save_path}")
    plot_loss_graph(trainer.state.log_history, args.graph_path)
    trainer.save_model(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    print(f"[+] 최종 모델 저장 완료: {args.final_model_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="SikuRoBERTa Dynamic Padding Training")
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikuroberta")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Vocab 확장을 위한 원본 텍스트")
    parser.add_argument("--dataset_path", type=str, required=True, help="전처리된 데이터셋 경로")
    parser.add_argument("--output_dir", type=str, default="./output/checkpoints")
    parser.add_argument("--final_model_dir", type=str, default="./output/final_model")
    parser.add_argument("--graph_path", type=str, default="./output/loss_graph.png")
    parser.add_argument("--save_split_dataset", action="store_true", help="분할된 데이터셋을 별도로 저장할지 여부")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
'''
        }
    },
    {
        "path": "nlp/gemini/ExaOne_성능_평가",
        "title": "ExaOne 성능 평가 코드",
        "purpose": "대규모 LLM인 ExaOne-3.5 모델을 사용하여 한문 금석문의 번역 성능을 측정",
        "summary": "음독, 고유명사 추출, 번역을 모두 수행하고 Kiwi 기반 BLEU 평가로 결과를 분석",
        "files": {
            "translator.py": r'''# -*- coding: utf-8 -*-
"""
Hanja Translation Script using EXAONE Model
Author: [Your Name]
Description: 한문 금석문을 입력받아 음독, 고유명사 추출, 국역을 수행하는 스크립트입니다.
"""
import os
import re
import pandas as pd
import torch
from kiwipiepy import Kiwi
import sacrebleu
from transformers import AutoModelForCausalLM, AutoTokenizer

class Config:
    MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    INPUT_CSV_PATH = "../data/pun_ksm_gsko_filtered.csv"
    OUTPUT_CSV_PATH = "../data/translation_results_exaone_filtered.csv"
    TARGET_COUNT = 1000
    BATCH_SIZE = 50
    MAX_NEW_TOKENS = 600
    REPETITION_PENALTY = 1.2
    SYSTEM_PROMPT = """<role>
당신은 한문 금석문 번역 전문가입니다.
</role>

<task>
입력된 한자 원문을 분석하여 반드시 다음 3가지 항목으로 답변하십시오:
1. [음독]: 원문의 모든 한자를 띄어쓰기 없이 정확한 한글 독음으로만 변환
2. [고유명사 추출]: 인명, 관직, 지명, 연호를 '한글(한자)' 형태로 나열
3. [최종 번역]: 위 고유명사를 활용하여 예스러운 문어체(~하니라, ~하다)로 직역
</task>

<constraints>
- 원문에 표기되지 않은 글자를 임의로 생성하거나 유추하여 해석에 포함하지 마십시오
- 원문에 있는 결락 기호(▨)는 생략하거나 추측하지 말고, 반드시 해당 개수만큼 ▨ 기호를 그대로 유지하여 번역문에 포함하십시오.
- 설명이나 사족을 덧붙이지 마십시오
- 정확한 형식을 반드시 준수하십시오
</constraints>

<output_format>
[음독]: (한글 음독)
[고유명사 추출]: (추출된 고유명사들)
[최종 번역]: (최종 번역문)
</output_format>"""
    FEW_SHOT_EXAMPLES = """<examples>
<example>
<input>府尹嚴相公善政碑 公諱鼎耉 字重叔 歲戊戌夏四月下車 己亥冬十月以病辭歸</input>
<output>
[음독]: 부윤엄상공선정비공휘정구자중숙세무술하사월하차기해동시월이병사귀
[고유명사 추출]: 부윤(府尹), 엄상공(嚴相公), 선정비(善政碑), 휘(諱), 정구(鼎耉), 자(字), 중숙(重叔), 무술년(戊戌), 기해년(己亥)
[최종 번역]: 부윤(府尹) 엄 상공(嚴相公)의 선정비(善政碑). 공의 휘는 정구(鼎耉)이고, 자는 중숙(重叔)이다. 무술년(戊戌) 여름 4월에 부임하였고, 기해년(己亥) 겨울 10월에 병으로 사직하고 돌아갔다.
</output>
</example>
</examples>"""

class TextUtils:
    def __init__(self):
        print("⚙️ Kiwi 형태소 분석기 초기화 중...")
        self.kiwi = Kiwi()
    def get_nouns_only(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\([^)]*\)', ' ', text)
        text = re.sub(r'[^가-힣\s]', '', text)
        try:
            tokens = self.kiwi.tokenize(text)
            targets = {'NNG', 'NNP', 'NR', 'NP', 'NNB'}
            nouns = [t.form for t in tokens if t.tag in targets]
            return " ".join(nouns)
        except Exception:
            return ""
    @staticmethod
    def calculate_bleu(reference: str, hypothesis: str, noun_extractor_func) -> tuple:
        ref_nouns = noun_extractor_func(reference)
        hyp_nouns = noun_extractor_func(hypothesis)
        if not ref_nouns or not hyp_nouns:
            return 0.0, ref_nouns, hyp_nouns
        bleu = sacrebleu.sentence_bleu(hyp_nouns, [ref_nouns], tokenize='char')
        return bleu.score, ref_nouns, hyp_nouns
    @staticmethod
    def extract_translation_part(full_text: str) -> str:
        markers = ["[최종 번역]:", "[최종 번역]", "[번역]:"]
        for marker in markers:
            if marker in full_text:
                return full_text.split(marker)[1].strip()
        lines = full_text.split('\n')
        return lines[-1] if lines else full_text

class HanjaTranslator:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._load_model()
    def _load_model(self):
        print(f"⚙️ 모델 로딩 중... ({self.config.MODEL_NAME}) on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto"
            )
            print("✅ 모델 로드 완료!")
        except Exception as e:
            raise RuntimeError(f"❌ 모델 로드 실패: {e}")
    def translate(self, input_text: str) -> str:
        messages = [
            {"role": "system", "content": self.config.SYSTEM_PROMPT},
            {"role": "user", "content": f"{self.config.FEW_SHOT_EXAMPLES}\n\n### 문제\n원문: {input_text}\n반드시 위 예시와 같은 형식으로 답변하십시오.\n[음독]:"}
        ]
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            output = self.model.generate(
                input_ids.to(self.device),
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=self.config.REPETITION_PENALTY
            )
            input_length = input_ids.shape[1]
            generated_tokens = output[0][input_length:]
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            if not result.startswith("[음독]"):
                result = "[음독]: " + result
            return result
        except Exception as e:
            return f"Error: {str(e)}"

def save_results(data: list, path: str):
    df = pd.DataFrame(data)
    if not os.path.exists(path):
        df.to_csv(path, index=False, encoding='utf-8-sig', mode='w')
    else:
        df.to_csv(path, index=False, encoding='utf-8-sig', mode='a', header=False)
    print(f"💾 {len(data)}건 저장 완료")

def main():
    config = Config()
    utils = TextUtils()
    if not os.path.exists(config.INPUT_CSV_PATH):
        print(f"❌ 파일을 찾을 수 없습니다: {config.INPUT_CSV_PATH}")
        return
    df = pd.read_csv(config.INPUT_CSV_PATH)
    print(f"📂 원본 데이터 로드: {len(df)}건")
    translator = HanjaTranslator(config)
    actual_sample_size = min(config.TARGET_COUNT, len(df))
    target_df = df.sample(n=actual_sample_size, random_state=42).copy()
    print(f"🚀 {actual_sample_size}개 데이터 번역 시작...")
    print(f"💾 저장 경로: {config.OUTPUT_CSV_PATH}\n")
    results_buffer = []
    for i, (idx, row) in enumerate(target_df.iterrows(), 1):
        src = row.get('pun_transcription', '')
        ref = row.get('translation', '')
        if pd.isna(src) or str(src).strip() == "":
            continue
        full_output = translator.translate(src)
        if "Error" in full_output:
            print(f"⚠️ [Skip] {idx}번 에러: {full_output}")
            continue
        hyp_clean = utils.extract_translation_part(full_output)
        score, ref_nouns, hyp_nouns = utils.calculate_bleu(ref, hyp_clean, utils.get_nouns_only)
        results_buffer.append({
            'original_index': idx,
            'src_hanja': src,
            'ref_korean': ref,
            'hyp_full': full_output,
            'hyp_clean': hyp_clean,
            'ref_nouns': ref_nouns,
            'hyp_nouns': hyp_nouns,
            'bleu_score': score
        })
        print(f"[{i}/{actual_sample_size}] BLEU: {score:.2f} | 정답: {ref_nouns[:10]}... | 예측: {hyp_nouns[:10]}...")
        if len(results_buffer) >= config.BATCH_SIZE:
            save_results(results_buffer, config.OUTPUT_CSV_PATH)
            results_buffer = []
    if results_buffer:
        save_results(results_buffer, config.OUTPUT_CSV_PATH)
    print("\n✅ 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
''',
            "analyzer.py": r'''# -*- coding: utf-8 -*-
"""
Translation Result Analyzer
Author: [Your Name]
Description: 번역 결과 CSV 파일을 읽어 BLEU 점수(Corpus & Sentence)를 분석합니다.
"""
import os
import pandas as pd
import sacrebleu
RESULT_CSV_PATH = "../data/translation_results_exaone_filtered.csv"

def analyze_translation_results(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return
    df = pd.read_csv(file_path)
    valid_df = df.dropna(subset=['ref_nouns', 'hyp_nouns'])
    print("=" * 50)
    print(f"📊 번역 품질 분석 보고서")
    print("=" * 50)
    print(f"• 전체 데이터: {len(df)}건")
    print(f"• 유효 데이터: {len(valid_df)}건 (누락 데이터 제외)")
    print("-" * 50)
    if len(valid_df) == 0:
        print("⚠️ 유효한 평가 데이터가 없습니다.")
        return
    avg_sentence_bleu = valid_df['bleu_score'].mean()
    refs = valid_df['ref_nouns'].tolist()
    preds = valid_df['hyp_nouns'].tolist()
    corpus_bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize='char')
    print(f"✅ 1. Sentence BLEU 평균 : {avg_sentence_bleu:.2f}")
    print(f"🏆 2. Corpus BLEU Score   : {corpus_bleu.score:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    analyze_translation_results(RESULT_CSV_PATH)
'''
        }
    },
    {
        "path": "nlp/gemini/Qwen_성능_평가",
        "title": "Qwen 성능 평가 코드",
        "purpose": "Qwen 모델을 사용하여 한문 번역 결과를 생성하고 평가",
        "summary": "번역문만 출력하도록 프롬프트를 설계하여 번역을 수행하고 BLEU 평가로 성능을 측정",
        "files": {
            "translator.py": r'''# -*- coding: utf-8 -*-
"""
Hanja Translation Script using Qwen Model
Author: [Your Name]
Description: Qwen 모델을 활용하여 한문 금석문을 격식 있는 국문으로 번역하는 스크립트입니다.
"""
import os
import re
import pandas as pd
import torch
from kiwipiepy import Kiwi
import sacrebleu
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class Config:
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    INPUT_CSV_PATH = "../data/pun_ksm_gsko_filtered.csv"
    OUTPUT_CSV_PATH = "../data/qwen_translation_results.csv"
    TARGET_COUNT = 750
    BATCH_SIZE = 50
    MAX_NEW_TOKENS = 600
    REPETITION_PENALTY = 1.1
    SYSTEM_PROMPT = """<role>
당신은 고전 한문 번역 전문가입니다.
</role>

<task>
주어진 한문을 문맥을 고려하여 격식 있는 한국어 문어체(예: ~하니라, ~하였더라, ~이니라)로 번역하십시오.
</task>

<constraints>
1. 부가적인 설명, 주석, 발음(음독), 고유명사 추출 목록 등을 **절대 포함하지 마십시오.**
2. 오직 **최종 번역문만** 출력하십시오.
3. 원문에 있는 결락 기호(▨)는 생략하거나 추측하지 말고, 개수를 유지하여 번역문에 그대로 포함하십시오.
4. 문체는 옛스러운 어조(~하니라, ~하더라)를 유지하십시오.
</constraints>"""
    FEW_SHOT_EXAMPLES = """<examples>
<example>
<input>府尹嚴相公善政碑 公諱鼎耉 字重叔 歲戊戌夏四月下車 己亥冬十月以病辭歸</input>
<output>
부윤 엄 상공의 선정비라. 공의 휘는 정구요 자는 중숙이니, 무술년 여름 4월에 부임하여 기해년 겨울 10월에 병으로 사직하고 돌아갔느니라.
</output>
</example>
</examples>"""

class TextUtils:
    def __init__(self):
        print("⚙️ Kiwi 형태소 분석기 초기화 중...")
        self.kiwi = Kiwi()
    def get_nouns_only(self, text: str) -> str:
        if not isinstance(text, str): 
            return ""
        text = re.sub(r'\([^)]*\)', ' ', text)
        text = re.sub(r'[^가-힣\s]', '', text)
        try:
            tokens = self.kiwi.tokenize(text)
            targets = {'NNG', 'NNP', 'NR', 'NP', 'NNB'}
            nouns = [t.form for t in tokens if t.tag in targets]
            return " ".join(nouns)
        except Exception:
            return ""
    def calculate_bleu(self, reference: str, hypothesis: str) -> tuple:
        ref_nouns = self.get_nouns_only(reference)
        hyp_nouns = self.get_nouns_only(hypothesis)
        if not ref_nouns or not hyp_nouns:
            return 0.0, ref_nouns, hyp_nouns
        bleu = sacrebleu.sentence_bleu(hyp_nouns, [ref_nouns], tokenize='char')
        return bleu.score, ref_nouns, hyp_nouns
    @staticmethod
    def extract_translation(full_text: str) -> str:
        clean_text = full_text.strip()
        markers = ["[최종 번역]:", "[최종 번역]", "[Output]:"]
        for marker in markers:
            if marker in clean_text:
                clean_text = clean_text.split(marker)[1].strip()
                break
        return clean_text

class QwenTranslator:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_model()
    def _load_model(self):
        print(f"⚙️ 모델 로딩 중... ({self.config.MODEL_NAME}) on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ 모델 로드 완료!")
        except Exception as e:
            raise RuntimeError(f"❌ 모델 로드 실패: {e}")
    def translate(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self.config.SYSTEM_PROMPT},
            {"role": "user", "content": f"{self.config.FEW_SHOT_EXAMPLES}\n\n### 문제\n원문: {text}\n위 예시와 같이 오직 번역문만 출력하십시오."}
        ]
        try:
            text_input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=self.config.REPETITION_PENALTY,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            input_len = model_inputs.input_ids.shape[1]
            generated_ids = generated_ids[0][input_len:]
            result = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return result
        except Exception as e:
            return f"Error: {str(e)}"

def save_results(data: list, path: str):
    df = pd.DataFrame(data)
    if not os.path.exists(path):
        df.to_csv(path, index=False, encoding='utf-8-sig', mode='w')
    else:
        df.to_csv(path, index=False, encoding='utf-8-sig', mode='a', header=False)


def main():
    config = Config()
    utils = TextUtils()
    if not os.path.exists(config.INPUT_CSV_PATH):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {config.INPUT_CSV_PATH}")
        return
    df = pd.read_csv(config.INPUT_CSV_PATH)
    print(f"📂 원본 데이터 로드: {len(df)}건")
    translator = QwenTranslator(config)
    actual_sample_size = min(config.TARGET_COUNT, len(df))
    target_df = df.sample(n=actual_sample_size, random_state=42).copy()
    print(f"\n🚀 번역 시작! (총 {actual_sample_size}건)")
    print(f"💾 저장 경로: {config.OUTPUT_CSV_PATH}\n")
    results_buffer = []
    for i, (idx, row) in enumerate(tqdm(target_df.iterrows(), total=actual_sample_size, desc="Translating"), 1):
        src = row.get('pun_transcription', '')
        ref = row.get('translation', '')
        if pd.isna(src) or str(src).strip() == "":
            continue
        full_output = translator.translate(src)
        if "Error" in full_output:
            continue
        hyp_clean = utils.extract_translation(full_output)
        score, ref_nouns, hyp_nouns = utils.calculate_bleu(ref, hyp_clean)
        results_buffer.append({
            'original_index': idx,
            'src_hanja': src,
            'ref_korean': ref,
            'hyp_full': full_output,
            'hyp_clean': hyp_clean,
            'ref_nouns': ref_nouns,
            'hyp_nouns': hyp_nouns,
            'bleu_score': score
        })
        if len(results_buffer) >= config.BATCH_SIZE:
            save_results(results_buffer, config.OUTPUT_CSV_PATH)
            results_buffer = []
    if results_buffer:
        save_results(results_buffer, config.OUTPUT_CSV_PATH)

if __name__ == "__main__":
    main()
''',
            "analyzer.py": r'''# -*- coding: utf-8 -*-
"""
Translation Result Analyzer (Qwen)
Author: [Your Name]
Description: Qwen 모델의 번역 결과 CSV를 읽어 BLEU 점수를 분석합니다.
"""
import os
import pandas as pd
import sacrebleu
RESULT_CSV_PATH = "../data/qwen_translation_results.csv"

def analyze_results(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ 결과 파일을 찾을 수 없습니다: {file_path}")
        return
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ CSV 파일 로드 중 오류 발생: {e}")
        return
    valid_df = df.dropna(subset=['ref_nouns', 'hyp_nouns'])
    print("=" * 60)
    print(f"📊 Qwen 번역 품질 분석 보고서")
    print("=" * 60)
    print(f"• 전체 데이터 수 : {len(df)}건")
    print(f"• 유효 평가 데이터: {len(valid_df)}건 (누락 제외)")
    print("-" * 60)
    if len(valid_df) == 0:
        print("⚠️ 평가할 유효 데이터가 없습니다.")
        return
    avg_sentence_bleu = valid_df['bleu_score'].mean()
    refs = valid_df['ref_nouns'].tolist()
    preds = valid_df['hyp_nouns'].tolist()
    corpus_bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize='char')
    print(f"✅ 1. Sentence BLEU 평균 : {avg_sentence_bleu:.2f}")
    print("-" * 40)
    print(f"🏆 2. Corpus BLEU Score   : {corpus_bleu.score:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    analyze_results(RESULT_CSV_PATH)
'''
        }
    },
    {
        "path": "vision/preprocessing/OpenCV_1-6",
        "title": "OpenCV 이용한 이미지 전처리 1-6",
        "purpose": "OpenCV를 활용하여 탁본 이미지의 대비 향상과 이진화 등 일련의 전처리 단계 수행",
        "summary": "노이즈 억제, 조명 보정, 글자 극성 판별, 대비 강화, 언샤프, 이진화, 모폴로지, 배경 반전 등을 포함한 파이프라인 구현",
        "files": {
            "preprocess_takbon_safe.py": r'''import cv2
import numpy as np
import json
from pathlib import Path

def save_img(p, img):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)

def gentle_unsharp(gray, radius=3, amount=0.45):
    blur = cv2.GaussianBlur(gray, ((radius|1), (radius|1)), 0)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return sharp

def linear_stretch(gray, lo=1.0, hi=99.0):
    p1, p2 = np.percentile(gray, [lo, hi])
    if p2 <= p1 + 1e-6:
        return gray
    out = np.clip((gray - p1) * (255.0 / (p2 - p1)), 0, 255).astype(np.uint8)
    return out

def estimate_text_polarity(gray):
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    band = cv2.dilate(edges, k, iterations=1)
    edge_vals = gray[band > 0]
    bg_mask = cv2.erode((band == 0).astype(np.uint8), k, iterations=2)
    bg_vals = gray[bg_mask > 0]
    if len(edge_vals) < 100 or len(bg_vals) < 100:
        return (gray.mean() < 120)
    return (edge_vals.mean() > bg_vals.mean())

def preprocess_takbon_safe(image_path, out_dir="./out_safe2"):
    name = Path(image_path).stem
    outdir = Path(out_dir)/name
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"file": str(image_path)}
    src = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError(image_path)
    save_img(outdir/"00_src.png", src)
    den = cv2.medianBlur(src, 3)
    save_img(outdir/"01_denoise.png", den)
    k = 71 if min(src.shape) > 1200 else 41
    if k % 2 == 0: k += 1
    bg = cv2.medianBlur(den, k)
    norm = cv2.normalize((den.astype(np.float32) / (bg.astype(np.float32) + 1e-6)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_img(outdir/"02_illum_norm.png", norm)
    need_invert = estimate_text_polarity(norm)
    gray = cv2.bitwise_not(norm) if need_invert else norm
    meta["invert_applied"] = bool(need_invert)
    save_img(outdir/"03_gray_after_polarity.png", gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    c1 = clahe.apply(gray)
    c2 = linear_stretch(c1, 2.0, 98.0)
    save_img(outdir/"04_contrast.png", c2)
    sh = gentle_unsharp(c2, radius=3, amount=0.35)
    save_img(outdir/"05_sharp.png", sh)
    H, W = sh.shape
    win = int(max(25, (min(H, W)//48) | 1))
    bin_adp = cv2.adaptiveThreshold(sh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, win, 8)
    _, bin_otsu = cv2.threshold(sh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    def balance_score(b):
        p = float(np.mean(b == 255))
        return -abs(p - 0.5)
    bin_final = bin_adp if balance_score(bin_adp) >= balance_score(bin_otsu) else bin_otsu
    save_img(outdir/"06_bin_raw.png", bin_final)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    bin_clean = cv2.morphologyEx(bin_final, cv2.MORPH_OPEN, open_k, iterations=2)
    save_img(outdir/"07_bin_clean.png", bin_clean)
    white_ratio = float(np.mean(bin_clean == 255))
    if white_ratio < 0.5:
        bin_clean = cv2.bitwise_not(bin_clean)
    save_img(outdir/f"{name}_ocrprep.png", bin_clean)
    save_img(outdir/f"{name}_master.png", sh)
    meta.update({
        "illum_kernel": int(k),
        "adaptive_win": int(win),
        "white_ratio": white_ratio
    })
    with open(outdir/"params.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ 완료: {image_path} → {outdir}")
    return outdir

if __name__ == "__main__":
    image_name = "test3.jpg"
    preprocess_takbon_safe(image_name)
'''
        }
    },
    {
        "path": "vision/preprocessing/briefnet",
        "title": "briefnet을 이용한 이미지 전처리",
        "purpose": "BiRefNet 등 세그멘테이션 모델로 전경을 분리하여 배경을 제거",
        "summary": "딥러닝 세그멘테이션을 이용해 한자 영역만 분리하고 마스크를 이용한 배경제거를 구현",
        "files": {
            "birefnet_segmentation.py": r'''from transformers import AutoModelForImageSegmentation, AutoImageProcessor
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

def run_birefnet_segmentation(img_path: str, model_id: str = "Zhengpeng7/BiRefNet"):
    processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageSegmentation.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    img_path_obj = Path(img_path)
    if img_path_obj.is_file():
        img = Image.open(img_path_obj).convert("RGB")
    else:
        # fallback: download a demo image
        demo_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-image-classification/resolve/main/imagenet_classification/000000039769.png"
        img = Image.open(requests.get(demo_url, stream=True).raw).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, (list, tuple)):
        logits = outputs[0]
    else:
        logits = outputs
    if logits.shape[1] == 1:
        pred_mask = torch.sigmoid(logits[0, 0])
    else:
        probs = torch.softmax(logits[0], dim=0)
        fg_class = 1
        pred_mask = probs[fg_class]
    pred_mask_np = pred_mask.detach().cpu().numpy()
    mask_bin = (pred_mask_np > 0.5).astype(np.uint8)
    img_np = np.array(img)
    mask_3ch = np.repeat(mask_bin[..., None], 3, axis=2)
    white_bg = np.ones_like(img_np, dtype=np.uint8) * 255
    seg_result = np.where(mask_3ch == 1, img_np, white_bg)
    return img_np, pred_mask_np, seg_result
'''
        }
    },
    {
        "path": "vision/preprocessing/DBNet",
        "title": "DBNet을 이용한 이미지 전처리",
        "purpose": "PaddleOCR의 DBNet 모델을 사용하여 글자 영역을 검출하고 배경을 제거",
        "summary": "CLAHE 대비 강화 후 DBNet으로 텍스트 영역을 검출하고 마스크를 만들어 흰 배경으로 합성",
        "files": {
            "dbnet_preprocess.py": r'''from paddleocr import PaddleOCR
import cv2
import numpy as np
from pathlib import Path

ocr = PaddleOCR(use_angle_cls=False, use_gpu=False, det=True, rec=False, lang='ch')

def text_cutout_whitebg(src_path, dst_path=None, contrast_boost=True):
    src_path = Path(src_path)
    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(src_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if contrast_boost:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    tmp_path = src_path.with_name(src_path.stem + "_tmp_for_ocr.png")
    cv2.imwrite(str(tmp_path), gray)
    res = ocr.ocr(str(tmp_path), det=True, rec=False)
    if not res or res[0] is None or len(res[0]) == 0:
        print("⚠️ 글자 박스가 검출되지 않았습니다. 대비나 밝기 조정이 필요할 수 있습니다.")
        return None
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for line in res[0]:
        if line is None:
            continue
        points = np.array(line[0]).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    mask_3 = cv2.merge([mask]*3)
    result = np.where(mask_3 == 255, img, white_bg)
    if dst_path is None:
        dst_path = src_path.with_name(src_path.stem + "_text_cutout.png")
    cv2.imwrite(str(dst_path), result)
    print(f"✅ 저장 완료: {dst_path}")
    tmp_path.unlink(missing_ok=True)
    return str(dst_path)
'''
        }
    },
    {
        "path": "vision/preprocessing/U2Net",
        "title": "U²-Net을 이용한 이미지 전처리",
        "purpose": "U²-Net 모델로 배경 제거와 pepper noise 제거를 수행",
        "summary": "rembg를 이용해 RGBA 컷아웃을 얻고 OpenCV로 노이즈를 제거한 후 흰 배경으로 합성",
        "files": {
            "u2net_preprocess.py": r'''from rembg import remove, new_session
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# 세션 초기화 (정확도 모드)
session = new_session("u2net")

def takbon_cutout_and_clean(src_path, dst_path=None, noise_kernel=3, min_area=20):
    src = Path(src_path)
    if dst_path is None:
        dst_path = src.with_name(src.stem + "_white_bg.png")
    im = Image.open(src).convert("RGBA")
    removed = remove(im, session=session)
    rgba = np.array(removed)
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((noise_kernel, noise_kernel), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean)
    mask_final = np.zeros_like(mask_clean)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            mask_final[labels == i] = 255
    white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
    mask_3ch = cv2.merge([mask_final]*3)
    result = np.where(mask_3ch == 255, rgb, white_bg)
    Image.fromarray(result).save(dst_path)
    print(f"✅ 결과 저장 완료: {dst_path}")
    return str(dst_path)
'''
        }
    },
    {
        "path": "ocr/craft/CRAFT",
        "title": "CRAFT를 이용한 한자 인식",
        "purpose": "CRAFT 탐지기로 한자 영역을 검출하고 뭉텅이를 watershed로 분할",
        "summary": "CRAFT 버전 호환을 감안해 detect_text를 호출하고, 큰 영역은 거리변환+watershed로 나눠 작은 글자를 추출",
        "files": {
            "craft_ocr.py": r'''# -*- coding: utf-8 -*-
"""
CRAFT 한자(텍스트) 탐지 + 뭉텅이 분할(watershed) 스크립트
- craft-text-detector 버전 차이 자동 대응
- 큰 덩어리(붙은 글자들)를 거리변환+watershed로 글자 단위로 분할
- 결과: 오버레이 PNG, 박스 JSON, (옵션) 크롭 저장
"""
import json
import inspect
from pathlib import Path
import cv2
import numpy as np
from craft_text_detector import Craft

IMG_PATH = r"C:\Users\myjew\takbon\test5.png"
OUT_DIR  = r"C:\hanja_craft_out"
SAVE_CROPS = True
PARAMS = dict(
    text_threshold=0.72,
    low_text=0.40,
    link_threshold=0.20,
    long_size=1920,
    cuda=False,
    refiner=False,
)
MIN_BOX_W, MIN_BOX_H = 8, 8
SPLIT_ENABLE = True
AREA_FACTOR = 3.0
SPLIT_MIN_CHAR_PIXELS = 8
SPLIT_MAX_CHARS = 32

def sort_poly_clockwise(poly):
    poly = np.array(poly, dtype=np.float32)
    c = np.mean(poly, axis=0)
    ang = np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0])
    idx = np.argsort(ang)
    return poly[idx].tolist()

def poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    return [x1, y1, int(x2 - x1), int(y2 - y1)]

def as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def run_craft_detect(img_path: str, params: dict):
    craft = Craft(cuda=params.get("cuda", False), refiner=params.get("refiner", False))
    sig = inspect.signature(craft.detect_text)
    accepts = set(sig.parameters.keys())
    if {"text_threshold", "low_text", "link_threshold", "long_size"}.issubset(accepts):
        prediction = craft.detect_text(
            img_path,
            text_threshold=params["text_threshold"],
            low_text=params["low_text"],
            link_threshold=params["link_threshold"],
            long_size=params["long_size"],
        )
        return craft, prediction
    craft = Craft(
        cuda=params.get("cuda", False),
        refiner=params.get("refiner", False),
        text_threshold=params.get("text_threshold", 0.7),
        low_text=params.get("low_text", 0.4),
        link_threshold=params.get("link_threshold", 0.4),
        long_size=params.get("long_size", 1280),
    )
    prediction = craft.detect_text(img_path)
    return craft, prediction

def split_blob_into_chars(crop_bgr, min_char=8, max_chars=16):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, k, iterations=1)
    dist = cv2.distanceTransform(binv, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, peaks = cv2.threshold((dist_norm * 255).astype(np.uint8), 120, 255, cv2.THRESH_BINARY)
    peaks = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, k, iterations=1)
    n_labels, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[binv == 0] = 0
    ws_in = cv2.cvtColor(binv, cv2.COLOR_GRAY2BGR)
    cv2.watershed(ws_in, markers)
    boxes = []
    for lab in range(2, n_labels + 1):
        mask = (markers == lab).astype(np.uint8) * 255
        if cv2.countNonZero(mask) < min_char:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        if w >= min_char and h >= min_char:
            boxes.append((x, y, w, h))
    if len(boxes) > max_chars:
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_chars]
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlay"; overlay_dir.mkdir(exist_ok=True)
    crops_dir = out_dir / "crops"; crops_dir.mkdir(exist_ok=True)
    img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    assert img_bgr is not None, f"이미지를 열 수 없습니다: {IMG_PATH}"
    H, W = img_bgr.shape[:2]
    craft = None
    try:
        craft, prediction = run_craft_detect(IMG_PATH, PARAMS)
        polys = as_list(prediction.get("polys"))
        boxes = as_list(prediction.get("boxes"))
        regions = polys if len(polys) > 0 else boxes
        raw = []
        areas = []
        for region in regions:
            poly = sort_poly_clockwise(region)
            x, y, bw, bh = poly_to_bbox(poly)
            if bw < MIN_BOX_W or bh < MIN_BOX_H:
                continue
            raw.append((poly, x, y, bw, bh))
            areas.append(bw * bh)
        median_area = float(np.median(areas)) if areas else 0.0
        results = []
        vis = img_bgr.copy()
        for poly, x, y, bw, bh in raw:
            area = bw * bh
            split_done = False
            if SPLIT_ENABLE and median_area > 0 and area >= AREA_FACTOR * median_area:
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x + bw), min(H, y + bh)
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    char_boxes = split_blob_into_chars(
                        crop,
                        min_char=SPLIT_MIN_CHAR_PIXELS,
                        max_chars=SPLIT_MAX_CHARS
                    )
                    if len(char_boxes) >= 2:
                        split_done = True
                        for (cx, cy, cw, ch) in char_boxes:
                            gx, gy, gw, gh = x + cx, y + cy, cw, ch
                            if gw < MIN_BOX_W or gh < MIN_BOX_H:
                                continue
                            results.append({
                                "index": len(results) + 1,
                                "poly": [[gx, gy], [gx+gw, gy], [gx+gw, gy+gh], [gx, gy+gh]],
                                "bbox": [int(gx), int(gy), int(gw), int(gh)],
                                "split": True
                            })
                            cv2.rectangle(vis, (gx, gy), (gx+gw, gy+gh), (0, 200, 255), 2)
                            cv2.putText(vis, f"{len(results)}", (gx, max(0, gy-5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                            if SAVE_CROPS:
                                gx1, gy1 = max(0, gx), max(0, gy)
                                gx2, gy2 = min(W, gx+gw), min(H, gy+gh)
                                gcrop = img_bgr[gy1:gy2, gx1:gx2]
                                if gcrop.size > 0:
                                    cv2.imwrite(str(crops_dir / f"char_{len(results):04d}.png"), gcrop)
            if not split_done:
                results.append({
                    "index": len(results) + 1,
                    "poly": [[int(px), int(py)] for px, py in poly],
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "split": False
                })
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 0, 0), 1)
                cv2.putText(vis, f"{len(results)}", (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                if SAVE_CROPS:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(W, x + bw), min(H, y + bh)
                    crop = img_bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(str(crops_dir / f"blob_{len(results):04d}.png"), crop)
        base = Path(IMG_PATH).stem
        cv2.imwrite(str((Path(OUT_DIR) / "overlay" / f"{base}_overlay.png")), vis)
        with open(Path(OUT_DIR) / f"{base}_boxes.json", "w", encoding="utf-8") as f:
            json.dump({
                "image": str(Path(IMG_PATH)),
                "size": {"w": W, "h": H},
                "count": len(results),
                "median_area": median_area,
                "area_factor": AREA_FACTOR,
                "split_enable": SPLIT_ENABLE,
                "results": results
            }, f, ensure_ascii=False, indent=2)
        print(f"[완료] 총 {len(results)}개 박스 (분할 포함)")
        print(f" - 오버레이: {Path(OUT_DIR) / 'overlay' / (base + '_overlay.png')}")
        print(f" - 박스 JSON: {Path(OUT_DIR) / (base + '_boxes.json')}")
        if SAVE_CROPS:
            print(f" - 크롭 폴더: {Path(OUT_DIR) / 'crops'}")
    finally:
        if craft is not None:
            try: craft.unload_craftnet_model()
            except Exception: pass
            try: craft.unload_refinenet_model()
            except Exception: pass

if __name__ == "__main__":
    main()
'''
        }
    },
    {
        "path": "nlp/sikuroberta/Colab_백업",
        "title": "[참고용] Colab용 코드 백업",
        "purpose": "Colab에서 작성된 코드의 백업본",
        "summary": "택본 데이터 전처리 및 학습에 사용된 여러 Colab 노트북 코드의 모음입니다.",
        "files": {
            "README.md": "# 참고용 Colab 코드\n\n이 폴더에는 Colab에서 실행했던 전처리 및 학습 스크립트의 백업이 포함되어 있습니다. 각 스크립트는 Notion 페이지를 참고하여 최신 구조로 리팩토링 되기 전에 사용된 실험 코드입니다."
        }
    },
    {
        "path": "nlp/translation/ExaOne_평가",
        "title": "ExaOne 성능 평가 코드",
        "purpose": "Gemini 번역 모델 중 ExaOne을 사용한 번역 성능 평가",
        "summary": "위의 nlp/gemini/ExaOne 폴더와 중복되어 있지만 예제 유지용으로 생성.",
        "files": {
            "README.md": "# ExaOne 번역 평가 코드\n\nGemini 번역 실험을 위해 ExaOne 모델을 이용해 번역 및 BLEU 평가를 수행하는 예제입니다."
        }
    },
    {
        "path": "image/briefnet/briefnet_을_이용한_이미지_전처리",
        "title": "briefnet을 이용한 이미지 전처리",
        "purpose": "BiRefNet 세그멘테이션 모델을 활용하여 탁본 이미지의 전경을 분리",
        "summary": "BiRefNet 모델로 segmentation mask를 생성하고, thresholding으로 글자 영역을 추출하여 백그라운드를 흰색으로 처리합니다.",
        "files": {
            "README.md": "# briefnet 전처리\n\n이 폴더에는 BiRefNet 세그멘테이션 모델을 이용해 탁본 이미지의 전경(한자)과 배경을 분리하는 예제가 포함됩니다. 상세 코드는 birefnet_segmentation.py를 참고하세요."
        }
    },
    {
        "path": "ocr/paddle/Paddle_OCR",
        "title": "Paddle OCR",
        "purpose": "PaddleOCR를 이용한 한자 OCR 실험",
        "summary": "PaddleOCR의 det+rec 모드를 사용하여 텍스트 박스를 검출하고 한자 인식을 수행하는 여러 시행착오를 진행합니다.",
        "files": {
            "methodB.py": "# Paddle OCR Method B\n\n이 스크립트는 Notion 페이지에서 상세히 설명된 PaddleOCR(det+rec) 파이프라인의 요약본입니다. 전체 코드는 노션을 참고하세요."
        }
    },
    {
        "path": "ocr/paddle_ensemble/Paddle_OCR_과_고문서OCR_앙상블",
        "title": "Paddle OCR과 고문서OCR 앙상블",
        "purpose": "PaddleOCR 결과와 고문서 OCR(HRCN) 결과를 앙상블하여 한자 인식률 향상을 꾀함",
        "summary": "세로쓰기 정렬과 앙상블 알고리즘을 적용하여 두 OCR 엔진의 예측 결과를 결합합니다.",
        "files": {
            "fusion.py": "# Fusion Placeholder\n\n이 파일은 PaddleOCR와 HRCN OCR의 앙상블 예제에 대한 자리표시자입니다. 실제 코드 구현은 Notion 페이지를 참고하세요."
        }
    },
    {
        "path": "ocr/aihub/AI_Hub_고문서_OCR_단독_실행_코드",
        "title": "AI Hub 고문서 OCR 단독 실행 코드",
        "purpose": "AI Hub 고문서 OCR 모델을 단독으로 실행하는 스크립트",
        "summary": "ResNet 기반 탐지 모델과 인식기를 로드하여 한자 OCR 추론을 수행합니다.",
        "files": {
            "aihub_ocr.py": "# AI Hub OCR Placeholder\n\nAI Hub 고문서 OCR 모델을 실행하는 실제 코드는 Notion 페이지에 자세히 나와 있습니다. 여기는 자리표시자입니다."
        }
    },
    {
        "path": "ocr/easyocr/Easy_OCR_한자_한_글자_인식",
        "title": "Easy OCR 한자 한 글자 인식",
        "purpose": "EasyOCR를 활용하여 한자 한 글자 단위로 인식하는 실험",
        "summary": "EasyOCR 엔진과 다양한 전처리 조합을 사용하여 한자 1글자 데이터셋 생성 가능성을 탐색.",
        "files": {
            "easyocr_experiment.py": "# EasyOCR Placeholder\n\nEasyOCR를 이용한 한자 한 글자 인식 실험 코드는 Notion 페이지에 있습니다."
        }
    },
    {
        "path": "ocr/hrnet/Faster_RCNN_HRNet",
        "title": "Faster-RCNN + HRNet",
        "purpose": "HRNet 백본의 Faster R-CNN으로 한자 객체 검출 실험",
        "summary": "MMDetection 프레임워크를 사용하여 한자 객체 검출 모델을 학습하고 평가하는 실험입니다.",
        "files": {
            "faster_rcnn_hrnet.py": "# Faster R-CNN + HRNet Placeholder\n\n해당 실험을 위한 MMDetection 설정 및 훈련 스크립트는 Notion 페이지와 MMDetection 설정 파일을 참고하세요."
        }
    },
    {
        "path": "ocr/hrnet/Faster_RCNN_HRNet_Crop_augmentation",
        "title": "Faster R-CNN+HRNet Crop augmentation",
        "purpose": "크롭 증강을 추가한 Faster R-CNN+HRNet 훈련",
        "summary": "랜덤 크롭을 활용해 작은 글자 검출 성능을 향상시키고 과적합을 완화하려는 실험입니다.",
        "files": {
            "crop_aug_script.py": "# Crop Augmentation Placeholder\n\nFaster R-CNN+HRNet 모델에 크롭 증강을 적용한 코드의 자리표시자입니다."
        }
    },
    {
        "path": "ocr/fcos/FCOS",
        "title": "FCOS",
        "purpose": "FCOS one-stage 검출기를 한자 검출에 적용하는 실험",
        "summary": "Detectron2 기반 FCOS 모델로 한자 위치를 탐지하는 초기 실험을 진행합니다.",
        "files": {
            "fcos_experiment.py": "# FCOS Placeholder\n\nFCOS 모델을 사용한 실험 코드의 자세한 내용은 Notion 페이지를 참고하세요."
        }
    },
    {
        "path": "ocr/yolo/YOLO",
        "title": "YOLO",
        "purpose": "YOLO 모델을 활용한 한자 영역 탐지 실험",
        "summary": "Ultralytics YOLO를 이용해 빠른 실험을 진행하고 한자 검출 가능성을 확인합니다.",
        "files": {
            "yolo_experiment.py": "# YOLO Placeholder\n\nYOLO 기반 한자 검출 실험을 위한 코드 자리표시자입니다."
        }
    },
    {
        "path": "ocr/kakren/Kakren_CHAT_OCR",
        "title": "Kakren(CHAT OCR)",
        "purpose": "Kraken OCR과 CHAT 모델을 활용해 한자 OCR 성능을 평가",
        "summary": "Kraken 및 CHAT OCR 모델을 설치하고 CLI와 Python API로 한자 고문서 OCR을 수행한 실험입니다.",
        "files": {
            "kakren_ocr.py": "# Kakren OCR Placeholder\n\nKraken/CHAT OCR 실험 코드는 Notion 페이지를 참고하십시오."
        }
    },
    {
        "path": "ocr/deepseek/DeepSeek_OCR",
        "title": "DeepSeek OCR",
        "purpose": "DeepSeek OCR 엔진을 활용하여 한자 OCR 성능을 테스트",
        "summary": "DeepSeek OCR의 출력 형태를 분석하고 후처리 기법으로 성능 개선을 시도하는 실험입니다.",
        "files": {
            "deepseek_ocr.py": "# DeepSeek OCR Placeholder\n\nDeepSeek OCR 엔진 테스트 코드 자리표시자입니다."
        }
    },
    {
        "path": "ocr/google/Google_OCR",
        "title": "Google OCR",
        "purpose": "Google Cloud Vision OCR을 활용한 한자 인식 및 후처리 실험",
        "summary": "Google OCR 결과를 기반으로 다양한 룰 기반 후처리와 고문서 OCR 앙상블을 테스트한 코드 실험입니다.",
        "files": {
            "google_ocr.py": "# Google OCR Placeholder\n\nGoogle Cloud Vision OCR에 대한 후처리 및 앙상블 코드의 자리표시자입니다."
        }
    }
]

def remove_emojis(text: str) -> str:
    """
    문자열에서 이모지(emojis)를 제거합니다.

    Emoji 범위는 여러 Unicode 블록에 걸쳐 있으므로 정규식을 사용합니다.
    대부분의 이모지는 U+1F300~U+1F6FF, U+1F900~U+1F9FF 등에 포함됩니다.

    Args:
        text (str): 원본 문자열

    Returns:
        str: 이모지가 제거된 문자열
    """
    # 이모지 패턴 정의
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"  # 기호 및 아이콘
        "\U0001F600-\U0001F64F"  # 이모티콘 표정
        "\U0001F680-\U0001F6FF"  # 교통 및 지도 기호
        "\U0001F700-\U0001F77F"  # 기호 및 아이콘 확장
        "\U0001F780-\U0001F7FF"  # 추가 기호
        "\U0001F800-\U0001F8FF"  # 추가 화살표
        "\U0001F900-\U0001F9FF"  # 보충 기호 및 픽토그램
        "\U0001FA70-\U0001FAFF"  # 음식/기타 기호
        "\U00002702-\U000027B0"  # 특수 기호
        "\U000024C2-\U0001F251"  # 기호
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub('', text)


def main():
    """
    메인 함수는 실험 목록을 순회하며 각 디렉터리를 생성하고 README 및 코드 파일을 작성합니다.

    - 기존 파일이 있으면 삭제 후 재생성합니다.
    - README와 코드 파일에서 이모지를 제거합니다.
    - 각 코드 파일 상단에 한국어 주석을 추가하여 제목, 목적, 요약을 설명합니다.
    """
    created_files = []
    for exp in experiments:
        # 실험용 폴더 경로를 생성합니다.
        dir_path = os.path.join(BASE_PATH, *exp['path'].split('/'))
        os.makedirs(dir_path, exist_ok=True)
        # README 내용 작성 (이모지 제거)
        readme_content = (
            f"# {exp['title']}\n\n"
            f"## 목적\n"
            f"- {exp['purpose']}\n\n"
            f"## 시행착오 요약\n"
            f"- {exp['summary']}\n"
        )
        readme_content = remove_emojis(readme_content)
        readme_path = os.path.join(dir_path, "README.md")
        # 기존 README가 있으면 삭제
        if os.path.exists(readme_path):
            os.remove(readme_path)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        created_files.append(readme_path)
        # 코드 파일 생성
        for filename, content in exp['files'].items():
            file_path = os.path.join(dir_path, filename)
            # 기존 파일이 있으면 삭제
            if os.path.exists(file_path):
                os.remove(file_path)
            # 이모지 제거 및 주석 헤더 추가
            content_clean = remove_emojis(content)
            header_lines = [
                f"# {exp['title']}",
                f"# 목적: {exp['purpose']}",
                f"# 요약: {exp['summary']}",
                f"# 작성일: 2025-12-10",
                "",
            ]
            header = "\n".join(header_lines)
            content_with_header = header + content_clean
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content_with_header)
            created_files.append(file_path)
    # 생성된 모든 파일 경로 출력
    for path in created_files:
        print(path)

if __name__ == "__main__":
    main()

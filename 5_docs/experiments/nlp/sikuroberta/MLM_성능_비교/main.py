# MLM 성능 비교
# 목적: BERT 기반 한문 MLM 모델들의 성능을 비교 평가
# 요약: 모델별 Top-K 정확도 및 질적 비교를 통해 성능 차이를 분석
# 작성일: 2025-12-10
# -*- coding: utf-8 -*-
"""
Historical BERT MLM Evaluation Framework
========================================
Description:
          3 BERT  
    Masked Language Modeling (MLM)     .

    [  ]
    1. SillokBERT (HuggingFace):  
    2. SikuRoBERTa (HuggingFace):  
    3. HUE (Local):     (  )

Features:
    -   : Git Clone    
    -   : (Hanja)    
    -  : (Top-K Acc)  (Side-by-side) 

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
# 1. Logger Setup ( )
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. Configuration ( )
# -----------------------------------------------------------------------------
class Config:
    """     """
    # [Project Root]   (main.py)    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #  
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
        """   fill-mask  ."""
        is_local = os.path.isabs(model_path) or os.path.sep in model_path
        if is_local:
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                logger.warning(f"  [Skip] '{name}'    .")
                logger.warning(f"    : {model_path}")
                return None
        try:
            logger.info(f"⏳ [{name}]   ...")
            pipe = pipeline(
                "fill-mask",
                model=model_path,
                tokenizer=model_path,
                device=Config.DEVICE,
                top_k=top_k
            )
            logger.info(f" [{name}]  ")
            return pipe
        except Exception as e:
            logger.error(f" [{name}]  : {e}")
            return None

# -----------------------------------------------------------------------------
# 4. Data Processor
# -----------------------------------------------------------------------------
class DataProcessor:
    @staticmethod
    def load_and_clean(file_path: str) -> List[str]:
        """   XML    ."""
        if not os.path.exists(file_path):
            logger.error(f"     : {file_path}")
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"   : {e}")
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
        logger.info(f"   : {len(clean_lines)} ")
        return clean_lines

    @staticmethod
    def create_masked_dataset(lines: List[str], num_samples: int) -> List[Dict[str, str]]:
        """  (Hanja)   [MASK]  ."""
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
        logger.info(f"     :   {len(dataset)}")
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
        logger.info(f" [{name}]   ( {len(dataset)})...")
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
        print(f" [Qualitative Analysis]    ")
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
    print(" [Start] Historical BERT Comparison Framework")
    print(f" Project Root: {Config.BASE_DIR}")
    pipelines = {}
    p1 = ModelHandler.load_pipeline(Config.SILLOK_MODEL_PATH, "SillokBERT", Config.TOP_K)
    if p1: pipelines["SillokBERT"] = p1
    p2 = ModelHandler.load_pipeline(Config.SIKU_MODEL_PATH, "SikuRoBERTa", Config.TOP_K)
    if p2: pipelines["SikuRoBERTa"] = p2
    p3 = ModelHandler.load_pipeline(Config.HUE_MODEL_PATH, "HUE (Local)", Config.TOP_K)
    if p3: pipelines["HUE (Local)"] = p3
    if not pipelines:
        logger.error("    .  .")
        return
    raw_lines = DataProcessor.load_and_clean(Config.TEST_DATA_PATH)
    if not raw_lines:
        return
    test_dataset = DataProcessor.create_masked_dataset(raw_lines, Config.NUM_SAMPLES)
    if not test_dataset:
        logger.error("    .")
        return
    print("\n" + "#"*60)
    print("    (Quantitative Evaluation)")
    print("#"*60)
    for name, pipe in pipelines.items():
        res = Evaluator.evaluate_quantitative(name, pipe, test_dataset, Config.TOP_K)
        print(f"\n  Model: {name}")
        print(f"   - Top-1 Accuracy: {res['acc_top1']:.2f}%")
        print(f"   - Top-{Config.TOP_K} Accuracy: {res['acc_topk']:.2f}%")
        print(f"   - Valid Samples : {res['count']}")
    if test_dataset:
        Evaluator.compare_qualitative(test_dataset[0], pipelines, Config.TOP_K)
    print("\n   .")

if __name__ == "__main__":
    main()

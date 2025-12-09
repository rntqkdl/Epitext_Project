"""
SikuRoBERTa Evaluation Task
======================================================================
목적: 학습된 모델의 성능(Loss, Perplexity, Top-K Accuracy) 평가
작성자: Epitext Project Team
======================================================================
"""

import sys
import math
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
from torch.utils.data import DataLoader

try:
    from config import EvalConfig
except ImportError:
    from .config import EvalConfig


# ======================================================================
# 리소스 로드 함수
# ======================================================================
def load_model_and_data():
    """모델 및 테스트 데이터셋 로드"""
    print("[Info] Loading Model & Test Data...")
    
    # 모델 경로 확인
    if not EvalConfig.MODEL_PATH.exists():
        print(f"[Error] Model not found: {EvalConfig.MODEL_PATH}")
        print("Please train the model first.")
        sys.exit(1)
        
    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(str(EvalConfig.MODEL_PATH))
    model = BertForMaskedLM.from_pretrained(str(EvalConfig.MODEL_PATH))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 데이터 경로 확인
    if not EvalConfig.TEST_DATA_PATH.exists():
        print(f"[Error] Test data not found: {EvalConfig.TEST_DATA_PATH}")
        sys.exit(1)
        
    # 데이터셋 로드
    datasets = load_from_disk(str(EvalConfig.TEST_DATA_PATH))
    
    if "test" not in datasets:
        print("[Error] 'test' split not found in dataset.")
        sys.exit(1)
        
    test_data = datasets["test"]
    print(f" Test Samples: {len(test_data)}")
    
    return tokenizer, model, test_data, device


# ======================================================================
# 평가 함수
# ======================================================================
def evaluate(model, dataloader, device):
    """모델 평가 루프"""
    print("\n[Info] Starting Evaluation...")
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct1 = 0
    correct5 = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 데이터를 디바이스로 이동
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            labels = batch["labels"]
            
            # 마스킹된 토큰만 추출 (-100은 무시됨)
            mask = labels != -100
            if mask.sum() == 0:
                continue
                
            active_logits = logits[mask]
            active_labels = labels[mask]
            
            # Loss 누적
            total_loss += loss.item() * active_labels.size(0)
            total_tokens += active_labels.size(0)
            
            # Top-1 Accuracy
            preds = active_logits.argmax(dim=-1)
            correct1 += (preds == active_labels).sum().item()
            
            # Top-5 Accuracy
            _, top5_preds = active_logits.topk(5, dim=-1)
            # unsqueeze로 차원 맞추고 비교
            correct5 += (top5_preds == active_labels.unsqueeze(-1)).any(dim=-1).sum().item()
            
    return total_loss, total_tokens, correct1, correct5


# ======================================================================
# 메인 실행 함수
# ======================================================================
def main():
    print("======================================================")
    print(" SikuRoBERTa MLM Evaluation Start")
    print("======================================================")
    
    EvalConfig.print_config()
    
    # 1. 리소스 준비
    tokenizer, model, test_data, device = load_model_and_data()
    
    # 2. 데이터 로더 설정
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=EvalConfig.MLM_PROBABILITY
    )
    
    loader = DataLoader(
        test_data, 
        batch_size=EvalConfig.BATCH_SIZE, 
        collate_fn=collator
    )
    
    # 3. 평가 수행
    loss, tokens, c1, c5 = evaluate(model, loader, device)
    
    if tokens == 0:
        print("[Warning] No masked tokens found for evaluation.")
        return

    # 4. 결과 계산
    avg_loss = loss / tokens
    perplexity = math.exp(avg_loss)
    acc1 = c1 / tokens
    acc5 = c5 / tokens
    
    # 5. 결과 출력 및 저장
    result_msg = (
        f"\n======================================\n"
        f" Evaluation Results\n"
        f"======================================\n"
        f" Date:        {pd.Timestamp.now()}\n"
        f" Loss:        {avg_loss:.4f}\n"
        f" Perplexity:  {perplexity:.4f}\n"
        f" Top-1 Acc:   {acc1:.2%}\n"
        f" Top-5 Acc:   {acc5:.2%}\n"
        f"======================================\n"
    )
    
    print(result_msg)
    
    # 결과 파일 저장
    EvalConfig.RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EvalConfig.RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(result_msg)
        
    print(f"[Done] Results saved to: {EvalConfig.RESULT_FILE}")


if __name__ == "__main__":
    main()
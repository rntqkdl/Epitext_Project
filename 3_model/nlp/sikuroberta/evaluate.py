"""
SikuRoBERTa MLM 평가 코드
======================================================================
목적: 학습된 SikuRoBERTa의 Test 성능 평가 (Loss, Perplexity, Top-1/Top-5)
작성자: 4조 복원왕 김탁본
날짜: 2025-01-XX
======================================================================
"""

import os
import math
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk
from torch.utils.data import DataLoader


# ======================================================================
# 설정
# ======================================================================
class Config:
    """평가 설정"""
    BASE_PATH = r"C:\Users\김선민\Downloads\punmodel"
    CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints_simple_128")
    SPLIT_DATASET_DIR = os.path.join(BASE_PATH, "tokenized_sikuroberta_simple_128_split")
    
    EVAL_BATCH_SIZE = 8
    MLM_PROBABILITY = 0.15


# ======================================================================
# 유틸리티 함수
# ======================================================================
def load_checkpoint(config):
    """최신 체크포인트 로드"""
    print("\n--- 체크포인트 탐색 ---")
    
    last_checkpoint = get_last_checkpoint(config.CHECKPOINT_DIR)
    
    if last_checkpoint is None:
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {config.CHECKPOINT_DIR}"
        )
    
    print(f"✅ 최신 체크포인트: {last_checkpoint}")
    return last_checkpoint


def load_model_and_tokenizer(checkpoint_path):
    """모델 및 토크나이저 로드"""
    print("\n--- 모델 및 토크나이저 로드 ---")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
    model = BertForMaskedLM.from_pretrained(checkpoint_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Vocab Size: {len(tokenizer)}")
    print(f"Device: {device}")
    
    return tokenizer, model, device


def load_test_dataset(config):
    """테스트 데이터셋 로드"""
    print("\n--- 테스트 데이터셋 로드 ---")
    
    if not os.path.exists(config.SPLIT_DATASET_DIR):
        raise FileNotFoundError(
            f"분할된 데이터셋이 없습니다: {config.SPLIT_DATASET_DIR}"
        )
    
    datasets = load_from_disk(config.SPLIT_DATASET_DIR)
    test_dataset = datasets["test"]
    
    print(f"✅ Test 샘플 수: {len(test_dataset)}")
    return test_dataset


def create_test_dataloader(test_dataset, tokenizer, config):
    """테스트 DataLoader 생성"""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.MLM_PROBABILITY,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )
    
    print(f"✅ 배치 수: {len(test_loader)}")
    return test_loader


# ======================================================================
# 평가 함수
# ======================================================================
def evaluate(model, test_loader, device):
    """MLM 성능 평가 (Loss, Perplexity, Top-1, Top-5)"""
    print("\n--- 평가 시작 ---")
    
    total_loss = 0.0
    total_tokens = 0
    correct1 = 0
    correct5 = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 배치를 device로 이동
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits  # (B, L, V)
            labels = batch["labels"]  # (B, L)
            
            # 마스크된 위치만 추출
            vocab_size = logits.size(-1)
            logits = logits.view(-1, vocab_size)  # (B*L, V)
            labels = labels.view(-1)              # (B*L,)
            
            mask = labels != -100  # -100은 비마스크 토큰
            
            if mask.sum().item() == 0:
                continue
            
            logits = logits[mask]  # (N_masked, V)
            labels = labels[mask]  # (N_masked,)
            
            # Loss 누적
            total_loss += loss.item() * labels.size(0)
            total_tokens += labels.size(0)
            
            # Top-1 정확도
            top1 = logits.argmax(dim=-1)
            correct1 += (top1 == labels).sum().item()
            
            # Top-5 정확도
            k = min(5, logits.size(-1))
            top5_indices = logits.topk(k, dim=-1).indices
            match_top5 = (top5_indices == labels.unsqueeze(-1)).any(dim=-1)
            correct5 += match_top5.sum().item()
    
    return total_loss, total_tokens, correct1, correct5


def print_results(total_loss, total_tokens, correct1, correct5):
    """결과 출력"""
    if total_tokens == 0:
        print("❌ 평가할 마스크 토큰이 없습니다")
        return
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    top1_acc = correct1 / total_tokens
    top5_acc = correct5 / total_tokens
    
    print("\n" + "=" * 70)
    print("📊 Test 결과 (최신 체크포인트)")
    print("=" * 70)
    print(f"Test Loss (per masked token): {avg_loss:.4f}")
    print(f"Test Perplexity            : {ppl:.4f}")
    print(f"Top-1 Accuracy (masked)    : {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"Top-5 Accuracy (masked)    : {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print("=" * 70)


# ======================================================================
# 메인 평가 함수
# ======================================================================
def main():
    """메인 평가 실행"""
    print("=" * 70)
    print("SikuRoBERTa MLM 평가 시작")
    print("=" * 70)
    
    config = Config()
    
    # 1. 체크포인트 로드
    checkpoint_path = load_checkpoint(config)
    
    # 2. 모델 및 토크나이저 로드
    tokenizer, model, device = load_model_and_tokenizer(checkpoint_path)
    
    # 3. 테스트 데이터셋 로드
    test_dataset = load_test_dataset(config)
    
    # 4. DataLoader 생성
    test_loader = create_test_dataloader(test_dataset, tokenizer, config)
    
    # 5. 평가 실행
    total_loss, total_tokens, correct1, correct5 = evaluate(
        model, test_loader, device
    )
    
    # 6. 결과 출력
    print_results(total_loss, total_tokens, correct1, correct5)
    
    print("\n🎉 평가 완료!")


if __name__ == "__main__":
    main()

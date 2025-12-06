"""
SikuRoBERTa MLM 평가 코드
======================================================================
목적: 학습된 SikuRoBERTa의 Test 성능 평가 (Loss, Perplexity, Top-1/Top-5)
작성자: 4조 복원왕 김탁본
날짜: 2025-12-07
======================================================================
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import math
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, 
    BertForMaskedLM, 
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk
from torch.utils.data import DataLoader

# 설정 파일 import
try:
    from config import PathConfig, EvalConfig
except ImportError:
    from sikuroberta.config import PathConfig, EvalConfig


# ======================================================================
# 유틸리티 함수
# ======================================================================
def load_checkpoint():
    """최신 체크포인트 로드"""
    print("\n--- 체크포인트 탐색 ---")
    
    last_checkpoint = get_last_checkpoint(str(PathConfig.CHECKPOINT_DIR))
    
    if last_checkpoint is None:
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {PathConfig.CHECKPOINT_DIR}\n"
            f"먼저 학습을 실행하세요."
        )
    
    print(f"최신 체크포인트: {last_checkpoint}")
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


def load_test_dataset():
    """테스트 데이터셋 로드"""
    print("\n--- 테스트 데이터셋 로드 ---")
    
    if not PathConfig.SPLIT_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"분할된 데이터셋이 없습니다: {PathConfig.SPLIT_DATASET_PATH}\n"
            f"먼저 학습을 실행하세요 (학습 시 자동으로 분할됩니다)."
        )
    
    datasets = load_from_disk(str(PathConfig.SPLIT_DATASET_PATH))
    test_dataset = datasets["test"]
    
    print(f"Test 샘플 수: {len(test_dataset)}")
    return test_dataset


def create_test_dataloader(test_dataset, tokenizer):
    """테스트 DataLoader 생성"""
    print("\n--- DataLoader 생성 ---")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=EvalConfig.MLM_PROBABILITY,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=EvalConfig.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )
    
    print(f"배치 수: {len(test_loader)}")
    return test_loader


# ======================================================================
# 평가 함수
# ======================================================================
def evaluate(model, test_loader, device):
    """MLM 성능 평가 (Loss, Perplexity, Top-1, Top-5)"""
    print("\n" + "=" * 70)
    print("평가 시작")
    print("=" * 70 + "\n")
    
    total_loss = 0.0
    total_tokens = 0
    correct1 = 0
    correct5 = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", ncols=100):
            # 배치를 device로 이동
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits  # (B, L, V)
            labels = batch["labels"]  # (B, L)
            
            # 마스크된 위치만 추출
            vocab_size = logits.size(-1)
            logits_flat = logits.view(-1, vocab_size)  # (B*L, V)
            labels_flat = labels.view(-1)              # (B*L,)
            
            mask = labels_flat != -100  # -100은 비마스크 토큰
            
            if mask.sum().item() == 0:
                continue
            
            logits_masked = logits_flat[mask]  # (N_masked, V)
            labels_masked = labels_flat[mask]  # (N_masked,)
            
            # Loss 누적
            total_loss += loss.item() * labels_masked.size(0)
            total_tokens += labels_masked.size(0)
            
            # Top-1 정확도
            top1 = logits_masked.argmax(dim=-1)
            correct1 += (top1 == labels_masked).sum().item()
            
            # Top-5 정확도
            k = min(5, logits_masked.size(-1))
            top5_indices = logits_masked.topk(k, dim=-1).indices
            match_top5 = (top5_indices == labels_masked.unsqueeze(-1)).any(dim=-1)
            correct5 += match_top5.sum().item()
    
    return total_loss, total_tokens, correct1, correct5


def print_and_save_results(total_loss, total_tokens, correct1, correct5):
    """결과 출력 및 저장"""
    if total_tokens == 0:
        print("평가할 마스크 토큰이 없습니다")
        return
    
    # 지표 계산
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    top1_acc = correct1 / total_tokens
    top5_acc = correct5 / total_tokens
    
    # 콘솔 출력
    result_text = f"""
{"=" * 70}
Test 결과 (최신 체크포인트)
{"=" * 70}
Test Loss (per masked token): {avg_loss:.4f}
Test Perplexity            : {ppl:.4f}
Top-1 Accuracy (masked)    : {top1_acc:.4f} ({top1_acc*100:.2f}%)
Top-5 Accuracy (masked)    : {top5_acc:.4f} ({top5_acc*100:.2f}%)

총 마스크 토큰 수           : {total_tokens:,}
Top-1 정답 수              : {correct1:,}
Top-5 정답 수              : {correct5:,}
{"=" * 70}
"""
    
    print(result_text)
    
    # 파일로 저장
    with open(PathConfig.RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("SikuRoBERTa MLM Test Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"날짜: {pd.Timestamp.now()}\n\n")
        f.write(result_text)
    
    print(f"결과 저장: {PathConfig.RESULTS_PATH}\n")


# ======================================================================
# 메인 평가 함수
# ======================================================================
def main():
    """메인 평가 실행"""
    print("=" * 70)
    print("SikuRoBERTa MLM 평가 시작")
    print("=" * 70)
    
    # 경로 및 설정 출력
    PathConfig.print_paths()
    EvalConfig.print_config()
    
    # 1. 체크포인트 로드
    checkpoint_path = load_checkpoint()
    
    # 2. 모델 및 토크나이저 로드
    tokenizer, model, device = load_model_and_tokenizer(checkpoint_path)
    
    # 3. 테스트 데이터셋 로드
    test_dataset = load_test_dataset()
    
    # 4. DataLoader 생성
    test_loader = create_test_dataloader(test_dataset, tokenizer)
    
    # 5. 평가 실행
    total_loss, total_tokens, correct1, correct5 = evaluate(
        model, test_loader, device
    )
    
    # 6. 결과 출력 및 저장
    print_and_save_results(total_loss, total_tokens, correct1, correct5)
    
    print("=" * 70)
    print("평가 완료!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import pandas as pd  # 날짜 출력용
    main()

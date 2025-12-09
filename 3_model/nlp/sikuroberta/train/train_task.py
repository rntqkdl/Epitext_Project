"""
SikuRoBERTa Training Task
======================================================================
목적: MLM(Masked Language Modeling) 방식으로 모델 학습 수행
작성자: Epitext Project Team
======================================================================
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk, DatasetDict

try:
    from config import TrainConfig
except ImportError:
    from .config import TrainConfig


# ======================================================================
# 리소스 로드 함수
# ======================================================================
def load_tokenizer_and_model():
    """토크나이저 및 모델 초기화"""
    print("[Info] Loading Tokenizer & Model...")
    
    # 토크나이저 로드 (데이터셋 경로에서 vocab 로드)
    tokenizer = AutoTokenizer.from_pretrained(
        str(TrainConfig.DATASET_PATH), 
        use_fast=False
    )
    
    # 모델 로드
    model = BertForMaskedLM.from_pretrained(TrainConfig.MODEL_NAME)
    
    # 토크나이저 크기에 맞춰 임베딩 사이즈 조정
    model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model


def load_dataset():
    """데이터셋 로드 및 분할 (Train/Valid/Test)"""
    print("[Info] Loading Dataset...")
    
    if not TrainConfig.DATASET_PATH.exists():
        print(f"[Error] Dataset not found: {TrainConfig.DATASET_PATH}")
        sys.exit(1)
        
    dataset = load_from_disk(str(TrainConfig.DATASET_PATH))
    
    # 이미 분할된 데이터셋인 경우 바로 반환
    if isinstance(dataset, DatasetDict):
        print("[Info] Using existing split dataset.")
        return dataset
        
    print("[Info] Splitting dataset (80/10/10)...")
    # 분할 로직 (Train 80% / Valid 10% / Test 10%)
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=42)
    
    final_dataset = DatasetDict({
        "train": train_test["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"]
    })
    
    return final_dataset


# ======================================================================
# 시각화 함수
# ======================================================================
def save_loss_graph(trainer):
    """학습 손실 그래프 저장"""
    print("[Info] Generating Loss Graph...")
    
    history = pd.DataFrame(trainer.state.log_history)
    
    # 손실 데이터 필터링
    if "loss" not in history.columns and "eval_loss" not in history.columns:
        return
        
    plt.figure(figsize=(10, 6))
    
    if "loss" in history.columns:
        train_loss = history.dropna(subset=["loss"])
        plt.plot(train_loss["epoch"], train_loss["loss"], label="Train Loss")
        
    if "eval_loss" in history.columns:
        eval_loss = history.dropna(subset=["eval_loss"])
        plt.plot(eval_loss["epoch"], eval_loss["eval_loss"], label="Valid Loss")
        
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 그래프 저장
    TrainConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(TrainConfig.GRAPH_SAVE_PATH))
    plt.close()
    print(f"[Info] Graph saved: {TrainConfig.GRAPH_SAVE_PATH}")


# ======================================================================
# 메인 실행 함수
# ======================================================================
def main():
    print("======================================================")
    print(" SikuRoBERTa MLM Training Start")
    print("======================================================")
    
    TrainConfig.print_config()
    
    # 1. 리소스 준비
    tokenizer, model = load_tokenizer_and_model()
    datasets = load_dataset()
    
    # 2. 데이터 콜레이터 (MLM 설정)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=TrainConfig.MLM_PROBABILITY
    )
    
    # 3. 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=str(TrainConfig.OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=TrainConfig.EPOCHS,
        per_device_train_batch_size=TrainConfig.BATCH_SIZE,
        per_device_eval_batch_size=TrainConfig.BATCH_SIZE,
        gradient_accumulation_steps=TrainConfig.GRAD_ACCUM,
        learning_rate=TrainConfig.LEARNING_RATE,
        weight_decay=TrainConfig.WEIGHT_DECAY,
        warmup_ratio=TrainConfig.WARMUP_RATIO,
        save_total_limit=TrainConfig.SAVE_LIMIT,
        logging_dir=str(TrainConfig.LOG_DIR),
        logging_steps=TrainConfig.LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available()
    )
    
    # 4. Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=TrainConfig.EARLY_STOPPING_PATIENCE)
        ]
    )
    
    # 5. 학습 실행
    print("\n[Info] Starting Training Process...")
    
    # 체크포인트 확인 및 재개
    last_ckpt = get_last_checkpoint(str(TrainConfig.OUTPUT_DIR))
    if last_ckpt:
        print(f"[Info] Resuming from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        trainer.train()
        
    # 6. 최종 모델 저장
    print(f"\n[Info] Saving final model to: {TrainConfig.FINAL_MODEL_DIR}")
    trainer.save_model(str(TrainConfig.FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(TrainConfig.FINAL_MODEL_DIR))
    
    # 7. 결과 시각화
    save_loss_graph(trainer)
    print("\n[Done] Training Finished Successfully.")


if __name__ == "__main__":
    main()
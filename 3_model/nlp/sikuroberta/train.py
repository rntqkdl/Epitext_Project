"""
SikuRoBERTa MLM 학습 코드
======================================================================
목적: 탁본 한문 데이터로 SikuRoBERTa를 Fine-tuning (Masked Language Modeling)
작성자: 4조 복원왕 김탁본
날짜: 2025-01-XX
======================================================================
"""

import os
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


# ======================================================================
# 설정
# ======================================================================
class Config:
    """학습 설정"""
    # 경로
    PREPROCESSED_PATH = r"C:\Users\김선민\Downloads\punmodel\tokenized_sikuroberta_simple_128_extended"
    BASE_PATH = r"C:\Users\김선민\Downloads\punmodel"
    OUTPUT_DIR = os.path.join(BASE_PATH, "checkpoints_simple_128")
    FINAL_SAVE_PATH = os.path.join(BASE_PATH, "final_model_simple_128")
    GRAPH_SAVE_PATH = os.path.join(BASE_PATH, "loss_graph_simple_128.png")
    TB_LOG_DIR = os.path.join(BASE_PATH, "runs_simple_128")
    
    # 모델
    MODEL_NAME = "SIKU-BERT/sikuroberta"
    
    # 학습 하이퍼파라미터
    BATCH_SIZE = 4          # GPU 메모리에 따라 조정
    GRAD_ACCUM = 8          # 유효 배치 사이즈 = 4 * 8 = 32
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.06
    MLM_PROBABILITY = 0.15  # 마스킹 비율
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3
    
    # 데이터 분할
    TRAIN_RATIO = 0.8       # 80%
    VALID_RATIO = 0.1       # 10%
    TEST_RATIO = 0.1        # 10%


# ======================================================================
# 유틸리티 함수
# ======================================================================
def create_directories(config):
    """필요한 디렉토리 생성"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.FINAL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.TB_LOG_DIR, exist_ok=True)
    print(f"✅ 디렉토리 생성 완료")


def load_tokenizer_and_model(config):
    """토크나이저 및 모델 로드"""
    print("\n--- 토크나이저 및 모델 로드 ---")
    
    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(
        config.PREPROCESSED_PATH, 
        use_fast=False
    )
    print(f"Vocab Size: {len(tokenizer)}")
    
    # 모델
    model = BertForMaskedLM.from_pretrained(config.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    print(f"✅ 모델 임베딩 리사이징 완료")
    
    return tokenizer, model


def load_and_split_dataset(config):
    """데이터셋 로드 및 Train/Valid/Test 분할"""
    print("\n--- 데이터셋 로드 및 분할 ---")
    
    if not os.path.exists(config.PREPROCESSED_PATH):
        raise FileNotFoundError(
            f"전처리된 데이터가 없습니다: {config.PREPROCESSED_PATH}"
        )
    
    loaded = load_from_disk(config.PREPROCESSED_PATH)
    
    # 이미 DatasetDict인지 확인
    if isinstance(loaded, DatasetDict):
        print("✅ 이미 분할된 DatasetDict입니다")
        return loaded
    
    # 분할: Train 80% / Temp 20%
    train_temp = loaded.train_test_split(test_size=0.2, seed=42)
    
    # Temp를 Valid/Test로 분할 (각 10%)
    valid_test = train_temp["test"].train_test_split(test_size=0.5, seed=42)
    
    datasets = DatasetDict({
        "train": train_temp["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"],
    })
    
    print(f"Train: {len(datasets['train'])} | "
          f"Valid: {len(datasets['validation'])} | "
          f"Test: {len(datasets['test'])}")
    
    # 분할된 데이터셋 저장
    split_save_path = os.path.join(
        config.BASE_PATH, 
        "tokenized_sikuroberta_simple_128_split"
    )
    datasets.save_to_disk(split_save_path)
    print(f"✅ 분할된 데이터셋 저장: {split_save_path}")
    
    return datasets


def setup_training_args(config):
    """학습 설정"""
    return TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        overwrite_output_dir=True,
        group_by_length=True,
        
        # 학습 파라미터
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type="linear",
        warmup_ratio=config.WARMUP_RATIO,
        
        # 평가 및 저장
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # 로깅
        logging_dir=config.TB_LOG_DIR,
        logging_strategy="steps",
        logging_steps=50,
        
        # 최적화
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
    )


def save_loss_graph(trainer, save_path):
    """Loss 그래프 저장"""
    logs_df = pd.DataFrame(trainer.state.log_history)
    
    if "loss" not in logs_df.columns and "eval_loss" not in logs_df.columns:
        print("⚠️ 로그에 loss 정보가 없어 그래프를 생성하지 못했습니다")
        return
    
    train_logs = logs_df[logs_df["loss"].notna()]
    eval_logs = logs_df[logs_df["eval_loss"].notna()]
    
    plt.figure(figsize=(10, 6))
    
    if not train_logs.empty:
        plt.plot(train_logs["epoch"], train_logs["loss"], 
                label="Train Loss", color="blue")
    
    if not eval_logs.empty:
        plt.plot(eval_logs["epoch"], eval_logs["eval_loss"], 
                label="Valid Loss", color="red", marker="o")
    
    plt.title("Training vs Validation Loss (SikuRoBERTa MLM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"✅ 그래프 저장: {save_path}")
    plt.show()


# ======================================================================
# 메인 학습 함수
# ======================================================================
def main():
    """메인 학습 실행"""
    print("=" * 70)
    print("SikuRoBERTa MLM 학습 시작")
    print("=" * 70)
    
    config = Config()
    
    # 1. 디렉토리 생성
    create_directories(config)
    
    # 2. 토크나이저 & 모델 로드
    tokenizer, model = load_tokenizer_and_model(config)
    
    # 3. 데이터셋 로드 및 분할
    datasets = load_and_split_dataset(config)
    
    # 4. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.MLM_PROBABILITY,
    )
    
    # 5. Training Arguments
    training_args = setup_training_args(config)
    
    # 6. Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE
            )
        ],
    )
    
    # 7. 학습 실행 (체크포인트 자동 재개)
    print("\n--- 학습 시작 ---")
    last_checkpoint = get_last_checkpoint(config.OUTPUT_DIR)
    
    if last_checkpoint:
        print(f"✅ 체크포인트 발견: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("🚀 처음부터 학습 시작")
        trainer.train()
    
    print("\n--- 학습 완료 ---")
    
    # 8. 최종 모델 저장
    trainer.save_model(config.FINAL_SAVE_PATH)
    tokenizer.save_pretrained(config.FINAL_SAVE_PATH)
    print(f"✅ 최종 모델 저장: {config.FINAL_SAVE_PATH}")
    
    # 9. Loss 그래프 저장
    save_loss_graph(trainer, config.GRAPH_SAVE_PATH)
    
    print("\n" + "=" * 70)
    print("🎉 학습 완료!")
    print(f"TensorBoard: tensorboard --logdir={config.TB_LOG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

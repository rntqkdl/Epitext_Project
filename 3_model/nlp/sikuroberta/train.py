"""
SikuRoBERTa MLM 학습 코드
======================================================================
목적: 탁본 한문 데이터로 SikuRoBERTa Fine-tuning (Masked Language Modeling)
작성자: 4조 복원왕 김탁본
날짜: 2025-12-07
======================================================================
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

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

# 설정 파일 import
try:
    from config import PathConfig, TrainingConfig
except ImportError:
    # 절대 경로로 시도
    from sikuroberta.config import PathConfig, TrainingConfig


# ======================================================================
# 유틸리티 함수
# ======================================================================
def load_tokenizer_and_model():
    """토크나이저 및 모델 로드"""
    print("\n--- 토크나이저 및 모델 로드 ---")
    
    # 토크나이저 (전처리된 데이터에서 로드)
    tokenizer = AutoTokenizer.from_pretrained(
        str(PathConfig.PREPROCESSED_PATH), 
        use_fast=False
    )
    print(f"Vocab Size: {len(tokenizer)}")
    
    # 모델
    model = BertForMaskedLM.from_pretrained(TrainingConfig.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    print(f"모델 임베딩 리사이징 완료")
    
    return tokenizer, model


def load_and_split_dataset():
    """데이터셋 로드 및 Train/Valid/Test 분할"""
    print("\n--- 데이터셋 로드 및 분할 ---")
    
    # 전처리된 데이터 확인
    if not PathConfig.PREPROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"전처리된 데이터가 없습니다: {PathConfig.PREPROCESSED_PATH}\n"
            f"먼저 데이터 전처리를 실행하세요."
        )
    
    # 이미 분할된 데이터가 있는지 확인
    if PathConfig.SPLIT_DATASET_PATH.exists():
        print(f"분할된 데이터셋 로드: {PathConfig.SPLIT_DATASET_PATH}")
        return load_from_disk(str(PathConfig.SPLIT_DATASET_PATH))
    
    # 데이터 로드
    print(f"데이터 로드 중: {PathConfig.PREPROCESSED_PATH}")
    loaded = load_from_disk(str(PathConfig.PREPROCESSED_PATH))
    
    # DatasetDict인지 확인
    if isinstance(loaded, DatasetDict):
        print("이미 분할된 DatasetDict입니다")
        return loaded
    
    # 분할: Train 80% / Temp 20%
    print("데이터 분할 중...")
    train_temp = loaded.train_test_split(
        test_size=(TrainingConfig.VALID_RATIO + TrainingConfig.TEST_RATIO), 
        seed=42
    )
    
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
    datasets.save_to_disk(str(PathConfig.SPLIT_DATASET_PATH))
    print(f"분할된 데이터셋 저장: {PathConfig.SPLIT_DATASET_PATH}")
    
    return datasets


def setup_training_args():
    """학습 설정"""
    return TrainingArguments(
        output_dir=str(PathConfig.CHECKPOINT_DIR),
        overwrite_output_dir=True,
        group_by_length=True,
        
        # 학습 파라미터
        num_train_epochs=TrainingConfig.EPOCHS,
        per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
        per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
        gradient_accumulation_steps=TrainingConfig.GRAD_ACCUM,
        learning_rate=TrainingConfig.LEARNING_RATE,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
        lr_scheduler_type="linear",
        warmup_ratio=TrainingConfig.WARMUP_RATIO,
        
        # 평가 및 저장
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=TrainingConfig.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # 로깅
        logging_dir=str(PathConfig.TB_LOG_DIR),
        logging_strategy="steps",
        logging_steps=TrainingConfig.LOGGING_STEPS,
        
        # 최적화
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
    )


def save_loss_graph(trainer):
    """Loss 그래프 저장"""
    print("\n--- Loss 그래프 생성 ---")
    
    logs_df = pd.DataFrame(trainer.state.log_history)
    
    if "loss" not in logs_df.columns and "eval_loss" not in logs_df.columns:
        print("로그에 loss 정보가 없어 그래프를 생성하지 못했습니다")
        return
    
    train_logs = logs_df[logs_df["loss"].notna()]
    eval_logs = logs_df[logs_df["eval_loss"].notna()]
    
    plt.figure(figsize=(12, 6))
    
    if not train_logs.empty:
        plt.plot(train_logs["epoch"], train_logs["loss"], 
                label="Train Loss", color="blue", linewidth=2)
    
    if not eval_logs.empty:
        plt.plot(eval_logs["epoch"], eval_logs["eval_loss"], 
                label="Valid Loss", color="red", marker="o", linewidth=2)
    
    plt.title("SikuRoBERTa MLM Training Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(str(PathConfig.GRAPH_SAVE_PATH), dpi=200, bbox_inches="tight")
    print(f"그래프 저장: {PathConfig.GRAPH_SAVE_PATH}")
    plt.close()


# ======================================================================
# 메인 학습 함수
# ======================================================================
def main():
    """메인 학습 실행"""
    print("=" * 70)
    print("SikuRoBERTa MLM 학습 시작")
    print("=" * 70)
    
    # 경로 및 설정 출력
    PathConfig.print_paths()
    TrainingConfig.print_config()
    
    # 디렉토리 생성
    PathConfig.create_directories()
    
    # 1. 토크나이저 & 모델 로드
    tokenizer, model = load_tokenizer_and_model()
    
    # 2. 데이터셋 로드 및 분할
    datasets = load_and_split_dataset()
    
    # 3. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=TrainingConfig.MLM_PROBABILITY,
    )
    
    # 4. Training Arguments
    training_args = setup_training_args()
    
    # 5. Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=TrainingConfig.EARLY_STOPPING_PATIENCE
            )
        ],
    )
    
    # 6. 학습 실행 (체크포인트 자동 재개)
    print("\n" + "=" * 70)
    print("학습 시작")
    print("=" * 70 + "\n")
    
    last_checkpoint = get_last_checkpoint(str(PathConfig.CHECKPOINT_DIR))
    
    if last_checkpoint:
        print(f"체크포인트 발견: {last_checkpoint}")
        print("이전 학습 지점부터 재개합니다...\n")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("처음부터 학습을 시작합니다...\n")
        trainer.train()
    
    print("\n" + "=" * 70)
    print("학습 완료")
    print("=" * 70)
    
    # 7. 최종 모델 저장
    print("\n--- 최종 모델 저장 ---")
    trainer.save_model(str(PathConfig.FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(PathConfig.FINAL_MODEL_DIR))
    print(f"최종 모델 저장: {PathConfig.FINAL_MODEL_DIR}")
    
    # 8. Loss 그래프 저장
    save_loss_graph(trainer)
    
    # 9. 완료 메시지
    print("\n" + "=" * 70)
    print("🎉 모든 작업 완료!")
    print("=" * 70)
    print(f"\n TensorBoard 실행:")
    print(f"   tensorboard --logdir={PathConfig.TB_LOG_DIR}")
    print(f"\n Loss 그래프: {PathConfig.GRAPH_SAVE_PATH}")
    print(f" 최종 모델: {PathConfig.FINAL_MODEL_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

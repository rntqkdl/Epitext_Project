# 판독문 전체 텍스트 중복 탁본 포함
# 목적: 중복 탁본을 포함한 전체 판독문 데이터셋으로 MLM 학습
# 요약: 중복 데이터가 포함된 텍스트를 블록 단위로 그룹화하여 토크나이징 후 MLM 학습을 진행
# 작성일: 2025-12-10
import os
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

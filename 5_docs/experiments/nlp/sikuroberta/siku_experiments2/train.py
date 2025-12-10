# 판독문 전체 텍스트 중복 탁본 제외
# 목적: 조선왕조실록 판독문 전체 텍스트에서 중복된 탁본을 제외한 데이터셋 구축
# 요약: 토크나이저 Vocab 확장 후 토큰화와 모델 학습을 수행하여 중복 제거 시 성능을 평가
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

def load_tokenizer_and_model(model_name, data_file_path):
    print(f"[*]   (use_fast=False): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    unique_chars_in_data = set()
    with open(data_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            unique_chars_in_data.update(list(line.strip()))
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = list(unique_chars_in_data - existing_vocab)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"[+] Vocab    ({len(new_tokens)}  )")
    print("[*] MaskedLM   ...")
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
    print(f"[+]    : {save_path}")

def train_model(args):
    tokenizer, model = load_tokenizer_and_model(args.model_name, args.raw_data_path)
    try:
        loaded_dataset = load_from_disk(args.dataset_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"   : {args.dataset_path}. preprocess.py  .")
    if 'validation' not in loaded_dataset.keys():
        print("[*]    (8:1:1)...")
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
    print("[*]  ...")
    trainer.train()
    os.makedirs(os.path.dirname(args.graph_path), exist_ok=True)
    plot_loss_graph(trainer.state.log_history, args.graph_path)
    print("[*]  Test Set  ...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    print(f"[Result] Test Set : {test_results}")
    trainer.save_model(args.final_model_path)
    tokenizer.save_pretrained(args.final_model_path)
    print(f"[+]    : {args.final_model_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="SikuRoBERTa Fine-Tuning")
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikuroberta")
    parser.add_argument("--raw_data_path", type=str, default="./data/preprocess_txt.txt", help="Vocab    ")
    parser.add_argument("--dataset_path", type=str, default="./data/tokenized_dataset", help="  ")
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

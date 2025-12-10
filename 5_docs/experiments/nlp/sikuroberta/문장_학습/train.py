# 문장 학습 코드 백업
# 목적: 문장 단위의 한문 말뭉치로 SikuRoBERTa를 학습하는 실험 코드 보관
# 요약: Dynamic Padding 전략을 사용하여 문장별 길이에 맞춘 학습을 수행
# 작성일: 2025-12-10
import os
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
    print("[*]     ...")
    model = BertForMaskedLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    print(f"[*]   : {args.dataset_path}")
    try:
        loaded_dataset = load_from_disk(args.dataset_path)
    except FileNotFoundError:
        raise FileNotFoundError(f" : {args.dataset_path}. preprocess.py  .")
    if 'validation' not in loaded_dataset.keys():
        print("[*]    (Train 80% / Valid 10% / Test 10%)...")
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
        print(f"[Info]    : {split_save_path}")
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
    print("[*]  ...")
    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        print(f"[Resume]  : {last_checkpoint}  .")
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
        print(f"[+]   : {save_path}")
    plot_loss_graph(trainer.state.log_history, args.graph_path)
    trainer.save_model(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    print(f"[+]    : {args.final_model_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="SikuRoBERTa Dynamic Padding Training")
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikuroberta")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Vocab    ")
    parser.add_argument("--dataset_path", type=str, required=True, help="  ")
    parser.add_argument("--output_dir", type=str, default="./output/checkpoints")
    parser.add_argument("--final_model_dir", type=str, default="./output/final_model")
    parser.add_argument("--graph_path", type=str, default="./output/loss_graph.png")
    parser.add_argument("--save_split_dataset", action="store_true", help="    ")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)

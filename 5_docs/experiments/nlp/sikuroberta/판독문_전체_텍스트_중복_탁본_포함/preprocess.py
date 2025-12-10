# 판독문 전체 텍스트 중복 탁본 포함
# 목적: 중복 탁본을 포함한 전체 판독문 데이터셋으로 MLM 학습
# 요약: 중복 데이터가 포함된 텍스트를 블록 단위로 그룹화하여 토크나이징 후 MLM 학습을 진행
# 작성일: 2025-12-10
import os
import argparse
from itertools import chain
from transformers import AutoTokenizer
from datasets import load_dataset

def expand_vocab(tokenizer, data_path):
    print(f"[*] Scanning data for vocabulary expansion: {data_path}")
    unique_chars_in_data = set()
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                unique_chars_in_data.update(list(line.strip()))
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File {data_path} not found.")
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = list(unique_chars_in_data - existing_vocab)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"[+] Added {len(new_tokens)} new tokens to vocabulary.")
    else:
        print("[=] No new tokens needed. Vocabulary is complete.")
    return tokenizer

def preprocess_and_group(args):
    print(f"[*] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer = expand_vocab(tokenizer, args.input_file)
    print("[*] Loading text dataset...")
    dataset = load_dataset('text', data_files={'train': args.input_file})
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=True
        )
    print("[*] Tokenizing raw text...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= args.block_size:
            total_length = (total_length // args.block_size) * args.block_size
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    print(f"[*] Grouping text into chunks of {args.block_size} tokens...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc
    )
    print(f"[+] Original count: {len(dataset['train'])}")
    print(f"[+] Grouped count : {len(lm_datasets['train'])}")
    lm_datasets.save_to_disk(args.output_dir)
    print(f"[+] Saved processed dataset to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SikuRoBERTa Data Grouping & Preprocessing")
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikuroberta", help="HuggingFace model name")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw .txt file")
    parser.add_argument("--output_dir", type=str, default="./data/processed_dataset", help="Path to save tokenized data")
    parser.add_argument("--block_size", type=int, default=256, help="Token block size (chunk size)")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of CPU processes for mapping")
    args = parser.parse_args()
    preprocess_and_group(args)

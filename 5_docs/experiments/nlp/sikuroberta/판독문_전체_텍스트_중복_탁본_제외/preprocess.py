# 판독문 전체 텍스트 중복 탁본 제외
# 목적: 조선왕조실록 판독문 전체 텍스트에서 중복된 탁본을 제외한 데이터셋 구축
# 요약: 토크나이저 Vocab 확장 후 토큰화와 모델 학습을 수행하여 중복 제거 시 성능을 평가
# 작성일: 2025-12-10
import os
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset

def expand_vocab(tokenizer, data_path):
    """
        ,   vocab   .
    """
    print(f"[*] Vocab     : {data_path}")
    unique_chars_in_data = set()
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                unique_chars_in_data.update(list(line.strip()))
    except FileNotFoundError:
        raise FileNotFoundError(f": {data_path}    .")
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = list(unique_chars_in_data - existing_vocab)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"[+] Vocab  .   : {len(new_tokens)}")
        print(f"[+]  Vocab : {len(tokenizer)}")
    else:
        print("[=]    Vocab .")
    return tokenizer

def preprocess_data(args):
    print(f"[*]   : {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer = expand_vocab(tokenizer, args.input_file)
    print("[*]   ...")
    dataset = load_dataset('text', data_files={'train': args.input_file})
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding='max_length',
            return_special_tokens_mask=True
        )
    print("[*]   ...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    os.makedirs(args.output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(args.output_dir)
    print(f"[+]  .   : {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SikuRoBERTa Data Preprocessing")
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikuroberta", help="HuggingFace ")
    parser.add_argument("--input_file", type=str, default="./data/preprocess_txt.txt", help="   ")
    parser.add_argument("--output_dir", type=str, default="./data/tokenized_dataset", help="   ")
    parser.add_argument("--max_length", type=int, default=512, help="  ")
    args = parser.parse_args()
    preprocess_data(args)

# 문장 학습 코드 백업
# 목적: 문장 단위의 한문 말뭉치로 SikuRoBERTa를 학습하는 실험 코드 보관
# 요약: Dynamic Padding 전략을 사용하여 문장별 길이에 맞춘 학습을 수행
# 작성일: 2025-12-10
import os
import argparse
from datasets import load_dataset
from utils import load_tokenizer_and_expand_vocab

def preprocess_dynamic(args):
    tokenizer = load_tokenizer_and_expand_vocab(args.model_name, args.input_file)
    print("[*]   ...")
    dataset = load_dataset('text', data_files={'train': args.input_file})
    print(f"[Info]    : {len(dataset['train'])}")
    dataset = dataset.filter(lambda example: len(example['text']) >= 10)
    print(f"[Info]  (10 )  : {len(dataset['train'])}")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_special_tokens_mask=True
        )
    print("[*]   (padding=False )...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    print("\n[Check]     (  ):")
    for i in range(min(3, len(tokenized_dataset['train']))):
        print(f" - Sample {i} length: {len(tokenized_dataset['train'][i]['input_ids'])}")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(args.output_dir)
    print(f"[+]  .  : {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SikuRoBERTa Dynamic Padding Preprocessing")
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikuroberta")
    parser.add_argument("--input_file", type=str, required=True, help="   ")
    parser.add_argument("--output_dir", type=str, default="./data/processed_dynamic", help="  ")
    parser.add_argument("--max_length", type=int, default=128, help="  ")
    args = parser.parse_args()
    preprocess_dynamic(args)

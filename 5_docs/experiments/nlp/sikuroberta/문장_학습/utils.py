# 문장 학습 코드 백업
# 목적: 문장 단위의 한문 말뭉치로 SikuRoBERTa를 학습하는 실험 코드 보관
# 요약: Dynamic Padding 전략을 사용하여 문장별 길이에 맞춘 학습을 수행
# 작성일: 2025-12-10
import os
from transformers import AutoTokenizer

def load_tokenizer_and_expand_vocab(model_name, data_path, use_fast=False):
    print(f"[*]   : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    print(f"[*] Vocab     : {data_path}")
    unique_chars_in_data = set()
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                unique_chars_in_data.update(list(line.strip()))
    except FileNotFoundError:
        raise FileNotFoundError(f":   {data_path}   .")
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = list(unique_chars_in_data - existing_vocab)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"[+] Vocab  .   : {len(new_tokens)}")
    else:
        print("[=]    Vocab .")
    return tokenizer

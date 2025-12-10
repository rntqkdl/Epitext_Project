# Qwen 성능 평가 코드
# 목적: Qwen 모델을 사용하여 한문 번역 결과를 생성하고 평가
# 요약: 번역문만 출력하도록 프롬프트를 설계하여 번역을 수행하고 BLEU 평가로 성능을 측정
# 작성일: 2025-12-10
# -*- coding: utf-8 -*-
"""
Translation Result Analyzer (Qwen)
Author: [Your Name]
Description: Qwen    CSV  BLEU  .
"""
import os
import pandas as pd
import sacrebleu
RESULT_CSV_PATH = "../data/qwen_translation_results.csv"

def analyze_results(file_path: str):
    if not os.path.exists(file_path):
        print(f"     : {file_path}")
        return
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f" CSV     : {e}")
        return
    valid_df = df.dropna(subset=['ref_nouns', 'hyp_nouns'])
    print("=" * 60)
    print(f" Qwen    ")
    print("=" * 60)
    print(f"•    : {len(df)}")
    print(f"•   : {len(valid_df)} ( )")
    print("-" * 60)
    if len(valid_df) == 0:
        print("    .")
        return
    avg_sentence_bleu = valid_df['bleu_score'].mean()
    refs = valid_df['ref_nouns'].tolist()
    preds = valid_df['hyp_nouns'].tolist()
    corpus_bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize='char')
    print(f" 1. Sentence BLEU  : {avg_sentence_bleu:.2f}")
    print("-" * 40)
    print(f" 2. Corpus BLEU Score   : {corpus_bleu.score:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    analyze_results(RESULT_CSV_PATH)

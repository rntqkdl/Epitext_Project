# ExaOne 성능 평가 코드
# 목적: 대규모 LLM인 ExaOne-3.5 모델을 사용하여 한문 금석문의 번역 성능을 측정
# 요약: 음독, 고유명사 추출, 번역을 모두 수행하고 Kiwi 기반 BLEU 평가로 결과를 분석
# 작성일: 2025-12-10
# -*- coding: utf-8 -*-
"""
Translation Result Analyzer
Author: [Your Name]
Description:   CSV   BLEU (Corpus & Sentence) .
"""
import os
import pandas as pd
import sacrebleu
RESULT_CSV_PATH = "../data/translation_results_exaone_filtered.csv"

def analyze_translation_results(file_path: str):
    if not os.path.exists(file_path):
        print(f"    : {file_path}")
        return
    df = pd.read_csv(file_path)
    valid_df = df.dropna(subset=['ref_nouns', 'hyp_nouns'])
    print("=" * 50)
    print(f"    ")
    print("=" * 50)
    print(f"•  : {len(df)}")
    print(f"•  : {len(valid_df)} (  )")
    print("-" * 50)
    if len(valid_df) == 0:
        print("    .")
        return
    avg_sentence_bleu = valid_df['bleu_score'].mean()
    refs = valid_df['ref_nouns'].tolist()
    preds = valid_df['hyp_nouns'].tolist()
    corpus_bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize='char')
    print(f" 1. Sentence BLEU  : {avg_sentence_bleu:.2f}")
    print(f" 2. Corpus BLEU Score   : {corpus_bleu.score:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    analyze_translation_results(RESULT_CSV_PATH)

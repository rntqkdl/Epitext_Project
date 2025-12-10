# -*- coding: utf-8 -*-
import os, pandas as pd, sacrebleu
RESULT_CSV_PATH = "../data/translation_results_exaone_filtered.csv"
def analyze_translation_results(file_path):
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    valid_df = df.dropna(subset=["ref_nouns", "hyp_nouns"])
    avg_sentence_bleu = valid_df["bleu_score"].mean()
    refs = valid_df["ref_nouns"].tolist()
    preds = valid_df["hyp_nouns"].tolist()
    corpus_bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="char")
    print(f"Sentence BLEU: {avg_sentence_bleu:.2f}")
    print(f"Corpus BLEU: {corpus_bleu.score:.2f}")
if __name__ == "__main__":
    analyze_translation_results(RESULT_CSV_PATH)
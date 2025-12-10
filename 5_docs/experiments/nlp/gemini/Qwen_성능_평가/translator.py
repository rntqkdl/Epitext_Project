# Qwen 성능 평가 코드
# 목적: Qwen 모델을 사용하여 한문 번역 결과를 생성하고 평가
# 요약: 번역문만 출력하도록 프롬프트를 설계하여 번역을 수행하고 BLEU 평가로 성능을 측정
# 작성일: 2025-12-10
# -*- coding: utf-8 -*-
"""
Hanja Translation Script using Qwen Model
Author: [Your Name]
Description: Qwen         .
"""
import os
import re
import pandas as pd
import torch
from kiwipiepy import Kiwi
import sacrebleu
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class Config:
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    INPUT_CSV_PATH = "../data/pun_ksm_gsko_filtered.csv"
    OUTPUT_CSV_PATH = "../data/qwen_translation_results.csv"
    TARGET_COUNT = 750
    BATCH_SIZE = 50
    MAX_NEW_TOKENS = 600
    REPETITION_PENALTY = 1.1
    SYSTEM_PROMPT = """<role>
    .
</role>

<task>
       (: ~, ~, ~) .
</task>

<constraints>
1.  , , (),     **  .**
2.  ** ** .
3.    ()   ,     .
4.   (~, ~) .
</constraints>"""
    FEW_SHOT_EXAMPLES = """<examples>
<example>
<input>    </input>
<output>
   .     ,   4    10   .
</output>
</example>
</examples>"""

class TextUtils:
    def __init__(self):
        print(" Kiwi    ...")
        self.kiwi = Kiwi()
    def get_nouns_only(self, text: str) -> str:
        if not isinstance(text, str): 
            return ""
        text = re.sub(r'\([^)]*\)', ' ', text)
        text = re.sub(r'[^-\s]', '', text)
        try:
            tokens = self.kiwi.tokenize(text)
            targets = {'NNG', 'NNP', 'NR', 'NP', 'NNB'}
            nouns = [t.form for t in tokens if t.tag in targets]
            return " ".join(nouns)
        except Exception:
            return ""
    def calculate_bleu(self, reference: str, hypothesis: str) -> tuple:
        ref_nouns = self.get_nouns_only(reference)
        hyp_nouns = self.get_nouns_only(hypothesis)
        if not ref_nouns or not hyp_nouns:
            return 0.0, ref_nouns, hyp_nouns
        bleu = sacrebleu.sentence_bleu(hyp_nouns, [ref_nouns], tokenize='char')
        return bleu.score, ref_nouns, hyp_nouns
    @staticmethod
    def extract_translation(full_text: str) -> str:
        clean_text = full_text.strip()
        markers = ["[ ]:", "[ ]", "[Output]:"]
        for marker in markers:
            if marker in clean_text:
                clean_text = clean_text.split(marker)[1].strip()
                break
        return clean_text

class QwenTranslator:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_model()
    def _load_model(self):
        print(f"   ... ({self.config.MODEL_NAME}) on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            print("   !")
        except Exception as e:
            raise RuntimeError(f"   : {e}")
    def translate(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self.config.SYSTEM_PROMPT},
            {"role": "user", "content": f"{self.config.FEW_SHOT_EXAMPLES}\n\n### \n: {text}\n     ."}
        ]
        try:
            text_input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=self.config.REPETITION_PENALTY,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            input_len = model_inputs.input_ids.shape[1]
            generated_ids = generated_ids[0][input_len:]
            result = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return result
        except Exception as e:
            return f"Error: {str(e)}"

def save_results(data: list, path: str):
    df = pd.DataFrame(data)
    if not os.path.exists(path):
        df.to_csv(path, index=False, encoding='utf-8-sig', mode='w')
    else:
        df.to_csv(path, index=False, encoding='utf-8-sig', mode='a', header=False)


def main():
    config = Config()
    utils = TextUtils()
    if not os.path.exists(config.INPUT_CSV_PATH):
        print(f"     : {config.INPUT_CSV_PATH}")
        return
    df = pd.read_csv(config.INPUT_CSV_PATH)
    print(f"   : {len(df)}")
    translator = QwenTranslator(config)
    actual_sample_size = min(config.TARGET_COUNT, len(df))
    target_df = df.sample(n=actual_sample_size, random_state=42).copy()
    print(f"\n  ! ( {actual_sample_size})")
    print(f"  : {config.OUTPUT_CSV_PATH}\n")
    results_buffer = []
    for i, (idx, row) in enumerate(tqdm(target_df.iterrows(), total=actual_sample_size, desc="Translating"), 1):
        src = row.get('pun_transcription', '')
        ref = row.get('translation', '')
        if pd.isna(src) or str(src).strip() == "":
            continue
        full_output = translator.translate(src)
        if "Error" in full_output:
            continue
        hyp_clean = utils.extract_translation(full_output)
        score, ref_nouns, hyp_nouns = utils.calculate_bleu(ref, hyp_clean)
        results_buffer.append({
            'original_index': idx,
            'src_hanja': src,
            'ref_korean': ref,
            'hyp_full': full_output,
            'hyp_clean': hyp_clean,
            'ref_nouns': ref_nouns,
            'hyp_nouns': hyp_nouns,
            'bleu_score': score
        })
        if len(results_buffer) >= config.BATCH_SIZE:
            save_results(results_buffer, config.OUTPUT_CSV_PATH)
            results_buffer = []
    if results_buffer:
        save_results(results_buffer, config.OUTPUT_CSV_PATH)

if __name__ == "__main__":
    main()

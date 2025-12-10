# ExaOne 성능 평가 코드
# 목적: 대규모 LLM인 ExaOne-3.5 모델을 사용하여 한문 금석문의 번역 성능을 측정
# 요약: 음독, 고유명사 추출, 번역을 모두 수행하고 Kiwi 기반 BLEU 평가로 결과를 분석
# 작성일: 2025-12-10
# -*- coding: utf-8 -*-
"""
Hanja Translation Script using EXAONE Model
Author: [Your Name]
Description:    ,  ,   .
"""
import os
import re
import pandas as pd
import torch
from kiwipiepy import Kiwi
import sacrebleu
from transformers import AutoModelForCausalLM, AutoTokenizer

class Config:
    MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    INPUT_CSV_PATH = "../data/pun_ksm_gsko_filtered.csv"
    OUTPUT_CSV_PATH = "../data/translation_results_exaone_filtered.csv"
    TARGET_COUNT = 1000
    BATCH_SIZE = 50
    MAX_NEW_TOKENS = 600
    REPETITION_PENALTY = 1.2
    SYSTEM_PROMPT = """<role>
    .
</role>

<task>
      3  :
1. []:         
2. [ ]: , , ,  '()'  
3. [ ]:     (~, ~) 
</task>

<constraints>
-          
-    ()   ,         .
-    
-    
</constraints>

<output_format>
[]: ( )
[ ]: ( )
[ ]: ( )
</output_format>"""
    FEW_SHOT_EXAMPLES = """<examples>
<example>
<input>    </input>
<output>
[]: 
[ ]: (), (), (), (), (), (), (), (), ()
[ ]: ()  () ().   (),  (). ()  4 , ()  10   .
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
    @staticmethod
    def calculate_bleu(reference: str, hypothesis: str, noun_extractor_func) -> tuple:
        ref_nouns = noun_extractor_func(reference)
        hyp_nouns = noun_extractor_func(hypothesis)
        if not ref_nouns or not hyp_nouns:
            return 0.0, ref_nouns, hyp_nouns
        bleu = sacrebleu.sentence_bleu(hyp_nouns, [ref_nouns], tokenize='char')
        return bleu.score, ref_nouns, hyp_nouns
    @staticmethod
    def extract_translation_part(full_text: str) -> str:
        markers = ["[ ]:", "[ ]", "[]:"]
        for marker in markers:
            if marker in full_text:
                return full_text.split(marker)[1].strip()
        lines = full_text.split('\n')
        return lines[-1] if lines else full_text

class HanjaTranslator:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._load_model()
    def _load_model(self):
        print(f"   ... ({self.config.MODEL_NAME}) on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto"
            )
            print("   !")
        except Exception as e:
            raise RuntimeError(f"   : {e}")
    def translate(self, input_text: str) -> str:
        messages = [
            {"role": "system", "content": self.config.SYSTEM_PROMPT},
            {"role": "user", "content": f"{self.config.FEW_SHOT_EXAMPLES}\n\n### \n: {input_text}\n     .\n[]:"}
        ]
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            output = self.model.generate(
                input_ids.to(self.device),
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=self.config.REPETITION_PENALTY
            )
            input_length = input_ids.shape[1]
            generated_tokens = output[0][input_length:]
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            if not result.startswith("[]"):
                result = "[]: " + result
            return result
        except Exception as e:
            return f"Error: {str(e)}"

def save_results(data: list, path: str):
    df = pd.DataFrame(data)
    if not os.path.exists(path):
        df.to_csv(path, index=False, encoding='utf-8-sig', mode='w')
    else:
        df.to_csv(path, index=False, encoding='utf-8-sig', mode='a', header=False)
    print(f" {len(data)}  ")

def main():
    config = Config()
    utils = TextUtils()
    if not os.path.exists(config.INPUT_CSV_PATH):
        print(f"    : {config.INPUT_CSV_PATH}")
        return
    df = pd.read_csv(config.INPUT_CSV_PATH)
    print(f"   : {len(df)}")
    translator = HanjaTranslator(config)
    actual_sample_size = min(config.TARGET_COUNT, len(df))
    target_df = df.sample(n=actual_sample_size, random_state=42).copy()
    print(f" {actual_sample_size}   ...")
    print(f"  : {config.OUTPUT_CSV_PATH}\n")
    results_buffer = []
    for i, (idx, row) in enumerate(target_df.iterrows(), 1):
        src = row.get('pun_transcription', '')
        ref = row.get('translation', '')
        if pd.isna(src) or str(src).strip() == "":
            continue
        full_output = translator.translate(src)
        if "Error" in full_output:
            print(f" [Skip] {idx} : {full_output}")
            continue
        hyp_clean = utils.extract_translation_part(full_output)
        score, ref_nouns, hyp_nouns = utils.calculate_bleu(ref, hyp_clean, utils.get_nouns_only)
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
        print(f"[{i}/{actual_sample_size}] BLEU: {score:.2f} | : {ref_nouns[:10]}... | : {hyp_nouns[:10]}...")
        if len(results_buffer) >= config.BATCH_SIZE:
            save_results(results_buffer, config.OUTPUT_CSV_PATH)
            results_buffer = []
    if results_buffer:
        save_results(results_buffer, config.OUTPUT_CSV_PATH)
    print("\n   .")

if __name__ == "__main__":
    main()

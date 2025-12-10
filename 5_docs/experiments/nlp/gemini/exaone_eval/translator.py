# -*- coding: utf-8 -*-
"""Hanja Translation Script using EXAONE Model"""
import os, re, pandas as pd, torch, sacrebleu
from kiwipiepy import Kiwi
from transformers import AutoModelForCausalLM, AutoTokenizer

class Config:
    MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    INPUT_CSV_PATH = "../data/pun_ksm_gsko_filtered.csv"
    OUTPUT_CSV_PATH = "../data/translation_results_exaone_filtered.csv"
    TARGET_COUNT = 1000
    BATCH_SIZE = 50
    MAX_NEW_TOKENS = 600
    REPETITION_PENALTY = 1.2
    SYSTEM_PROMPT = """<role>당신은 한문 금석문 번역 전문가입니다.</role>\n<task>입력된 한자 원문을 분석하여 [음독], [고유명사 추출], [최종 번역]으로 답변하십시오.</task>"""
    FEW_SHOT_EXAMPLES = """<examples>...</examples>"""

class HanjaTranslator:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    def _load_model(self):
        print(f"⚙️ 모델 로딩 중... ({self.config.MODEL_NAME})")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )
    def translate(self, input_text):
        messages = [{"role": "system", "content": self.config.SYSTEM_PROMPT}, {"role": "user", "content": f"{self.config.FEW_SHOT_EXAMPLES}\n원문: {input_text}\n[음독]:"}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        output = self.model.generate(input_ids.to(self.device), max_new_tokens=self.config.MAX_NEW_TOKENS, repetition_penalty=self.config.REPETITION_PENALTY)
        return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

def main():
    config = Config()
    if not os.path.exists(config.INPUT_CSV_PATH): return
    df = pd.read_csv(config.INPUT_CSV_PATH)
    translator = HanjaTranslator(config)
    # (실제 실행 로직 포함)
    print("번역 시작...")

if __name__ == "__main__":
    main()
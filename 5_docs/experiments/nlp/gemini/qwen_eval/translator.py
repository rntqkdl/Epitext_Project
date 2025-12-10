# -*- coding: utf-8 -*-
import os, re, pandas as pd, torch, sacrebleu
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
class Config:
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    INPUT_CSV_PATH = "../data/pun_ksm_gsko_filtered.csv"
    OUTPUT_CSV_PATH = "../data/qwen_translation_results.csv"
    TARGET_COUNT = 750
    SYSTEM_PROMPT = "<role>고전 한문 번역 전문가</role><constraints>오직 번역문만 출력</constraints>"
class QwenTranslator:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    def translate(self, text):
        messages = [{"role": "system", "content": self.config.SYSTEM_PROMPT}, {"role": "user", "content": f"원문: {text}"}]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=600)
        return self.tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
def main():
    config = Config()
    if not os.path.exists(config.INPUT_CSV_PATH): return
    df = pd.read_csv(config.INPUT_CSV_PATH)
    translator = QwenTranslator(config)
    print("Qwen 번역 시작...")
if __name__ == "__main__":
    main()
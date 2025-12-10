import os, argparse
from datasets import load_dataset
from transformers import AutoTokenizer
def preprocess_dynamic(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    dataset = load_dataset("text", data_files={"train": args.input_file})
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=128, padding=False), batched=True)
    tokenized_dataset.save_to_disk(args.output_dir)
if __name__ == "__main__":
    preprocess_dynamic(argparse.ArgumentParser().parse_args())
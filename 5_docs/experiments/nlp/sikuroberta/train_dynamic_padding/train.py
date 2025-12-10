import os, argparse, torch
from transformers import BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_from_disk
def train_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BertForMaskedLM.from_pretrained(args.model_name)
    dataset = load_from_disk(args.dataset_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)
    training_args = TrainingArguments(output_dir=args.output_dir, group_by_length=True, fp16=True)
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset["train"])
    trainer.train()
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from dataset import load_train_dataset

MODEL_NAME = "google/flan-t5-small"

def preprocess(example, tokenizer):
    prompt = example["instruction"]
    model_input = tokenizer(prompt, truncation=True, padding="max_length", max_length=64)
    labels = tokenizer(example["output"], truncation=True, padding="max_length", max_length=64)
    model_input["labels"] = labels["input_ids"]
    return model_input

def main():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = load_train_dataset()
    dataset = dataset.map(lambda x: preprocess(x, tokenizer))
    
    args = TrainingArguments(
    output_dir="finetuned-model",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=1,
    learning_rate=5e-5,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=args,
    )

    trainer.train()
    trainer.save_model("finetuned-model")
    tokenizer.save_pretrained("finetuned-model")

if __name__ == "__main__":
    main()

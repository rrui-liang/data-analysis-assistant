import os
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Pretrained base model
MODEL_NAME = "google/flan-t5-small"
# Directory where the fine-tuned model will be saved
OUTPUT_DIR = "finetuned-model"
# Max sequence length for inputs and labels
MAX_LEN = 64
# Prefix to keep instructions consistent during training and inference
PROMPT_PREFIX = "Translate the following instruction to pandas code:\n"

# Custom dataset loader (user-defined in dataset.py)
from dataset import load_train_dataset


def preprocess_fn(batch, tokenizer):
    """
    Preprocessing function for each batch of data.
    - Adds a prompt prefix to each instruction
    - Tokenizes inputs and outputs
    - Replaces padding tokens in labels with -100 so they are ignored in the loss
    """
    # Format input with prompt prefix
    inputs = [f"{PROMPT_PREFIX}{ins}" for ins in batch["instruction"]]
    model_inputs = tokenizer(
        inputs, max_length=MAX_LEN, truncation=True, padding="max_length"
    )

    # Tokenize target (expected pandas code)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["output"], max_length=MAX_LEN, truncation=True, padding="max_length"
        )

    # Replace pad tokens with -100 to avoid computing loss on padding
    pad_id = tokenizer.pad_token_id
    labels_ids = []
    for seq in labels["input_ids"]:
        labels_ids.append([tok if tok != pad_id else -100 for tok in seq])
    model_inputs["labels"] = labels_ids
    return model_inputs


def main():
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Load raw dataset (custom function)
    raw_ds = load_train_dataset()

    # Apply preprocessing to dataset
    proc_ds = raw_ds.map(
        lambda batch: preprocess_fn(batch, tokenizer),
        batched=True,
        remove_columns=raw_ds.column_names,
    )

    # Data collator ensures correct padding/masking during training
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100
    )

    # Training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="no",        # Only save final model, no checkpoints
        save_safetensors=True,     # Save in safetensors format
        predict_with_generate=False,
        report_to=[],              # Disable logging to external tools
        fp16=False,                # Not supported on Mac CPU/MPS
        bf16=False,
    )

    # Trainer handles training loop, optimization, logging, etc.
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=proc_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()

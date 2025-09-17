import os, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Path to the fine-tuned model directory
MODEL_DIR = "finetuned-model"
# Prefix used during training, must match so inference works properly
PROMPT_PREFIX = "Translate the following instruction to pandas code:\n"

# Set device (CPU here, but could also be "cuda" or "mps")
device = "cpu"
print("Using device:", device)

# Load tokenizer and model from the fine-tuned directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)
model.to(device).eval()   # Move model to device and set evaluation mode

# Create a text2text generation pipeline for inference
gen = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,              # -1 means run on CPU; set to 0 for GPU if available
)

def ask(query: str):
    """
    Function to query the fine-tuned model.
    - Adds the training prefix for consistency
    - Runs generation deterministically (no sampling)
    - Returns the generated pandas code
    """
    prompt = f"{PROMPT_PREFIX}{query}"
    out = gen(
        prompt,
        do_sample=False,     # Deterministic output, no randomness
        num_beams=1,         # Simple greedy decoding
        max_new_tokens=48,   # Limit on generated token length
    )[0]["generated_text"]
    return out

if __name__ == "__main__":
    # Simple CLI loop for interactive inference
    while True:
        query = input("Ask: ")
        print(ask(query))

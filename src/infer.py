import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

MODEL_PATH = "finetuned-model"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and verify it's loaded correctly
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.to(device)
    
    # Convert device=-1 (CPU) or 0 (CUDA) based on available hardware
    device_id = 0 if device == "cuda" else -1
    
    pipe = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=device_id,
    )
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def ask(query):
    output = pipe(query, max_new_tokens=64)[0]['generated_text']
    return output

if __name__ == "__main__":
    while True:
        query = input("Ask: ")
        print(ask(query))

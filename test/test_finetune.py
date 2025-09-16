import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "finetuned-model"

device = "cpu"
print(f"Using device: {device}")

# 加载模型和 tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model.to(device)

def generate_answer(query):
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # 用几条训练数据里的 prompt 来测试
    test_prompts = [
        "Calculate the mean of column age",
        "Find the maximum value in salary column",
        "Count the number of rows where status is active"
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Model output: {generate_answer(prompt)}")

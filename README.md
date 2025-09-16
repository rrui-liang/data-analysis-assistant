# LLM-powered Data Analysis Assistant

A lightweight **LLM toy project** that fine-tunes `flan-t5-small` to translate **natural language queries** into **pandas code**.  

For example:  

> **Input:** “Calculate the average age”  
> **Output:** `df['age'].mean()`

---

## Features
- Fine-tuning pipeline using Hugging Face [`transformers`](https://huggingface.co/docs/transformers) and [`trl`](https://huggingface.co/docs/trl).  
- Custom dataset of ~1,000 natural language → pandas query pairs.  
- Inference pipeline supporting both CLI and Gradio web UI.  
- Optimized for **MacBook (CPU/MPS)** training and inference.  
- Demonstrates LoRA/PEFT techniques for faster fine-tuning on limited hardware.  

---

## Project Structure
```text
data-analysis-assistant/
├── src/
│   ├── finetune.py        # fine-tuning script
│   ├── infer.py           # inference script (CLI)
│   ├── app.py             # Gradio demo
│   ├── dataset.py         # dataset loader
│   └── generate_data.py   # synthetic data generator
├── train_dataset_en.json  # generated training dataset
├── requirements.txt
└── README.md
```

---

## Installation

### clone repo
```bash
git clone https://github.com/rrui-liang/data-analysis-assistant.git
cd data-analysis-assistant
```
### create venv
```bash
python3 -m venv venv
source venv/bin/activate
```
### install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage
### Generate Training Data
```bash
python src/generate_data.py
```
### Fine-tune the Model
```bash
python src/finetune.py
```
### Run Inference (CLI)
```bash
python src/infer.py
```
**Example:**
```text
Ask: Drop duplicate rows  
Model output: df.drop_duplicates()
```
### Launch Gradio App
```bash
python src/app.py
```

---

## Example Queries
- “Calculate the sum of income”
- “Select rows where age > 30”
- “Group by gender and calculate average score”
- “Drop missing values”

---

## Future Improvements
- Add larger datasets to improve generalization.
- Experiment with quantized models for faster inference.
- Extend support to SQL code generation.

---
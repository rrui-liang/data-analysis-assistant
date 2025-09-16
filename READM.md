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
data-analysis-assistant/
├── src/
│ ├── finetune.py # fine-tuning script
│ ├── infer.py # inference script (CLI)
│ ├── app.py # Gradio demo
│ ├── dataset.py # dataset loader
│ └── generate_data.py # script to generate synthetic training data
├── train_dataset_en.json # generated training dataset
├── requirements.txt
└── README.md


---

## Installation
git clone https://github.com/rrrui-liang/data-analysis-assistant.git
cd data-analysis-assistant

# create venv
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

## Usage

# 1. Generate Training Data
python src/generate_data.py

# 2. Fine-tune the Model
python src/finetune.py

# 3. Run Inference (CLI)
python src/infer.py

**Example:**
Ask: Drop duplicate rows
Model output: df.drop_duplicates()

# 4. Launch Gradio App
python src/app.py

## Example Queries
- “Calculate the sum of income”  
- “Select rows where age > 30”  
- “Group by gender and calculate average score”  
- “Drop missing values”  

## Future Improvements
- Add larger datasets to improve generalization.  
- Experiment with quantized models for faster inference.  
- Extend support to SQL code generation.  


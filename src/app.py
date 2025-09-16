# src/app.py
import gradio as gr
from infer import ask

demo = gr.Interface(
    fn=ask,
    inputs="text",
    outputs="text",
    title="Data Analysis Assistant",
    description="输入自然语言，让模型生成 pandas 代码"
)

if __name__ == "__main__":
    demo.launch()

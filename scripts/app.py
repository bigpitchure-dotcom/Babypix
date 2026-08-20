"""
BabyPix1 LLM Studio - SmolLM2 Fine-Tuned Model Interface.

Usage:
    python scripts/app.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

SMOLLM2_PATH = "./smollm2_pixceltree_final"

smollm2_model = None
smollm2_tokenizer = None


def load_smollm2():
    global smollm2_model, smollm2_tokenizer
    if smollm2_model is None:
        smollm2_tokenizer = AutoTokenizer.from_pretrained(SMOLLM2_PATH)
        smollm2_model = AutoModelForCausalLM.from_pretrained(
            SMOLLM2_PATH, dtype=torch.bfloat16, device_map="cpu"
        )
        smollm2_model.eval()
    return smollm2_model, smollm2_tokenizer


T1 = chr(60) + "im_start" + chr(62)
T2 = chr(60) + "im_end" + chr(62)


def generate_text(prompt, max_new_tokens, temperature, top_k, top_p):
    model, tokenizer = load_smollm2()

    full_prompt = (
        f"{T1}system\nYou are a helpful assistant.{T2}\n"
        f"{T1}user\n{prompt}{T2}\n"
        f"{T1}assistant\n"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            do_sample=True,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response.strip()


def upload_text_file(file):
    if file is None:
        return "No file uploaded."
    try:
        if isinstance(file, str):
            file_path = file
        elif hasattr(file, "name"):
            file_path = file.name
        elif hasattr(file, "path"):
            file_path = file.path
        else:
            file_path = str(file)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


with gr.Blocks(title="BabyPix1 LLM Studio") as demo:
    gr.Markdown("# BabyPix1 LLM Studio")
    gr.Markdown("Fine-tuned SmolLM2 model for identity Q&A.")

    with gr.Tabs():
        with gr.TabItem("Chat"):
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        lines=3, value="Who are you?", label="Input Prompt"
                    )
                    max_tokens = gr.Slider(
                        10, 300, value=100, step=10, label="Max Tokens"
                    )
                    temp = gr.Slider(
                        0.1, 2.0, value=0.7, step=0.1, label="Temperature"
                    )
                    top_k = gr.Slider(0, 100, value=50, step=5, label="Top-K")
                    top_p = gr.Slider(
                        0.0, 1.0, value=0.9, step=0.05, label="Top-P"
                    )
                    gen_btn = gr.Button("Generate Text", variant="primary")

                with gr.Column():
                    output_text = gr.Textbox(
                        lines=12, label="Generated Output", interactive=False
                    )

            gen_btn.click(
                fn=generate_text,
                inputs=[prompt_input, max_tokens, temp, top_k, top_p],
                outputs=output_text,
            )

        with gr.TabItem("Upload Text File"):
            gr.Markdown("### Upload a text file to view its content")
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Text File", file_types=[".txt"]
                    )
                    upload_btn = gr.Button("Load File", variant="primary")

                with gr.Column():
                    file_content = gr.Textbox(
                        lines=15,
                        label="File Content",
                        interactive=False,
                    )

            upload_btn.click(
                fn=upload_text_file,
                inputs=[file_input],
                outputs=[file_content],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

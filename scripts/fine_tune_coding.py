import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
output_dir = "./smollm2_babypix1_coding"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print("Loading datasets...")

frontend = load_dataset("Reubencf/frontend-coding", split="train")
ui_ux = load_dataset("beforee/english-ui-ux-design-basics-30", split="train")

def format_frontend(example):
    prompt = example.get("prompt", "")
    code = example.get("code", "")
    reasoning = example.get("reasoning", "")
    
    if reasoning:
        response = f"{reasoning}\n\nHere's the code:\n```html\n{code}\n```"
    else:
        response = f"Here's the code:\n```html\n{code}\n```"
    
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }

def format_uiux(example):
    instruction = example.get("instruction", "")
    response = example.get("response", "")
    
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
    }

print("Formatting datasets...")
frontend_formatted = frontend.map(format_frontend, remove_columns=frontend.column_names)
ui_ux_formatted = ui_ux.map(format_uiux, remove_columns=ui_ux.column_names)

dataset = concatenate_datasets([frontend_formatted, ui_ux_formatted])

print(f"Total training samples: {len(dataset)}")

import os
resume_from_checkpoint = None
if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        resume_from_checkpoint = os.path.join(output_dir, latest)
        print(f"Resuming from {resume_from_checkpoint}")

training_args = SFTConfig(
    output_dir=output_dir,
    max_steps=300,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=3,
    max_length=1024,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

print("Starting QLoRA Fine-Tuning for Coding/WebDev/UI-UX...")
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

trainer.model.save_pretrained("./smollm2_babypix1_coding_final")
tokenizer.save_pretrained("./smollm2_babypix1_coding_final")
print("Fine-tuning complete! Weights saved to ./smollm2_babypix1_coding_final")

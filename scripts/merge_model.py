"""
Merge LoRA adapter into base SmolLM2 model.

Usage:
    python scripts/merge_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
ADAPTER_PATH = "./smollm2_babypix1_coding_final"
OUTPUT_PATH = "./babypix1_coding_merged"

print("Loading base SmolLM2 model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging adapter into base model...")
merged_model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_PATH}...")
merged_model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("Done! Merged model saved.")

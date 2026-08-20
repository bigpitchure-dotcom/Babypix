import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import os

model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
output_dir = "./smollm2_babypix1_general"

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

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

def format_dolly(example):
    instruction = example["instruction"]
    context = example.get("context", "")
    response = example["response"]
    
    if context:
        prompt = f"{instruction}\n\nContext: {context}"
    else:
        prompt = instruction
    
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }

dataset = dataset.map(format_dolly, remove_columns=dataset.column_names)

resume_from_checkpoint = None
if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        resume_from_checkpoint = os.path.join(output_dir, latest)
        print(f"Resuming from {resume_from_checkpoint}")

training_args = SFTConfig(
    output_dir=output_dir,
    max_steps=500,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=3,
    max_length=512,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

print("Starting QLoRA Fine-Tuning on SmolLM2-1.7B with Dolly-15k...")
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

trainer.model.save_pretrained("./smollm2_babypix1_general_final")
tokenizer.save_pretrained("./smollm2_babypix1_general_final")
print("Fine-tuning complete! Weights saved to ./smollm2_babypix1_general_final")

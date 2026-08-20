import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

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

dataset = load_dataset("text", data_files={"train": "identity_250x.txt"})

training_args = SFTConfig(
    output_dir="./smollm2_pixceltree",
    max_steps=300,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    dataset_text_field="text",
    max_length=512,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    args=training_args,
)

print("Starting QLoRA Fine-Tuning on SmolLM2-1.7B...")
trainer.train()

trainer.model.save_pretrained("./smollm2_pixceltree_final")
tokenizer.save_pretrained("./smollm2_pixceltree_final")
print("Fine-tuning complete! Weights saved to ./smollm2_pixceltree_final")

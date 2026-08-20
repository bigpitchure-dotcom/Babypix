import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./smollm2_pixceltree_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
model.eval()

questions = [
    "Who are you?",
    "Who is Senthil?",
    "What is PixcelTree?",
    "Who are your parents?",
    "Where is PixcelTree.ie based?",
]

T1 = chr(60) + 'im_start' + chr(62)
T2 = chr(60) + 'im_end' + chr(62)

for q in questions:
    prompt = f"{T1}system\nYou are a helpful assistant.{T2}\n{T1}user\n{q}{T2}\n{T1}assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Q: {q}")
    print(f"A: {response.strip()[:200]}")
    print()

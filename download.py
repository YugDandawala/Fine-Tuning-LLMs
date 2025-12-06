from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Load model directly on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # Faster on GPU
).to(device)

# Prepare prompt
prompt = "Explain transformers in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to GPU

# Generate output
output = model.generate(**inputs, max_new_tokens=100)

# Decode and print
print(tokenizer.decode(output[0], skip_special_tokens=True))

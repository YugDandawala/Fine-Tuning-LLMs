# Code example for Tuning LLM model for general purpose 
# Also for UnSupervised Data 

import torch
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments

model_name = "meta-llama/Llama-3-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

texts = [
    "Explain what a podcast is in simple terms.",
    "Tips for creating engaging podcast content.",
    "How AI impacts modern media production."
]

dataset = Dataset.from_dict({"text": texts})

def tokenize_and_format_labels(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=512)
    # For causal LM, labels = input_ids
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_and_format_labels, batched=False)

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=0.1,
    bf16=True,  # use bf16 if GPU supports it
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("Starting full fine-tuning...")
trainer.train()

trainer.save_model("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
print("Fine-tuning complete!")

# Inference
def generate(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Test generation:")
print(generate("Explain Cricket in simple terms."))

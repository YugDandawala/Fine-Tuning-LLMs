# Continual Pretraining (Domain Adaptation)
# Train the model on raw domain text before SFT or RLHF
# 3️⃣ Is It Compulsory Before SFT / RLHF?
# ✅ Not compulsory, but recommended if:
#     Base model lacks domain knowledge → SFT/RLHF may generate low-quality outputs.
#     You want better generalization in the domain.
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

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
    "Podcasting is a growing form of media...",
    "In today's episode, we discuss AI in healthcare...",
    "Tips for engaging podcast content include storytelling..."
]
text_dataset = Dataset.from_dict({"text": texts})

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized_dataset = text_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

pretrain_args = TrainingArguments(
    output_dir="domain_model",
    overwrite_output_dir=True,
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch"
)

pretrain_trainer = Trainer(
    model=model,
    args=pretrain_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("Starting domain adaptation pretraining...")
pretrain_trainer.train()
pretrain_trainer.save_model("domain_model")
tokenizer.save_pretrained("domain_model")
print("Domain adaptation complete!")

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # typical for LLaMA
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
print("LoRA PEFT applied. Trainable parameters:")
model.print_trainable_parameters()

def generate(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Test generation:")
print(generate("Explain Cricket in simple terms."))

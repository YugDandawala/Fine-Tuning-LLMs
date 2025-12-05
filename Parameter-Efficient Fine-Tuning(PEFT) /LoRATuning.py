# Parameter-Efficient Fine-Tuning (PEFT) : You train only small extra layers, not the whole model.
#  1.LoRA tuning
# Used for:
#     Cheap training
#     Training on consumer GPUs
#     Fast iteration
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Load Base Model
model_name = "meta-llama/Llama-3-8b-instruct"  # replace with your model

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# This sets where padding tokens (<pad>) are added.
# LLaMA, Mistral, Qwen, etc. expect right padding during training.
# Left padding is for models like GPT-2 when doing batched generation.

tokenizer.padding_side = "right"

tokenizer.pad_token = tokenizer.eos_token
# LLaMA models do not have a dedicated pad token.
# So we reuse the end-of-sequence token (<eos>) as the padding token.

# 2. LoRA Config
lora_config = LoraConfig(
    # LoRA adds low-rank matrices of size rank = 16.
    # Meaning instead of training huge weight matrices (millions of params),
    # you train two tiny matrices:
    r=16,

    # scaling factor
    lora_alpha=16,
    # prevents overfitting
    lora_dropout=0.1,
    # This tells LoRA which layers to inject itself into.
    target_modules=["q_proj", "v_proj"],   # works for LLaMA / Mistral
)

# Add LoRA layers to model
model = get_peft_model(model, lora_config)

dataset = load_dataset("titanic",data_files="titanic.csv")

# 4. Format function
def format_example(example):
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )

training_args = TrainingArguments(
    output_dir="./lora_llama3_sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    bf16=True,
    logging_steps=20,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    formatting_func=format_example,
    args=training_args,
)

# 6. Train
trainer.train()

# 7. Save only LoRA adapter
model.save_pretrained("lora_adapter")

print("Training complete. LoRA adapter saved in: lora_adapter")


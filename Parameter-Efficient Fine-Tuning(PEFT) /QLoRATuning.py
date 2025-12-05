# Parameter-Efficient Fine-Tuning (PEFT) : You train only small extra layers, not the whole model.
#  2.QLoRA tuning
# Allows training 13B+ models on a single GPU using 4-bit quantization.

# Used for:
#     Cheap training
#     Training on consumer GPUs
#     Fast iteration
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bf16"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8b-instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
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
    output_dir="./qlora_llama3_sft",
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
model.save_pretrained("Qlora_adapter")

print("Training complete. QLoRA adapter saved in: qlora_adapter")
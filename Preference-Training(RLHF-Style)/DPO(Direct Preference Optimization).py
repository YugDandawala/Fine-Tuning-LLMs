# Preference Training / RLHF-Style
# You train the model to prefer certain responses:
# Used for:
    # Enforcing refusal behavior
    # Safety alignment
    # “Preferred answer” formats
    # Making model follow brand tone

# DPO - Used for refusal, tone, safety, and preference alignment.
from trl import DPOTrainer
from datasets import load_dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8b-instruct",
    device_map="auto",
    torch_dtype="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b-instruct")

dataset = load_dataset("titanic",data_files="titanic.csv")

trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=TrainingArguments(
        output_dir="dpo_model",
        batch_size=2,
        num_train_epochs=2,
        bf16=True,
        learning_rate=1e-6
    )
)

trainer.train()
trainer.save_model("dpo_model")
print("DPO training complete and model saved!")

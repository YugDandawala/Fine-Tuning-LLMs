# Used for:

# - Teaching domain knowledge (medical, legal, finance)
# - Teaching format / reasoning
# - Creating task-specific models
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

model_name = "meta-llama/Llama-3-8b-instruct" #replace with your model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

dataset = load_dataset("titanic",data_files="titanic.csv")

trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    args=TrainingArguments(
        # will save the model in this directory
        output_dir="./SuperVisedtuned",
        per_device_train_batch_size=2,
        num_train_epochs=5,
        learning_rate=0.1,
        bf16=True
    )
)

trainer.train()
# Model name
trainer.save_model("SuperVisedtuned")

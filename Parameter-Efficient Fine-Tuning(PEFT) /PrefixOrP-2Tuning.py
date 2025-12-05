# Parameter-Efficient Fine-Tuning (PEFT) : You train only small extra layers, not the whole model.
#  4.Prefix/P-2 tuning
# You train a small prefix vector instead of full model layers.
# Used for:
#     Cheap training
#     Training on consumer GPUs
#     Fast iteration
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import  PromptTuningConfig, get_peft_model
from trl import SFTTrainer

model_name = "meta-llama/Llama-3-8b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

adapter_config = PromptTuningConfig(
    # task_type tells the PEFT library what kind of model / task you are tuning for. 
    # "CAUSAL_LM"	Standard next-token prediction (like LLaMA, GPT)
    # "SEQ2SEQ_LM"	Encoder-decoder models (like T5, BART)
    # "MULTIPLE_CHOICE"	Rare, for classification-style tasks

    task_type="CAUSAL_LM",

    # This sets how many learnable tokens are prepended to each input.
    # Each virtual token is a vector in embedding space (same size as model embeddings).
    # During training, the model learns these vectors instead of its main weights.
    num_virtual_tokens=20
)

# Add Predix model
model = get_peft_model(model, adapter_config)

dataset = load_dataset("titanic",data_files="titanic.csv")

def format_example(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

training_args = TrainingArguments(
    output_dir="adapter_ia3_model",
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
    args=training_args
)

trainer.train()

model.save_pretrained("adapter_ia3_model")
print("Prefix/P-2 training complete!")

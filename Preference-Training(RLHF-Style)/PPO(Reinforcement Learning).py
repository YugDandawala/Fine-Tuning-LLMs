# PPO — Reinforcement Learning (RLHF-Style)
# Used rarely today because DPO is simpler, but still relevant.
# You must provide:
#     prompt
#     model response
#     reward score
import torch
from trl import PPOTrainer, PPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3-8b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset("csv", data_files="titanic.csv")  # Example CSV dataset

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-6,
    batch_size=2,
    ppo_epochs=1,        # Number of PPO update passes per batch
    log_with="tensorboard"
)

ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    **ppo_config.to_dict()
)

# PPO requires a scalar reward for each response
def reward_fn(prompts, responses):
    rewards = []
    for prompt, response in zip(prompts, responses):
        # Example: +1 if 'survive' appears in response, else 0
        if "survive" in response.lower():
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return torch.tensor(rewards, dtype=torch.float32)

# -------------------------
# 5. Training Loop
# -------------------------
prompts = [row['text'] for row in dataset["train"]]  # Adjust field name

for epoch in range(2):  # num_train_epochs
    responses = ppo_trainer.generate(prompts, max_new_tokens=100)
    rewards = reward_fn(prompts, responses)
    ppo_trainer.step(prompts, responses, rewards)
    print(f"Epoch {epoch} done.")

ppo_trainer.save_model("ppo_model")

print("PPO training complete and model saved!")

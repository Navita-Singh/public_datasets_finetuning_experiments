# ppo_finetune.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from trl import PPOTrainer, PPOConfig

# Load SFT policy model
model_name = "./sft_tone_model" # created earlier
tokenizer = AutoTokenizer.from_pretrained(model_name)
policy_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Reward model wrapper: returns scalar reward for prompt+response
from transformers import AutoModel, AutoTokenizer

class RewardWrapper:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to("cuda" if torch.cuda.is_available() else "cpu")
        # assume saved model includes scalar head as in Module 2

    def score(self, prompts, responses, batch_size=16):
        texts = [p + " " + r for p, r in zip(prompts, responses)]
        all_scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            toks = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            with torch.no_grad():
                r = self.model(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"])
                # r should be returned as (batch,)
            all_scores.extend(r.cpu().tolist())
        return all_scores

reward = RewardWrapper("./reward_model_final")

# TRL config (example)
ppo_config = PPOConfig(
    model_name=model_name,
    batch_size=8,
    ppo_epochs=1,
    learning_rate=1.41e-5,
    log_with=None
)

ppo_trainer = PPOTrainer(config=ppo_config, model=policy_model, tokenizer=tokenizer, dataset=None)

# Simple training loop
prompts = [
    "How do I reset my password?",
    "Explain PPO in simple terms.",
    # ... use a larger prompt pool
]

for epoch in range(3):
    for i in range(0, len(prompts), ppo_config.batch_size):
        batch_prompts = prompts[i:i+ppo_config.batch_size]
        
        # generate responses (use the trainer.generate or model.generate)
        responses = []
        for p in batch_prompts:
            in_ids = tokenizer("summarize: " + p, return_tensors="pt").input_ids.to(policy_model.device)
            out = policy_model.generate(in_ids, max_new_tokens=80, do_sample=True, top_p=0.95, temperature=0.8)
            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            responses.append(resp)
            
        rewards = reward.score(batch_prompts, responses)
        
        # optionally normalize rewards
        import numpy as np
        rewards = np.array(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # run PPO step
        stats = ppo_trainer.step(batch_prompts, responses, rewards.tolist())
        print(stats)
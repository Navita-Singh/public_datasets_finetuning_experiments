# create_pairwise_dataset.py
import json
import random
from pathlib import Path

# Example prompts (replace with domain-specific
# prompts or collect from logs)
prompts = [
    "How do I reset my password?",
    "Explain PPO in simple terms.",
    "How can I improve my team's productivity?",
    "Summarize the article about climate policy.",
]

def generate_candidate_responses(prompt):
    # In production: produce with your SFT model or
    # collect human responses.
    # Here we simulate two variants: 'better' and
    # 'worse' by simple heuristics.
    base = prompt
    better = f"{base} — A clear step-by-step answer that is helpful and polite."
    worse = f"{base} — A brief, less useful reply."
    
    # add slight randomness
    if random.random() < 0.2:
        better += " (extra tip included)"
    
    return better, worse

def simulate_pairwise(n_samples=200):
    data = []
    for _ in range(n_samples):
        p = random.choice(prompts)
        a, b = generate_candidate_responses(p)
        
        # choose which is chosen; simulate human
        # noise (flip sometimes)
        if random.random() < 0.9:
            chosen, rejected = a, b
        else:
            chosen, rejected = b, a
            
        data.append({"prompt": p, "chosen": chosen, "rejected": rejected})
    return data

# Execution and Saving
out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

data = simulate_pairwise(500)

with open(out_dir / "pairwise_train.jsonl", "w", encoding="utf-8") as fh:
    for row in data[:450]:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

with open(out_dir / "pairwise_val.jsonl", "w", encoding="utf-8") as fh:
    for row in data[450:]:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Wrote pairwise_train.jsonl and pairwise_val.jsonl")


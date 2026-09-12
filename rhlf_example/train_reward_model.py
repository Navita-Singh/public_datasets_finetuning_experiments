# train_reward_model.py

import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer

class PairwiseDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        chosen = self.tokenizer(
            row["prompt"] + " " + row["chosen"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        rejected = self.tokenizer(
            row["prompt"] + " " + row["rejected"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # squeeze tensors
        return {
            "chosen_input_ids": chosen["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
        }

class RewardModel(nn.Module):
    def __init__(self, backbone="distilroberta-base", dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(hidden, 1)
        )

    def forward(self, input_ids=None, attention_mask=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last = out.last_hidden_state # (batch, seq, hidden)
        
        mask = attention_mask.unsqueeze(-1)
        summed = (last * mask).sum(1)
        denom = mask.sum(1).clamp(min=1)
        pooled = summed / denom
        
        r = self.head(pooled).squeeze(-1)
        return r # (batch,)

class PairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs are batches of tensors
        chosen_ids = inputs["chosen_input_ids"]
        chosen_mask = inputs["chosen_attention_mask"]
        rejected_ids = inputs["rejected_input_ids"]
        rejected_mask = inputs["rejected_attention_mask"]

        # move to model device if necessary handled by Trainer
        r_chosen = model(input_ids=chosen_ids, attention_mask=chosen_mask) # (batch,)
        r_rejected = model(input_ids=rejected_ids, attention_mask=rejected_mask)
        
        diff = r_chosen - r_rejected
        loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()
        
        return (loss, None) if return_outputs else loss

def main():
    backbone = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    
    train_ds = PairwiseDataset("data/pairwise_train.jsonl", tokenizer)
    val_ds = PairwiseDataset("data/pairwise_val.jsonl", tokenizer)
    
    model = RewardModel(backbone=backbone)
    
    args = TrainingArguments(
        output_dir="./reward_model",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = PairwiseTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )
    
    trainer.train()
    trainer.save_model("./reward_model_final")
    tokenizer.save_pretrained("./reward_model_final")

if __name__ == "__main__":
    main()
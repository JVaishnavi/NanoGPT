#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:43:41 2023

@author: vaishnavijanakiraman
"""


import torch
from config import get_config
from model import BigramLanguageModel
from load_data import load_data

config = get_config()
device = config["device"]
train_data, val_data, vocab_size, decode = load_data()

model = BigramLanguageModel(vocab_size)
m = model.to(device)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack([data[i:i+config["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+1+config["block_size"]] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_losses():
    out = {}
    model.eval() #Setting model to eval phase
    for split in ["train", "val"]:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X, y = get_batch(split)
            logit, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() #Resetting model to training phase
    return out

def main():
    
    
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    
    
    for iter in range(config["max_iters"]):
        
        if iter%config["eval_interval"]==0:
            losses = estimate_losses()
            print(f"step {iter}: train loss: {losses['train']:.4f}, val losses: {losses['val']:.4f}")
            
        xb, yb = get_batch('train')
    
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print("Generate from the model")
    idx = torch.zeros((1, 1), dtype=torch.long, device = device)
    idx_preds = m.generate(idx, max_new_tokens=100)[0]
    print(decode(idx_preds.tolist()))
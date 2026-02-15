#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 22:30:16 2023

@author: vaishnavijanakiraman
"""


import torch

def load_data():
    with open("input.txt", "r", encoding = "utf-8") as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, vocab_size, decode
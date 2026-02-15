#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:57:17 2023

@author: vaishnavijanakiraman
"""

import torch

def get_config():
    config = {
        "batch_size" : 16,
        "block_size" : 32,
        "max_iters" : 5000,
        "learning_rate" : 1e-3,
        "eval_interval" : 100,
        "eval_iters" : 200,
        "n_embd" : 64,
        "n_layer" : 8,
        "n_head" : 8,
        "dropout" : 0.1,
        "device" :'cuda' if torch.cuda.is_available() else 'cpu'

        }

    return config
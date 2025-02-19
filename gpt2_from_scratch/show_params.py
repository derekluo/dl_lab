#!/usr/bin/env python3

import torch
from model import Model

model = Model(max_token_value=8193)

state_dict = torch.load("models/model-scifi.pth")
model.load_state_dict(state_dict)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params} \n")
print("-" * 80 + "\n")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.shape}")

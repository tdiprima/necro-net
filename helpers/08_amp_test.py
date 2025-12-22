# ⚡️ Can your system handle AMP right now?
import torch
from torch.cuda.amp import autocast, GradScaler

print("CUDA:", torch.cuda.is_available())
scaler = GradScaler()

x = torch.randn(32, 1024, device="cuda")
w = torch.randn(1024, 1024, device="cuda")

with autocast():
    y = x @ w

print("AMP test passed")

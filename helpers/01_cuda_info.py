# 🧠 Do you actually have a GPU PyTorch can use?
# nvidia-smi

import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))

# AMP only works on GPUs
# Best results if compute capability ≥ 7.0 (Volta+)

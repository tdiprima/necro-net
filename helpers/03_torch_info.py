# 🧪 Is AMP enabled in your PyTorch build?
# CUDA version printed
# AMP available: True
import torch

print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("AMP available:", hasattr(torch.cuda, "amp"))

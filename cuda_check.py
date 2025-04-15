import bitsandbytes as bnb
import torch

print("Bitsandbytes version:", bnb.__version__)
print("CUDA available:", torch.cuda.is_available())

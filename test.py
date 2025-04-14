import bitsandbytes as bnb
import torch

print("Bitsandbytes version:", bnb.__version__)
print("CUDA available:", torch.cuda.is_available())

# read hf_token from hf_token.txt
with open("hf_token.txt", "r") as f:
    hf_token = f.read().strip()

print("HF token:", hf_token)


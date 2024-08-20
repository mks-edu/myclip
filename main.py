import torch

print("Welcome the MyCLIP project.")

device = "cuda" if torch.cuda.is_available() else "cpu"

print('Use device:', device)
import torch

tensor = torch.randn(5, 10, 10)

print(tensor[0, :, :].shape)
print(tensor[0].shape)
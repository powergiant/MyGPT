import torch
# # x = torch.triu(torch.full((3, 3), float('inf')), diagonal = 1)
# # print(x[None, :, :].size())

# from model import GPTConfig

# config = GPTConfig(3, 100, 100, 10, 3, 0.2)

# # match config:
# #     case GPTConfig(x):
# #         print(x)

# print(config.__dict__)

torch.manual_seed(42)
x = torch.randn(3, 3)
y = torch.randn(3, 3)
print(x, y)
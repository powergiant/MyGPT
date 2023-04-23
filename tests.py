# import torch
# x = torch.triu(torch.full((3, 3), float('inf')), diagonal = 1)
# print(x[None, :, :].size())

from model import GPTConfig

config = GPTConfig(3, 100, 100, 10, 3)
list = [1, 2, 3]
match config:
    case GPTConfig(x, _):
        print(x)

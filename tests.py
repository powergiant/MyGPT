import torch
from torch import nn


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(3, 10)
        self.layer_2 = nn.Linear(10, 10)
        self.layer_3 = nn.Linear(10, 10)
        self.layer_4 = nn.Linear(10, 10)
        self.layer_5 = nn.Linear(10, 1)

def main(rank: int):
    torch.manual_seed(1353 + rank*8)
    x = torch.rand(3,3)
    if rank == 0:
        print(x)

if __name__ == '__main__':
    torch.multiprocessing.spawn(main, nprocs=3)
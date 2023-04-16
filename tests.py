import model
import torch
from enum import Enum
# from utils.tensor_typing import TensorNamed
# from typing import TypeVar

# # test of typing

# batch = TypeVar('batch')
# block_size = TypeVar('block_size')
# embd_dim = TypeVar('embd_dim')

# x: TensorNamed[batch, block_size, embd_dim] = torch.rand(2, 5, 7)
# print(type(x))

# test of Config transform 

import model
config_type = model.ConfigType.BlockConfig

# test of gelu

import torch
import numpy
from matplotlib import pyplot
import math

x = torch.linspace(-10, 10, 100)
y = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

pyplot.plot(x, y)
pyplot.show()


# test of attention



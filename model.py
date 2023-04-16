import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import TodoError
from dataclasses import dataclass
import math
from enum import Enum
from typing import Union


class ConfigType(Enum):
    SelfAttentionConfig = 1
    FeedForwardConfig = 2
    BlockConfig = 3
    GPTConfig = 4


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    def __init__(self, ndim, weight = None, bias = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))
        self.epsilon = 1e-5
        self.ndim = ndim

    def forward(self, vec):
        return F.layer_norm(vec, self.ndim, self.weight, self.bias, self.epsilon)

@dataclass
class SelfAttentionConfig:
    n_head: int
    n_embd: int
    n_blocksize: int
    if_bias: bool = False
    if_flash: bool = False

class SelfAttention(nn.Module):
    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        assert self.n_embd % self.n_head == 0

        self.n_blocksize = config.n_blocksize
        self.if_bias = config.if_bias
        self.if_flash = config.if_flash

        self.attention_qkv = nn.Linear(self.n_embd, self.n_embd*3, bias=self.if_bias)
        self.mask = torch.tril(torch.full((self.n_blocksize, self.n_blocksize), float('inf')))

    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        # input.size() = (batch, block_size, embd_dim)
        assert len(input.size()) == 3
        B, T, C = input.size() # batch size, sequence length (n_blocksize), embedding dimensionality (n_embd)
        assert T == self.n_blocksize and C == self.n_embd
        q, k, v = self.attention_qkv(input).split(self.n_embd, dim = 2) 
        # self.attention_qkv(input).size() = (batch, block_size, embd_dim*3)
        # q.size = k.size = v.size = (batch, block_size, embd_dim)
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        # q.size = (batch, n_head, block_size, embd_dim/n_head)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        # k.size = (batch, n_head, block_size, embd_dim/n_head)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        # v.size = (batch, n_head, block_size, embd_dim/n_head)

        if self.if_flash:
            # efficient attention using Flash Attention CUDA kernels
            result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else:
            # manual implementation of attention
            att = q@(k.transpose(-1,-2))/math.sqrt(C//self.n_head)
            # k.transpose().size() = (batch, n_head, embd_dim/n_head, block_size)
            # att.size() = (batch, n_head, block_size, block_size)
            att = att + self.mask.unsqueeze(0).unsqueeze(0) # replace the upper triangle part of att by infinity
            att = F.softmax(att, dim = -1)
            result = att@v
            # result.size() = (batch, n_head, block_size, embd_dim/n_head)
        result = result.transpose(1,2).view(B, T, C)
        # result.transpose(1,2).size() = (batch, block_size, n_head, embd_dim/n_head)
        # result.size() = (batch, block_size, embd_dim)
        return result

# No comment version
# class SelfAttention(nn.Module):
#     def __init__(self, config: SelfAttentionConfig) -> None:
#         super().__init__()
#         self.n_embd = config.n_embd
#         self.n_blocksize = config.n_blocksize
#         self.n_head = config.n_head
#         assert self.n_embd//self.n_head ==0
#         self.if_bias = config.if_bias
#         self.attention_qkv = nn.Linear(self.n_embd, self.n_embd*3, bias = self.if_bias)
#         self.if_flash = config.if_flash
#         self.mask = torch.tril(torch.full((self.n_blocksize, self.n_blocksize), float('inf')))
        
#     def forward(self, input):
#         B, T, C = input.size()
#         assert T == self.n_blocksize and C == self.n_embd
#         q, k, v = self.attention_qkv(input).split(self.n_embd, dim = 2)
#         q: torch.Tensor
#         k: torch.Tensor
#         v: torch.Tensor
#         q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
#         k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
#         v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)

#         if self.if_flash:
#             result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
#         else:
#             att = q@(k.transpose(-1,-2))/math.sqrt(C//self.n_head)
#             att = att + self.mask.unsqueeze(0).unsqueeze(0)
#             att = F.softmax(att, dim = -1)
#             result = att @ v
#         result = result.transpose(1, 2).view(B, T, C)
#         return result


@dataclass
class FeedForwardConfig:
    n_embd: int
    if_bias: bool = False

class FeedForward(nn.Module):
    def __init__(self, config: FeedForwardConfig) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.if_bias = config.if_bias
        self.layer_1 = nn.Linear(self.n_embd, 4*self.n_embd, bias = self.if_bias) 
        self.layer_2 = nn.Linear(4*self.n_embd, self.n_embd, bias = self.if_bias) 

    def forward(self, input: torch.Tensor):
        assert input.size(-1) == self.n_embd
        # input.size() = (batch, block_size, embd_dim)
        # forward is position-wise and only acts on the embedding dimension
        result = self.layer_1(input) 
        # result.size() = (batch, block_size, 4*embd_dim)
        result = new_gelu(result)
        result = self.layer_2(result) 
        return result


@dataclass
class BlockConfig:
    n_head: int
    n_embd: int
    n_blocksize: int
    if_bias: bool = False
    if_flash: bool = False

class Block(nn.module):
    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        assert self.n_embd % self.n_head == 0

        self.n_blocksize = config.n_blocksize
        self.if_bias = config.if_bias
        self.if_flash = config.if_flash
        self.att = SelfAttention(ConfigTransform(config, ConfigType.SelfAttentionConfig))
        self.ffn = FeedForward(ConfigTransform(config, ConfigType.FeedForwardConfig))
        self.ln = LayerNorm(config.n_embd, config.if_bias) # if layer normalization has weights, you need include two layer norms

    def forward(self, input: torch.tensor) -> torch.tensor:
        # input.size() = (batch, block_size, embd_dim)
        result = input + self.att(self.ln(input))
        result = result + self.ffn(self.ln(result))
        return result


@dataclass
class GPTConfig:
    def __init__(self):
        raise TodoError
    # Transformer config
    n_head: int
    n_embd: int
    n_blocksize: int

class GPT(nn.Module):
    pass


def ConfigTransform(config: Union[SelfAttentionConfig, FeedForwardConfig, BlockConfig, GPTConfig], objective_config_type: ConfigType) -> Union[SelfAttentionConfig, FeedForwardConfig, BlockConfig , GPTConfig]:
    match objective_config_type:
        case ConfigType.SelfAttentionConfig:
            match config:
                case SelfAttentionConfig():
                    return config
                case FeedForwardConfig():
                    raise TypeError('Can convert FeedForwardConfig to SelfAttentionConfig')
                case BlockConfig(n_head = n_head, n_embd = n_embd, n_blocksize = n_blocksize, if_bias = if_bias, if_flash = if_flash):
                    return SelfAttention(n_head = n_head, n_embd = n_embd, n_blocksize = n_blocksize, if_bias = if_bias, if_flash = if_flash)
                case GPTConfig():
                    raise NotImplementedError
        case ConfigType.FeedForwardConfig:
            match config:
                case SelfAttentionConfig():
                    raise NotImplementedError
                case FeedForwardConfig():
                    raise NotImplementedError
                case BlockConfig(n_head = n_head, n_embd = n_embd, n_blocksize = n_blocksize, if_bias = if_bias, if_flash = if_flash):
                    return FeedForward(n_embd = n_embd, if_bias = if_bias)
                case GPTConfig():
                    raise NotImplementedError
        case ConfigType.BlockConfig:
            match config:
                case SelfAttentionConfig():
                    raise NotImplementedError
                case FeedForwardConfig():
                    raise NotImplementedError
                case BlockConfig():
                    raise NotImplementedError
                case GPTConfig():
                    raise NotImplementedError
        case ConfigType.GPTConfig:
            match config:
                case SelfAttentionConfig():
                    raise NotImplementedError
                case FeedForwardConfig():
                    raise NotImplementedError
                case BlockConfig():
                    raise NotImplementedError
                case GPTConfig():
                    raise NotImplementedError
                
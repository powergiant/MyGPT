import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math
from enum import Enum
from typing import Union
import inspect


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
    def __init__(self, ndim: int, weight: nn.Parameter| None = None, bias: nn.Parameter| None = None) -> None:
        super().__init__()
        self.weight = weight if weight else nn.Parameter(torch.ones(ndim)) 
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.epsilon = 1e-5
        self.ndim = ndim

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """Todo"""
        return F.layer_norm(vec, self.weight.shape, self.weight, self.bias, self.epsilon)

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

    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        """Todo"""
        # input.size() = (n_batch, n_seq, n_embd)
        assert len(input.size()) == 3
        B, T, C = input.size() # batch size, sequence length n_seq (not confused with the n_blocksize), embedding dimensionality (n_embd)
        assert C == self.n_embd
        assert T <= self.n_blocksize, f"Cannot forward sequence of length {T}, block size is only {self.n_blocksize}"
        q, k, v = self.attention_qkv(input).split(self.n_embd, dim = 2) 
        # self.attention_qkv(input).size() = (n_batch, n_blocksize, n_embd*3)
        # q.size = k.size = v.size = (n_batch, n_blocksize, n_embd)
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        # q.size = (n_batch, n_head, n_blocksize, n_embd/n_head)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        # k.size = (n_batch, n_head, n_blocksize, n_embd/n_head)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        # v.size = (n_batch, n_head, n_blocksize, n_embd/n_head)

        if self.if_flash:
            # efficient attention using Flash Attention CUDA kernels
            result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else:
            # manual implementation of attention
            att = q@(k.transpose(-1,-2))/math.sqrt(C//self.n_head)
            # k.transpose().size() = (n_batch, n_head, n_embd/n_head, n_blocksize)
            # att.size() = (n_batch, n_head, n_blocksize, n_blocksize)
            mask = torch.triu(torch.full((T, T), - float('inf')), diagonal = 1)
            att = att + mask.unsqueeze(0).unsqueeze(0) # replace the upper triangle part of att by infinity
            att = F.softmax(att, dim = -1)
            result = att@v
            # result.size() = (n_batch, n_head, n_blocksize, n_embd/n_head)
        result = result.transpose(1,2).contiguous().view(B, T, C)
        # result.transpose(1,2).size() = (n_batch, n_blocksize, n_head, n_embd/n_head)
        # result.size() = (n_batch, n_blocksize, n_embd)
        return result

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Todo"""
        assert input.size(-1) == self.n_embd
        # input.size() = (n_batch, n_blocksize, n_embd)
        # forward is position-wise and only acts on the embedding dimension
        result = self.layer_1(input) 
        # result.size() = (n_batch, n_blocksize, 4*n_embd)
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

class Block(nn.Module):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Todo"""
        # input.size() = (n_batch, n_blocksize, n_embd)
        result = input + self.att(self.ln(input))
        result = result + self.ffn(self.ln(result))
        return result


@dataclass
class GPTConfig:
    n_head: int
    n_embd: int
    n_blocksize: int
    n_vocabsize: int
    n_layers: int
    if_bias: bool = False
    if_flash: bool = False

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_blocksize = config.n_blocksize
        self.n_layers = config.n_layers
        self.n_vocabsize = config.n_vocabsize
        self.n_embd = config.n_embd
        self.if_bias = config.if_bias

        self.wte = nn.Embedding(self.n_vocabsize, self.n_embd)
        self.wpe = nn.Embedding(self.n_blocksize, self.n_embd)
        self.blocks = nn.ModuleList([Block(ConfigTransform(config, ConfigType.BlockConfig)) for _ in range(self.n_layers)])
        self.ln_f = LayerNorm(self.n_embd, bias = self.if_bias)
        self.l_out = nn.Linear(self.n_embd, self.n_vocabsize, bias=False)
        # self.layers = nn.ModuleDict(dict(wte = self.wte, wpe = self.wpe, blocks = self.blocks, ln_f = self.ln_f, l_out = self.l_out))

        # weights tying https://paperswithcode.com/method/weight-tying
        self.wte.weight = self.l_out.weight

        self.apply(self._init_module)

    def _init_module(self, module: nn.Module):
        match module:
            case nn.Linear():
                torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            case nn.Embedding():
                torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor|None = None) -> torch.Tensor|tuple[torch.Tensor, torch.Tensor]:
        """Todo"""
        # An example input of input is torch.tensor([[11, 23],[5, 8]]), the integers represent words or chars
        # The first dimension is the batch, the second dimension is the length of each sequence
        # input.size() = target.size() = (n_batch, n_seq)
        device = input.device
        assert len(input.size()) == 2
        n_batch, n_seq = input.size() # batch size n_batch, sequence length n_seq (not confused with the n_blocksize)
        assert n_seq <= self.n_blocksize, f"Cannot forward sequence of length {n_seq}, block size is only {self.n_blocksize}"
        pos = torch.arange(0, n_seq, dtype = torch.long, device=device).unsqueeze(0)
        # pos.size() = (1, n_seq)

        tok_embd = self.wte(input)
        # tok_embd.size() = (n_batch, n_seq, n_embd)
        pos_embd = self.wpe(pos)
        # pos_embd.size() = (1, n_seq, n_embd)
        embd = tok_embd + pos_embd
        # embd.size() = (n_batch, n_seq, n_embd)

        for block in self.blocks:
            embd = block(embd)
        embd = self.ln_f(embd)

        match target:
            case torch.Tensor():
                logits = self.l_out(embd)
                logits: torch.Tensor
                # embd.size() = (n_batch, n_seq, n_embd)
                # logits.size() = (n_batch, n_seq, n_vocabsize)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index = -1)
                return logits, loss
            case _:
                logits = self.l_out(embd[:,[-1],:]) # embd[:,[-1],:] is the last word in the final layer and logits is the probability of it. 
                # embd[:,[-1],:].size() = (n_batch, 1, n_embd)
                # logits.size() = (n_batch, 1, n_vocabsize)
                return logits

    @torch.no_grad()
    def generate(self, input: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, topk: int| None = None):
        # input.size() = (n_batch, n_seq)
        output = input
        for _ in range(max_new_tokens):
            input_trun = output if output.size(-1) <=self.n_blocksize else output[...,-self.n_blocksize:]
            # input_trun.size() = (n_batch, n_seq)
            logits = self(input_trun)
            logits: torch.Tensor
            # logits.size() = (n_batch, 1, n_vocabsize)
            logits = logits[:,-1,:]/temperature
            # logits.size() = (n_batch, n_vocabsize)

            if topk is not None:
                topk, _ = torch.topk(logits, min(topk, logits.size(-1)))
                # topk.size() = (n_batch, k)
                logits[logits < topk[:, [-1]]] = - float('inf')
                # logits.size() = (n_batch, n_vocabsize)
            
            probs = F.softmax(logits)
            # probs.size() = (n_batch, n_vocabsize)
            word_next = torch.multinomial(probs, num_samples = 1)
            # word_next.size() = (n_batch, 1)
            output = torch.cat((output, word_next), dim = -1)

        return output
    
    def config_optimizer(self, learning_rate: torch.Tensor, weight_decay: torch.Tensor, betas: torch.Tensor, device_type: torch.device):
        decay = set()
        no_decay = set()
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if param_name.endswith('bias'):
                    no_decay.add(full_param_name)
                elif param_name.endswith('weight'):
                    match module:
                        case nn.Linear():
                            decay.add(full_param_name)
                        case LayerNorm()|nn.Embedding():
                            no_decay.add(full_param_name)

        decay.remove('l_out.weight')

        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [{'params': [param_dict[param_name] for param_name in sorted(list(decay))], 'weight_decay': weight_decay},
                         {'params': [param_dict[param_name] for param_name in sorted(list(no_decay))], 'weight_decay': 0.0}]

        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = betas, **extra_args)

        return optimizer



def ConfigTransform(config: Union[SelfAttentionConfig, FeedForwardConfig, BlockConfig, GPTConfig], objective_config_type: ConfigType) -> Union[SelfAttentionConfig, FeedForwardConfig, BlockConfig , GPTConfig]:
    match objective_config_type:
        case ConfigType.SelfAttentionConfig:
            match config:
                case SelfAttentionConfig():
                    return config
                case FeedForwardConfig():
                    raise TypeError('Can convert FeedForwardConfig to SelfAttentionConfig')
                case BlockConfig(n_head = n_head, n_embd = n_embd, n_blocksize = n_blocksize, if_bias = if_bias, if_flash = if_flash):
                    return SelfAttentionConfig(n_head = n_head, n_embd = n_embd, n_blocksize = n_blocksize, if_bias = if_bias, if_flash = if_flash)
                case GPTConfig():
                    raise NotImplementedError
        case ConfigType.FeedForwardConfig:
            match config:
                case SelfAttentionConfig():
                    raise NotImplementedError
                case FeedForwardConfig():
                    raise NotImplementedError
                case BlockConfig(n_head = n_head, n_embd = n_embd, n_blocksize = n_blocksize, if_bias = if_bias, if_flash = if_flash):
                    return FeedForwardConfig(n_embd = n_embd, if_bias = if_bias)
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
                case GPTConfig(n_head = n_head, n_embd = n_embd, n_blocksize = n_blocksize, n_vocabsize = n_vocabsize, n_layers=n_layers,  if_bias = if_bias, if_flash = if_flash):
                    return BlockConfig(n_head = n_head, n_embd = n_embd, n_blocksize = n_blocksize, if_bias = if_bias, if_flash = if_flash)
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
                
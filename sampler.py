import torch
from dataclasses import dataclass
from model import GPT
from typing import Callable

@dataclass
class SampleConfig:
    device: torch.device
    check_point_path: str
    max_new_tokens: int
    temperature: float
    top_k: int


def sample(start: str, model: GPT, sample_config: SampleConfig, encode_func: Callable[[list[str]], list[int]], decode_func: Callable[[list[int]], list[str]]) -> str:
    start_enc = encode_func(start)

    start_tensor = torch.tensor(start_enc, dtype = torch.long, device = sample_config.device)[None, ...]

    with torch.no_grad():
        output = model.generate(start_tensor, sample_config.max_new_tokens, sample_config.temperature, sample_config.top_k)[-1, ...]
    
    output = decode_func(output.tolist())
    return output
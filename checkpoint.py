import torch
import config
from config import decode_func
from model import GPT
from transformers import GPT2Tokenizer

sample_config = config.sample_config

device = config.sample_config.device

# dict = torch.load("out/ckpt.pt")
dict = torch.load('/Users/xiaom/Downloads/ckpt.pt', map_location=device)

model_config = config.model_config
model = GPT(model_config)
model.load_state_dict(dict['model'])
model.to(device)
model.eval()

output = model.generate(torch.tensor([[0]], device = device), max_new_tokens = sample_config.max_new_tokens, temperature =  sample_config.temperature, topk = sample_config.top_k)
output = decode_func(output.tolist())
print(output)

# for name, value in dict['model'].items():
#     name: str
#     if name.endswith('ln.weight'):
#         print(name)
#         print(value.max())

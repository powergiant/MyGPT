import torch
from config import train_shakespeare_char
from model import GPT

sample_config = train_shakespeare_char.sample_config

device = train_shakespeare_char.train_config.device

dict = torch.load("out/ckpt.pt")

model_config = train_shakespeare_char.model_config
model = GPT(model_config)
model.load_state_dict(dict['model'])
model.to(device)
model.eval()

model.generate(torch.tensor([[0]], device = device), max_new_tokens = sample_config.max_new_tokens, temperature =  sample_config.temperature, topk = sample_config.top_k)

for name, value in dict['model'].items():
    name: str
    if name.endswith('ln.weight'):
        print(name)
        print(value.max())

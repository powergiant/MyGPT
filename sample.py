import pickle
import torch
import os
import config
from model import GPT
from sampler import sample, SampleConfig

sample_config = config.sample_config

device = sample_config.device

check_point = torch.load(sample_config.check_point_path, map_location = device)

model_config = config.model_config
model = GPT(model_config)
model.load_state_dict(check_point['model'])

model.to(device)
model.eval()

dataset_name = config.dataset_name
with open(os.path.join(os.path.join("data", dataset_name), "meta.pkl"), 'rb') as f:
    meta = pickle.load(f)
    
stoi = meta['stoi']
itos = meta['itos']
encode_func = lambda s: [stoi[c] for c in s] 
decode_func = lambda l: ''.join([itos[c] for c in l])

# assert decode_func(encode_func('sdgsdg')) == 'sdgsdg'

# with open(os.path.join(os.path.join("data", dataset_name), "input.txt"), 'r', encoding='utf-8') as f:
#     start = f.read()
start = '\n'

output = sample(start, model, sample_config, encode_func, decode_func)
print(output)
print('---------------')





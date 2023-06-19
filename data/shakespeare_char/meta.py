import os
import pickle

path_meta = os.path.join(os.path.dirname(__file__), 'meta.pkl')

with open(path_meta, 'rb') as f:
    meta = pickle.load(f)

meta_vocab_size = meta['vocab_size']

stoi = meta['stoi']
itos = meta['itos']
encode_func = lambda s: [stoi[c] for c in s] 
decode_func = lambda l: ''.join([itos[c] for c in l])




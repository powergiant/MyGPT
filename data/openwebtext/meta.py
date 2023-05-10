
from transformers import GPT2Tokenizer
import os

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir = os.path.dirname(__file__))

def encode_func(s: str) -> list[int]:
    return gpt_tokenizer(s)['input_ids']

def decode_func(l: list[int]) -> str:
    gpt_tokenizer.decode(l) 

meta_vocab_size = gpt_tokenizer.vocab_size

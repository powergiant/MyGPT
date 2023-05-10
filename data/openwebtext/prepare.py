import os
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np
from tqdm import tqdm
from meta import encode_func

n_worker = os.cpu_count()//2

dataset = load_dataset('openwebtext', cache_dir = os.path.dirname(__file__))

# train val split

train_and_val = dataset['train'].train_test_split(test_size = 0.0005, seed = 2357, shuffle = True)
train_and_val['val'] = train_and_val.pop('test')


def process(example): 
    ids = encode_func(example['text'])
    return {'ids': ids, 'lens': len(ids)}

train_and_val_tokenized = train_and_val.map(process, desc='tokenizing the train and val set', remove_columns = ['text'], num_proc = n_worker)

# dataset to train.bin and val.bin
for split, data in train_and_val_tokenized.items():
    arr_len = np.sum(data['lens'])
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16
    arr = np.memmap(filename, dtype, mode = 'w+', shape = (arr_len,))
    total_batch = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batch), desc=f'writing {filename}'):
        batch = data.shard(num_shards=total_batch, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()



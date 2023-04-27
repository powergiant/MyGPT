from trainer import TrainConfig
from model import GPTConfig
from torch import device
import os
import pickle
from sampler import SampleConfig

# my_device = device('cpu')
my_device = device('cuda')

# train_config = TrainConfig(learning_rate = 1e-3, weight_decay = 0.1, beta1 = 0.9, 
#                      beta2 = 0.99, device = device('cpu'), n_batch = 64, n_minibatch = 64, 
#                      it_max = 5000, it_warmup = 100, it_learning_rate_decay = 5000, 
#                      learning_rate_min = 1e-4, if_learning_rate_decay = True, grad_clip = 1.0, 
#                      log_interval = 1)
train_config = TrainConfig(learning_rate = 1e-3, weight_decay = 0.1, beta1 = 0.9, 
                     beta2 = 0.99, device = my_device, n_batch = 480, n_minibatch = 12, 
                     it_max = 5000, it_warmup = 100, it_learning_rate_decay = 5000, 
                     learning_rate_min = 1e-4, if_learning_rate_decay = True, grad_clip = 1.0, 
                     log_interval = 1, check_point_interval = 10, out_dir = 'out')



path_data = 'data'
dataset_name = 'shakespeare_char'

path = os.path.join(path_data, dataset_name)
path_meta = os.path.join(path, 'meta.pkl')

with open(path_meta, 'rb') as f:
    meta = pickle.load(f)

meta_vocab_size = meta['vocab_size']




model_config = GPTConfig(n_head = 6, n_embd = 384, n_blocksize = 256, n_vocabsize = meta_vocab_size, n_layers = 6)
# model_config = GPTConfig(n_head = 4, n_embd = 128, n_blocksize = 64, n_vocabsize = meta_vocab_size, n_layers = 4)



sample_config = SampleConfig(device = my_device, 
                             check_point_path = os.path.join(train_config.out_dir, 'ckpt.pt'),
                             max_new_tokens = 500, temperature = 0.8, top_k = 200)






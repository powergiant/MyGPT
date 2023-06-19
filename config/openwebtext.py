from trainer import TrainConfig
from trainer_ddp import DDPConfig
from model import GPTConfig
from torch import device
import os
import pickle
from sampler import SampleConfig
from data.openwebtext.meta import meta_vocab_size, encode_func, decode_func

# my_device = device('cpu')
my_device = device('cuda')
my_inference_device = device('cpu')

if_ddp = True

if if_ddp:
    # 600000
    train_config = TrainConfig(learning_rate = 6e-4, weight_decay = 0.1, beta1 = 0.9, 
                        beta2 = 0.95, device = my_device, n_batch = 64, n_minibatch = 4, 
                        it_max = 20000, it_warmup = 2000, it_learning_rate_decay = 40000, 
                        learning_rate_min = 6e-5, if_learning_rate_decay = True, grad_clip = 1.0, 
                        log_interval = 10, check_point_interval = 100, out_dir = 'out')
    # ddp_config = {'train_batch_size': train_config.n_batch, 
    #               'train_micro_batch_size_per_gpu': train_config.n_minibatch, 
    #               "fp16": {"enabled": True}, 
    #               "zero_optimization": True, 
    #               'gradient_clipping': train_config.grad_clip}
    ddp_config = DDPConfig(world_size = 8, if_amp = True)
else:
    # train_config = TrainConfig(learning_rate = 1e-3, weight_decay = 0.1, beta1 = 0.9, 
    #                     beta2 = 0.99, device = my_device, n_batch = 480, n_minibatch = 12, 
    #                     it_max = 5000, it_warmup = 100, it_learning_rate_decay = 5000, 
    #                     learning_rate_min = 1e-4, if_learning_rate_decay = True, grad_clip = 1.0, 
    #                     log_interval = 1, check_point_interval = 10, out_dir = 'out')
    train_config = TrainConfig(learning_rate = 6e-4, weight_decay = 0.1, beta1 = 0.9, 
                        beta2 = 0.95, device = my_device, n_batch = 60, n_minibatch = 60, 
                        it_max = 600000, it_warmup = 2000, it_learning_rate_decay = 600000, 
                        learning_rate_min = 6e-5, if_learning_rate_decay = True, grad_clip = 1.0, 
                        log_interval = 10, check_point_interval = 1000, out_dir = 'out')
    
path_data = 'data'
dataset_name = 'openwebtext'

path = os.path.join(path_data, dataset_name)




# model_config = GPTConfig(n_head = 6, n_embd = 384, n_blocksize = 256, n_vocabsize = meta_vocab_size, n_layers = 6, dropout_rate = 0.2, if_flash =  True)
# model_config = GPTConfig(n_head = 4, n_embd = 128, n_blocksize = 64, n_vocabsize = meta_vocab_size, n_layers = 4, dropout_rate = 0.2)
model_config = GPTConfig(n_head = 12, n_embd = 768, n_blocksize = 1024, n_vocabsize = meta_vocab_size, n_layers = 12, dropout_rate = 0.0, if_flash = True)



sample_config = SampleConfig(device = my_inference_device, 
                             check_point_path = os.path.join(train_config.out_dir, 'ckpt.pt'),
                             max_new_tokens = 500, temperature = 0.8, top_k = 200)






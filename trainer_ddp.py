from model import GPT, GPTConfig
from dataclasses import dataclass
import torch
import math
import time
from dataset import Dataset, DataType
import os
import torch.distributed
from trainer import TrainConfig
import re
import torch.amp
from enum import Enum
from contextlib import nullcontext

class AmpDtype(Enum):
    float32 = 1
    bfloat16 = 2
    float16 = 3

@dataclass 
class DDPConfig:
    world_size: int
    backend: str = 'nccl'
    if_amp: bool = False
    

# dataset: Dataset should be a variable but it is a global variable to save memory
def train(rank: int, model: GPT, train_config: TrainConfig, model_config: GPTConfig, ddp_config: DDPConfig):
    from train import dataset
    torch.manual_seed(1337 + rank * 5)
    torch.cuda.manual_seed(1337 + rank * 5)
    
    set_up_ddp(rank, ddp_config)
    model.train()
    model.to(f'cuda:{rank}')
    optimizer = model.config_optimizer(train_config.learning_rate, train_config.weight_decay, (train_config.beta1, train_config.beta2), train_config.device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    if ddp_config.if_amp:
        amp_context, scaler = set_up_amp(dtype = AmpDtype.bfloat16, device = train_config.device)
    else:
        amp_context, scaler = set_up_amp(dtype =  AmpDtype.float32, device = train_config.device)

    lr_scheduler = get_lr_scheduler(optimizer, train_config)

    loss_val_best = 1e9

    if rank == 0:
        t_last = time.time()
    it = 0
    while True:

        # add training history

        assert train_config.n_batch%(train_config.n_minibatch*ddp_config.world_size) == 0
        gradient_accumulation_steps = train_config.n_batch//train_config.n_minibatch//ddp_config.world_size

        input, target = dataset.get_batch(model_config.n_blocksize, train_config.n_batch//ddp_config.world_size, DataType.TrainData, torch.device(f'cuda:{rank}'))

        for micro_step in range(gradient_accumulation_steps):
            model.require_backward_grad_sync = (micro_step ==  gradient_accumulation_steps - 1)
            index_start = micro_step*train_config.n_minibatch
            index_end = (micro_step+1)*train_config.n_minibatch
            with amp_context:
                logits, loss = model.forward(input[index_start: index_end, ...], 
                                            target[index_start: index_end, ...])
                loss = loss/gradient_accumulation_steps
            scaler.scale(loss).backward()
        
        if train_config.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        
        lr_scheduler(it)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        it += 1

        if rank == 0:
            if it%train_config.log_interval == 0:
                t_now = time.time()
                dt = t_now - t_last
                t_last = t_now
                lossf = loss.item()*gradient_accumulation_steps
                print(f"iter {it}: loss {lossf:.4f} time {dt*1000:.2f}ms")
            if it%train_config.check_point_interval == 0: 
                loss_val = 0
                step = train_config.n_batch//train_config.n_minibatch
                for _ in range(step):
                    loss_val += get_loss_val(model, dataset, train_config.n_minibatch, train_config, model_config)/step
                print(f"save check point to {train_config.out_dir}: iter {it} loss {lossf:.4f} loss_val {loss_val:.4f}")
                if loss_val < loss_val_best:
                    loss_val_best = loss_val
                    check_point = {
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_config': model_config,
                        'iter_num': it,
                        'best_val_loss': loss_val_best,
                        'train_config': train_config,
                    }
                    if os.path.exists(train_config.out_dir):
                        torch.save(check_point, os.path.join(train_config.out_dir, 'ckpt.pt'))
                    else:
                        os.makedirs(train_config.out_dir)
                        torch.save(check_point, os.path.join(train_config.out_dir, 'ckpt.pt'))

        if it > train_config.it_max:
            break
    
    torch.distributed.destroy_process_group()


def set_up_ddp(rank: int, ddp_config: DDPConfig):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12335'
    torch.distributed.init_process_group(ddp_config.backend, rank=rank, world_size=ddp_config.world_size)



def set_up_amp(dtype: AmpDtype, device: torch.device) -> tuple[torch.amp.autocast|nullcontext, torch.cuda.amp.GradScaler]:
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype==AmpDtype.float16))
    match dtype:
        case AmpDtype.float32:
            dtype_str = torch.float32
        case AmpDtype.bfloat16:
            dtype_str = torch.bfloat16
        case AmpDtype.float16:
            dtype_str = torch.float16
    amp_context = nullcontext() if device.type == 'cpu' else torch.amp.autocast(device_type = device.type, dtype = dtype_str)
    return amp_context, scaler

def get_lr_scheduler(optimizer: torch.optim.Optimizer, train_config: TrainConfig):
    def get_lr(it: int):
        lr = train_config.learning_rate
        lr_min = train_config.learning_rate_min
        if not train_config.if_learning_rate_decay:
            return lr
        if it < train_config.it_warmup:
            return lr*it/train_config.it_warmup
        if it > train_config.it_learning_rate_decay:
            return lr_min
        decay_ratio = (it - train_config.it_warmup)/(train_config.it_learning_rate_decay - train_config.it_warmup)
        assert 0 <= decay_ratio <= 1
        coef = 0.5 + (1.0 + math.cos(math.pi * decay_ratio))

        return lr_min + coef*(lr - lr_min) 
    def lr_scheduler(it: int):
        for param in optimizer.param_groups:
            param['lr'] = get_lr(it)
    # return torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    return lr_scheduler

@torch.no_grad()
def get_loss_val(model: GPT, dataset: Dataset, n_minibatch: int, train_config: TrainConfig, model_config: GPTConfig, amp_context: torch.amp.autocast|nullcontext = nullcontext()):
    model.eval() # change to eval to handle the possibly present dropout

    input_val, target_val = dataset.get_batch(model_config.n_blocksize, n_minibatch, DataType.ValData, device = train_config.device)
    with amp_context:
        logits, loss_val = model.forward(input_val, target_val)

    model.train()
    return loss_val

# def model_dict_rename(state_dict: dict):
#     """Remove the 'module' in the name of parameters of DDP model"""
#     model_dict = dict()
#     pattern = re.compile('module.')
#     for k,v in state_dict.items():
#         if re.search("module", k):
#             model_dict[re.sub(pattern, '', k)] = v
#         else:
#             model_dict = state_dict
#     return model_dict

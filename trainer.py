from model import GPT
from dataclasses import dataclass
import torch
import math
import time
from dataset import Dataset, DataType
import os

@dataclass
class TrainConfig:
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    device: torch.device
    n_batch: int
    n_minibatch: int
    it_max: int
    it_warmup: int
    it_learning_rate_decay: int
    learning_rate_min: float
    if_learning_rate_decay: bool
    grad_clip: float|None
    log_interval: int
    check_point_interval: int
    out_dir: str
    

def train(model: GPT, train_config: TrainConfig, dataset: Dataset, model_history: list = []):
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    
    optimizer = model.config_optimizer(train_config.learning_rate, train_config.weight_decay, (train_config.beta1, train_config.beta2), train_config.device)
    model.train()
    model.to(train_config.device)
    it = 0

    t_last = time.time()
    loss_val_best = 1e9

    while True:
        lr = get_lr(it, train_config)
        for param in optimizer.param_groups:
            param['lr'] = lr


        # add training history

        assert train_config.n_batch%train_config.n_minibatch == 0
        gradient_accumulation_steps = train_config.n_batch//train_config.n_minibatch

        for _ in range(gradient_accumulation_steps):
            input, target = dataset.get_batch(model.n_blocksize, train_config.n_minibatch, DataType.TrainData, train_config.device)
            logits, loss = model.forward(input, target)
            loss = loss/gradient_accumulation_steps
            loss.backward()
        
        if train_config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        it += 1

        if it%train_config.log_interval == 0:
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            lossf = loss.item()*gradient_accumulation_steps
            print(f"iter {it}: loss {lossf:.4f} time {dt*1000:.2f}ms")
        if it%train_config.check_point_interval == 0: 
            loss_val = get_loss_val(model, dataset, train_config)
            print(f"save check point to {train_config.out_dir}: iter {it} loss {lossf:.4f} loss_val {loss_val:.4f}")
            if loss_val < loss_val_best:
                loss_val_best = loss_val
                check_point = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_config': model.config,
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

def get_lr(it: int, train_config: TrainConfig):
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

@torch.no_grad()
def get_loss_val(model: GPT, dataset: Dataset, train_config: TrainConfig):
    loss_val = 0
    gradient_accumulation_steps = train_config.n_batch//train_config.n_minibatch
    model.eval() # change to eval to handle the possibly present dropout

    for _ in range(gradient_accumulation_steps):
        input_val, target_val = dataset.get_batch(model.n_blocksize, train_config.n_minibatch, DataType.ValData, device = train_config.device)
        logits, loss = model.forward(input_val, target_val)
        loss_val += loss/gradient_accumulation_steps

    model.train()
    return loss_val

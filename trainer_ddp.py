from model import GPT
from dataclasses import dataclass
import torch
import math
import time
from dataset import Dataset, DataType
import os
import torch.distributed
import deepspeed
from trainer import TrainConfig


def train(model, train_config: TrainConfig, ddp_config: dict, dataset: Dataset, model_history: list = []):
    start_time = time.time()
    torch.distributed.init_process_group(backend='nccl')
    optimizer = model.config_optimizer(train_config.learning_rate, train_config.weight_decay, (train_config.beta1, train_config.beta2), train_config.device)
    lr_scheduler = get_lr_scheduler(optimizer, train_config)
    model_ddp, optimizer, _, lr_scheduler =  deepspeed.initialize(model = model, optimizer = optimizer, lr_scheduler = lr_scheduler, model_parameters = optimizer.param_groups, config = ddp_config)
    model_ddp.train()
    it = 0

    t_last = time.time()
    loss_val_best = 1e9

    t_now = time.time()
    t_last = t_now
    while True:

        # add training history

        input, target = dataset.get_batch(model.n_blocksize, train_config.n_batch, DataType.TrainData, train_config.device)
        input.to(model_ddp.local_rank)
        target.to(model_ddp.local_rank)
        logits, loss = model_ddp(input, target)
        model_ddp.backward(loss)
        model_ddp.step()

        it += 1

        if it%train_config.log_interval == 0:
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            lossf = loss.item()
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
    return torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

@torch.no_grad()
def get_loss_val(model, dataset: Dataset, train_config: TrainConfig):
    model.eval() # change to eval to handle the possibly present dropout

    input_val, target_val = dataset.get_batch(model.n_blocksize, train_config.n_batch, DataType.ValData, device = train_config.device)
    logits, loss_val = model.forward(input_val, target_val)

    model.train()
    return loss_val

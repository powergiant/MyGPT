from trainer import TrainConfig
import trainer, trainer_ddp
from model import GPT
from dataset import Dataset
import config

import torch

if __name__ == '__main__':
    
    train_config: TrainConfig = config.train_config
    path_data: str = config.path_data
    dataset_name: str = config.dataset_name

    dataset = Dataset(path_data = path_data, dataset_name = dataset_name)

    model_config = config.model_config
    model = GPT(model_config)

    # load history

    # print experiment config
    print("Train config:")
    print(train_config.__dict__)
    print("Model config:")
    print(model_config.__dict__)
    print("Dataset config:")
    print({"path_data": path_data, "dataset_name": dataset_name})

    # train
    if config.if_ddp:   
        torch.multiprocessing.spawn(
            trainer_ddp.train,
            args = (model, train_config, config.model_config, config.ddp_config, dataset),
            nprocs = config.ddp_config.world_size
        )
        # trainer_ddp.train(model, train_config, config.ddp_config, dataset)
    else:
        trainer.train(model, train_config, dataset)


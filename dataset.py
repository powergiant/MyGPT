import torch
import os
import numpy as np
from enum import Enum
import pickle

class DataType(Enum):
    TrainData = 0
    ValData = 1

class Dataset(object):
    def __init__(self, path_data: str, dataset_name: str) -> None:
        self.path = os.path.join(path_data, dataset_name)
        self.path_train_data = os.path.join(self.path, 'train.bin')
        self.path_val_data = os.path.join(self.path, 'val.bin')
        self.train_data = np.memmap(self.path_train_data, dtype = np.uint16, mode = 'r')
        self.val_data = np.memmap(self.path_val_data, dtype = np.uint16, mode = 'r')
 

    def get_batch(self, block_size: int, batch_size: int, type: DataType, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        match type:
            case DataType.TrainData:
                data = self.train_data 
            case DataType.ValData:
                data = self.val_data 
        index_start = torch.randint(len(data) - block_size, (batch_size,))
        input = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in index_start])
        target = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in index_start])

        # device_type = 'cuda' if 'cuda' in device.type else 'cpu'
        # if device_type == 'cuda':
        #     input = input.pin_memory().to(device, non_blocking=True)
        #     target = target.pin_memory().to(device, non_blocking=True)
        # else:
        #     input = input.to(device)
        #     target = target.to(device)
        input = input.to(device)
        target = target.to(device)

        return input, target



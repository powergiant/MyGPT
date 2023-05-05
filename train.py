from trainer import train, TrainConfig
from model import GPT
from dataset import Dataset
from config import train_shakespeare_char

train_config: TrainConfig = train_shakespeare_char.train_config
path_data: str = train_shakespeare_char.path_data
dataset_name: str = train_shakespeare_char.dataset_name


dataset = Dataset(path_data = path_data, dataset_name = dataset_name)

model_config = train_shakespeare_char.model_config
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
train(model, train_config, dataset)


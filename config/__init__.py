from enum import Enum

class ConfigName(Enum):
    train_shakespeare_char = 1
    openwebtext = 2
    wikipedia = 3

config_name = ConfigName.train_shakespeare_char

match config_name:
    case ConfigName.train_shakespeare_char:
        from .train_shakespeare_char import *
    case ConfigName.openwebtext:
        from .openwebtext import *
    case ConfigName.wikipedia:
        raise NotImplementedError
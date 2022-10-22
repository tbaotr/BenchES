from .optimizers import BasicGD, SGD, Adam
from .policies import LinearPolicy 
from .buffer import GradBuffer
from .logger import CSVLogger


def get_optim(params):

    if params['optim'] == "bgd":
        return BasicGD(params)
    
    elif params['optim'] == "sgd":
        return SGD(params)
    
    elif params['optim'] == "Adam":
        return Adam(params)


def get_policy(params):

    if params['policy'] == "linear":
        return LinearPolicy(params)


__all__ = [
    "get_optim"
    "get_policy",
    "GradBuffer",
    "CSVLogger",
]

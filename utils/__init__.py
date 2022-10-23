from .optimizers import BasicGD, SGD, Adam
from .policies import LinearPolicy 
from .obs_norm import MeanStdNorm, NoNorm
from .fit_norm import divide_std, compute_z_norm, compute_centered_ranks
from .buffer import GradBuffer
from .logger import CSVLogger


def get_optim(params):

    if params['optim'] == 'bgd':
        return BasicGD(params)
    
    elif params['optim'] == 'sgd':
        return SGD(params)
    
    elif params['optim'] == 'adam':
        return Adam(params)


def get_policy(params):

    if params['policy'] == 'linear':
        return LinearPolicy(params)


def get_obs_norm(params):

    if params['obs_norm'] == 'meanstd':
        return MeanStdNorm(params)
    elif params['obs_norm'] == 'no':
        return NoNorm(params)


def get_fit_norm(params):

    if params['fit_norm'] == 'div_std':
        return divide_std
    elif params['fit_norm'] == 'z_norm':
        return compute_z_norm
    elif params['fit_norm'] == 'rank':
        return compute_centered_ranks
    elif params['fit_norm'] == 'no':
        return lambda x : x


__all__ = [
    "get_optim"
    "get_policy",
    "get_obs_norm",
    "get_fit_norm",
    "GradBuffer",
    "CSVLogger",
]

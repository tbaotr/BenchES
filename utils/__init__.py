from .optimizers import BasicGD, SGD, Adam
from .policies import LinearPolicy, ToeplitzPolicy
from .obs_norm import MeanStdFilter, NoFilter
from .act_norm import clip
from .fit_norm import divide_std, compute_z_score, compute_centered_ranks
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
    elif params['policy'] == 'toeplitz':
        return ToeplitzPolicy(params)


def get_obs_norm(params):

    if params['obs_norm'] == 'meanstd':
        return MeanStdFilter(params['ob_dim'])
    elif params['obs_norm'] == 'no':
        return NoFilter(params['ob_dim'])


def get_act_norm(params):
    
    if params['act_norm'] == 'clip':
        return clip
    elif params['act_norm'] == 'no':
        return lambda action, low_bound, up_bound: action


def get_fit_norm(params):

    if params['fit_norm'] == 'div_std':
        return divide_std
    elif params['fit_norm'] == 'z_score':
        return compute_z_score
    elif params['fit_norm'] == 'rank':
        return compute_centered_ranks
    elif params['fit_norm'] == 'no':
        return lambda x : x

__all__ = [
    "get_optim"
    "get_policy",
    "get_obs_norm",
    "get_act_norm",
    "get_fit_norm",
    "GradBuffer",
    "CSVLogger",
]

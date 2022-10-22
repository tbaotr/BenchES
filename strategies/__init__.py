from .es import SimpleES
from .ges import GuidedES

def get_strategy(params):

    if params['stg_name'] == 'es':
        return SimpleES(params)
    elif params['stg_name'] == 'ges':
        return GuidedES(params)

__all__ = [
    "get_strategy",
]

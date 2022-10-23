from .es import SimpleES
from .ges import GuidedES
from .pes import PersistentES
from .pges import GuidedPersistentES

def get_strategy(params):

    if params['stg_name'] == 'es':
        return SimpleES(params)
    elif params['stg_name'] == 'ges':
        return GuidedES(params)
    elif params['stg_name'] == 'pes':
        return PersistentES(params)
    elif params['stg_name'] == 'pges':
        return GuidedPersistentES(params)

__all__ = [
    "get_strategy",
]

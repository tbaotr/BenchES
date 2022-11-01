from .es import SimpleES
from .ges import GuidedES
from .asebo import ASEBO
from .pes import PersistentES
from .pges import GuidedPersistentES
from .pasebo import PASEBO

def get_strategy(params):

    if params['stg_name'] == 'es':
        return SimpleES(params)
    elif params['stg_name'] == 'ges':
        return GuidedES(params)
    elif params['stg_name'] == 'asebo':
        return ASEBO(params)
    elif params['stg_name'] == 'pes':
        return PersistentES(params)
    elif params['stg_name'] == 'pges':
        return GuidedPersistentES(params)
    elif params['stg_name'] == 'pasebo':
        return PASEBO(params)

__all__ = [
    "get_strategy",
]

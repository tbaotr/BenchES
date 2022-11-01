import numpy as np


def clip(action, low_bound, up_bound):
    return np.clip(action, low_bound, up_bound)

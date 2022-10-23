import numpy as np


def divide_std(x):
    return x / (x.std() + 1e-8)


def compute_z_norm(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def compute_centered_ranks(x):
    # Adapted from: https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    def compute_ranks(x):
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

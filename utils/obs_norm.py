# Adapted from :
# https://github.com/ray-project/ray/blob/master/python/ray/rllib/utils/filter.py
import numpy as np


class NoFilter(object):

    def __init__(self, *args):
        pass

    def __call__(self, x, update=True):
        return np.asarray(x, dtype=np.float64)

    def clear_buffer(self):
        pass

    def update(self, other, *args, **kwargs):
        pass

    def copy(self):
        return self

    def stats_increment(self):
        pass


# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):

    def __init__(self, shape=None):
        self.n = 0
        self.M = np.zeros(shape, dtype=np.float64)
        self.S = np.zeros(shape, dtype=np.float64)

    def copy(self):
        other = RunningStat()
        other.n = self.n
        other.M = np.copy(self.M)
        other.S = np.copy(self.S)
        return other

    def push(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        if self.n == 1:
            self.M = x
        else:
            delta = x - self.M
            self.M = self.M + delta / self.n
            self.S = self.S + delta * delta * (self.n - 1) / self.n
            
    def update(self, other):
        n1 = self.n
        n2 = other.n
        n = n1 + n2
        delta = self.M - other.M
        delta2 = delta * delta
        M = (n1 * self.M + n2 * other.M) / n
        S = self.S + other.S + delta2 * n1 * n2 / n
        self.n = n
        self.M = M
        self.S = S

    @property
    def mean(self):
        return self.M

    @property
    def var(self):
        return self.S / (self.n - 1) if self.n > 1 else np.square(self.M)

    @property
    def std(self):
        return np.sqrt(self.var)


class MeanStdFilter(object):

    def __init__(self, shape, demean=True, destd=True):
        self.shape = shape
        self.demean = demean
        self.destd = destd
        self.rs = RunningStat(shape)
        self.buffer = RunningStat(shape)

        self.mean = np.zeros(shape, dtype=np.float64)
        self.std = np.ones(shape, dtype=np.float64)

    def __call__(self, x, update=True):
        x = np.asarray(x, dtype=np.float64)
        if update:
            self.rs.push(x)
            self.buffer.push(x)
        if self.demean:
            x = x - self.mean
        if self.destd:
            x = x / (self.std + 1e-8)
        return x

    def clear_buffer(self):
        self.buffer = RunningStat(self.shape)

    def update(self, other):
        self.rs.update(other.buffer)

    def copy(self):
        other = MeanStdFilter(self.shape)
        other.demean = self.demean
        other.destd = self.destd
        other.rs = self.rs.copy()
        other.buffer = self.buffer.copy()
        return other
        
    def stats_increment(self):
        self.mean = self.rs.mean
        self.std = self.rs.std

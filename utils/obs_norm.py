import numpy as np


class MeanStdNorm(object):
    def __init__(self, params):
        self.params = params
        self.reset()

    def reset(self):
        self.num_data = 0

    def push(self, x):
        self.num_data += 1

        if self.num_data == 1:
            self.oldM = np.array(x)
            self.newM = np.array(x)
            self.oldS = np.zeros((self.params['ob_dim'],))
        else:
            self.newM = self.oldM + (x - self.oldM) / self.num_data
            self.newS = self.oldS + (x - self.oldM) * (x - self.newM)
            self.oldM = np.array(self.newM)
            self.oldS = np.array(self.newS)

    def get(self, x, update=True):
        x = np.array(x)

        if update:
            self.push(x)

        x = (x - self.mean) / (self.std + 1e-8)
        return x

    @property
    def mean(self):
        if self.num_data == 0:
            return np.zeros((self.params['ob_dim']))
        else:
            return self.newM
 
    @property
    def variance(self):
        if self.num_data <= 1:
            return np.zeros((self.params['ob_dim']))
        else:
            return self.newS / (self.num_data - 1)
    
    @property
    def std(self):
        return np.sqrt(self.variance)


class NoNorm(object):
    def __init__(self, params):
        pass
    
    def reset(self):
        pass

    def get(self, x, update=True):
        return x

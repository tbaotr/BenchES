import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params
        self.gen_counter = 0

    def update(self, x, dx):
        self.gen_counter += 1
        step = self.compute_step(dx)
        return x + step

    def compute_step(self, dx):
        raise NotImplementedError
        

class BasicGD(Optimizer):
    def __init__(self, params):
        Optimizer.__init__(self, params)

    def compute_step(self, dx):
        step = -self.params['lrate'] * dx
        return step


class SGD(Optimizer):
    def __init__(self, params):
        Optimizer.__init__(self, params)
        self.v = np.zeros((params['full_dims']))

    def compute_step(self, dx, momentum=0.9):
        self.v = (1. - momentum) * dx + momentum * self.v
        step = -self.params['lrate'] * self.v
        return step


class Adam(Optimizer):
    def __init__(self, params):
        Optimizer.__init__(self, params)
        self.m = np.zeros((params['full_dims']))
        self.v = np.zeros((params['full_dims']))

    def compute_step(self, dx, eps=1e-8, beta1=0.9, beta2=0.999):
        self.m = beta1 * self.m + (1 - beta1) * dx
        self.v = beta2 * self.v + (1 - beta2) * dx * dx
        mt = self.m / (1 - beta1 ** self.gen_counter)
        vt = self.v / (1 - beta2 ** self.gen_counter)
        step = -self.params['lrate'] * mt / (np.sqrt(vt) + eps)
        return step

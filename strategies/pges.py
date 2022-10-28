import numpy as np
from utils import get_optim, get_policy, get_fit_norm, GradBuffer

 
class GuidedPersistentES(object):

    def __init__(self, params):
    
        self.params = params
        self.rng = np.random.RandomState(params['seed'])

    def initialize(self):
    
        self.mu = get_policy(self.params).get_weight()
        self.params['full_dims'] = len(self.mu)
        self.optimizer = get_optim(self.params)
        self.fit_norm = get_fit_norm(self.params)
        self.grad_buffer = GradBuffer(self.params['sub_dims'], self.params['full_dims'])
        self.eps_accums = np.zeros((self.params['pop_size'], self.params['full_dims']))

    def ask(self):
    
        if self.grad_buffer.size < self.grad_buffer.max_size:
            a = np.sqrt(self.params['alpha'] / self.params['full_dims'])
            eps = a * self.rng.randn(self.params['pop_size']//2, self.params['full_dims'])
        else:
            U, _ = np.linalg.qr(self.grad_buffer.grads.T)
            a = np.sqrt(self.params['alpha'] / self.params['full_dims'])
            c = np.sqrt((1. - self.params['alpha']) / self.params['sub_dims'])
            eps1 = a * self.rng.randn(self.params['pop_size']//2, self.params['full_dims'])
            eps2 = c * self.rng.randn(self.params['pop_size']//2, self.params['sub_dims']) @ U.T
            eps = eps1 + eps2
        self.eps = np.concatenate([eps, -eps])
        self.eps_accums += self.eps

        X = self.mu.reshape(1, self.params['full_dims']) + self.params['sigma'] * self.eps
        return X

    def tell(self, f_vals, done):
    
        f_vals = self.fit_norm(f_vals)
        grad = 1. / (self.params['pop_size'] * self.params['sigma']) * np.dot(self.eps_accums.T, f_vals)
        self.mu = self.optimizer.update(self.mu, grad)
        self.grad_buffer.add(grad)

        for idx in done:
            self.eps_accums[idx] = np.zeros((1, self.params['full_dims']))

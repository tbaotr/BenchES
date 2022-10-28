import numpy as np
from utils import get_optim, get_policy, get_fit_norm


class SimpleES(object):

    def __init__(self, params):
    
        self.params = params
        self.rng = np.random.RandomState(params['seed'])
 
    def initialize(self):
    
        self.mu = get_policy(self.params).get_weight()
        self.params['full_dims'] = len(self.mu)
        self.optimizer = get_optim(self.params)
        self.fit_norm = get_fit_norm(self.params)
 
    def ask(self):
    
        eps = self.rng.randn(self.params['pop_size']//2, self.params['full_dims'])
        self.eps = np.concatenate([eps, -eps])
        
        X = self.mu.reshape(1, self.params['full_dims']) + self.params['sigma'] * self.eps
        return X
 
    def tell(self, f_vals, done):
    
        f_vals = self.fit_norm(f_vals)
        grad = 1. / (self.params['pop_size'] * self.params['sigma']) * np.dot(self.eps.T, f_vals)
        self.mu = self.optimizer.update(self.mu, grad)

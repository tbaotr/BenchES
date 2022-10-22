import numpy as np
from copy import copy


class LinearPolicy(object):

    def __init__(self, params):

        self.params = params
        self.rng = np.random.RandomState(params['seed'])

        self.w = self._initialize_weight(params['ob_dim'] * params['ac_dim'])
        self.W = self._build_layer(params['ob_dim'], params['ac_dim'], self.w)

        self.w_vec = copy(self.w)

    def _initialize_weight(self, dim):

        if self.params['init_weight'] == 'zero':
            w = np.zeros(dim)
        elif self.params['init_weight'] == 'uniform':
            w = self.rng.rand(dim) / np.sqrt(dim)

        return w
    
    def _build_layer(self, in_dim, out_dim, vec):

        W = vec.reshape(out_dim, in_dim)

        return W

    def get_weight(self):

        return self.w_vec

    def set_weight(self, vec):

        self.w = vec
        self.W = self._build_layer(self.params['ob_dim'], self.params['ac_dim'], self.w)

        self.w_vec = copy(self.w)

    def evaluate(self, X):
        
        X = X.reshape(X.size, 1)

        return np.dot(self.W, X)

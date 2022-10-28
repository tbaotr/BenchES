import numpy as np
from scipy.linalg import toeplitz

class LinearPolicy(object):

    def __init__(self, params):

        self.params = params
        self.rng = np.random.RandomState(params['seed'])

        self.w = self._initialize_weight(params['ob_dim'] * params['ac_dim'])
        self.W = self._build_layer(params['ob_dim'], params['ac_dim'], self.w)

        self.w_vec = np.array(self.w)

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

        self.w_vec = np.array(self.w)

    def evaluate(self, X):

        return np.dot(self.W, X)


class ToeplitzPolicy(object):

    def __init__(self, params):

        self.params = params
        self.rng = np.random.RandomState(params['seed'])

        self.w1 = self._initialize_weight(self.params['ob_dim'] + self.params['h_dim'] -1)
        self.w2 = self._initialize_weight(self.params['h_dim'] * 2 - 1)
        self.w3 = self._initialize_weight(self.params['ac_dim'] + self.params['h_dim'] - 1)
        
        self.W1 = self._build_layer(self.params['ob_dim'], self.params['h_dim'], self.w1)
        self.W2 = self._build_layer(self.params['h_dim'], self.params['h_dim'], self.w2)
        self.W3 = self._build_layer(self.params['h_dim'], self.params['ac_dim'], self.w3)
        
        self.b1 = self._initialize_weight(self.params['h_dim'])
        self.b2 = self._initialize_weight(self.params['h_dim'])
    
        self.w_vec = np.concatenate([self.w1, self.b1, self.w2, self.b2, self.w3])

    def _initialize_weight(self, dim):

        if self.params['init_weight'] == 'zero':
            w = np.zeros(dim)
        elif self.params['init_weight'] == 'uniform':
            w = self.rng.rand(dim) / np.sqrt(dim)

        return w
    
    def _build_layer(self, in_dim, out_dim, vec):

        col = vec[:out_dim]
        row = vec[(out_dim-1):]

        W = toeplitz(col, row)

        return W

    def get_weight(self):

        return self.w_vec

    def set_weight(self, vec):

        self.w_vec = vec
        
        self.w1 = vec[:len(self.w1)]
        vec = vec[len(self.w1):]

        self.b1 = vec[:len(self.b1)]
        vec = vec[len(self.b1):]
        
        self.w2 = vec[:len(self.w2)]
        vec = vec[len(self.w2):]

        self.b2 = vec[:len(self.b2)]
        vec = vec[len(self.b2):]

        self.w3 = vec
        
        self.W1 = self._build_layer(self.params['ob_dim'], self.params['h_dim'], self.w1)
        self.W2 = self._build_layer(self.params['h_dim'], self.params['h_dim'], self.w2)
        self.W3 = self._build_layer(self.params['h_dim'], self.params['ac_dim'], self.w3)

    def evaluate(self, X):

        z1 = np.tanh(np.dot(self.W1, X) + self.b1)
        z2 = np.tanh(np.dot(self.W2, z1) + self.b2)

        return(np.tanh(np.dot(self.W3, z2)))

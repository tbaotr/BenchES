import numpy as np
from utils import get_optim, get_policy, get_fit_norm

from sklearn.decomposition import PCA
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError


class ASEBO(object):

    def __init__(self, params):
    
        self.params = params
        self.rng = np.random.RandomState(params['seed'])
 
    def initialize(self):
    
        self.mu = get_policy(self.params).get_weight()
        self.params['full_dims'] = len(self.mu)
        self.optimizer = get_optim(self.params)
        self.fit_norm = get_fit_norm(self.params)
 
    def ask(self):
    
        if self.optimizer.gen_counter >= self.params['warm_up'] - 2:
            pca = PCA()
            pca_fit = pca.fit(self.G)
            var_exp = pca_fit.explained_variance_ratio_
            var_exp = np.cumsum(var_exp)
            n_samples = np.argmax(var_exp > self.params['threshold']) + 1
            if n_samples < self.params['min']:
                n_samples = self.params['min']
            U = pca_fit.components_[:n_samples]
            self.UUT = np.matmul(U.T, U)
            U_ort = pca_fit.components_[n_samples:]
            self.UUT_ort = np.matmul(U_ort.T, U_ort)
            alpha = self.params['alpha']
            if self.optimizer.gen_counter == self.params['warm_up'] - 2:
                n_samples = self.params['pop_size']//2
        else:
            self.UUT = np.zeros([self.params['full_dims'], self.params['full_dims']])
            alpha = 1
            n_samples = self.params['pop_size']//2
        
        cov = (alpha / self.params['full_dims']) * np.eye(self.params['full_dims']) + ((1. - alpha) / n_samples) * self.UUT
        cov *= self.params['sigma']
        eps = np.zeros((n_samples, self.params['full_dims']))
        try:
            l = cholesky(cov, check_finite=False, overwrite_a=True)
            for i in range(n_samples):
                try:
                    eps[i] = np.zeros(self.params['full_dims']) + l.dot(self.rng.randn(self.params['full_dims']))
                except LinAlgError:
                    eps[i] = self.rng.randn(self.params['full_dims'])
        except LinAlgError:
            for i in range(n_samples):
                eps[i] = self.rng.randn(self.params['full_dims'])  
        eps /= np.linalg.norm(eps, axis =-1)[:, np.newaxis]

        self.eps = np.concatenate([eps, -eps])
        
        self.params['pop_size'] = n_samples * 2
        
        X = self.mu.reshape(1, self.params['full_dims']) + self.eps
        return X
 
    def tell(self, f_vals, done):
    
        f_vals = self.fit_norm(f_vals)
        grad = 1. / (2 * self.params['sigma']) * np.dot(self.eps.T, f_vals)

        if self.optimizer.gen_counter >= self.params['warm_up'] - 2:
            self.params['alpha'] = np.linalg.norm(np.dot(grad, self.UUT_ort))/np.linalg.norm(np.dot(grad, self.UUT)) 

        if self.optimizer.gen_counter == 0:
            self.G = np.array(grad)
        else:
            self.G *= self.params['decay']
            self.G = np.vstack([self.G, grad])

        grad /= (np.linalg.norm(grad) / self.params['full_dims'] + 1e-8)

        self.mu = self.optimizer.update(self.mu, grad)

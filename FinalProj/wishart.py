from __future__ import division
import numpy as np
from numpy import exp
from numpy import trace
from numpy import sqrt
from numpy.linalg import det
from numpy.linalg import pinv
from numpy.linalg import cholesky
import scipy.stats as stats
import scipy

class Wishart:
    def __init__(self, V, n):
        self.V = V
        shape = V.shape
        assert shape[0]==shape[1]
        self.p = shape[0]
        self.n = float(n)
        assert type(n) is int, 'n is not an integer'
        assert type(self.p) is int, 'p is not an integer'
        assert n > self.p-1, 'n <= p-1'
        assert not False in (V>0), 'V is not > 0'
        
    def pdf(self, X):
        X_shape = X.shape
        assert X_shape[0] == X_shape[1], 'X is not a square matrix'
        assert X_shape[0] == self.p
        Z_denom = 2**(-self.n*self.p/2.0)*det(self.V)**(-self.n/2.0)*exp(scipy.special.multigammaln(self.n/2., self.p))
        return Z_denom*det(X)**((self.n-self.p-1)/2.0)*exp(-0.5*trace(pinv(self.V).dot(X)))

    def sample(self):
        chol = cholesky(self.V)
        if self.n <= 81 + self.p:
            x = np.random.randn(self.n, self.p)
        else:
            x = np.diag(sqrt(stats.chi2.rvs(self.n-np.arange(self.p))))
            x[np.triu_indices_from(x,1)] = np.random.nrandn(self.p*(self.p-1)/2)
        R = np.linalg.qr(x, 'r')
        T = scipy.linalg.solve_triangular(R.T, chol.T).T
        return np.dot(T, T.T)

    def sample_inv(self):
        chol = cholesky(self.V)
        if self.n <= 81 + self.p: #direct computation
            X = np.dot(chol, np.random.normal(size=(self.p, self.n)))
        else:
            A = np.diag(sqrt(np.random.chisquare(self.n - np.arange(0,self.p), size=self.p)))
            A[np.tri(self.p, k=-1, dtype=bool)] = np.random.normal(size=(self.p*(self.p-1)/2.))
            X = np.dot(chol, A)
        return np.dot(X, X.T)





import numpy as np
import scipy
from numpy import exp
from numpy import trace
from numpy.linalg import det
from numpy.linalg import pinv
from numpy.linalg import inv


class Wishart:
    def __init__(self, p, V, n):
	self.p = p
	self.V = V
	self.n = n
	assert n > p-1, 'n <= p-1'
	assert not False in (V>0), 'V is not > 0'
    def pdf(self, X):
	Z = 2**(-n*p/2.0)*det(V)**(-n/2.0)
	return Z*det(X)**((n-p-1)/2.0)*exp(-0.5*trace(pinv(V).dot(X)))

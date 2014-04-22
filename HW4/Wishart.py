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

    def evaluate_pdf(self, X):

	2**(-n*p/2.0)*det(V)**(-n/2.0)

	det(X)**((n-p-1)/2.0)*exp(-0.5*trace(inv(V).dot(X)))

    


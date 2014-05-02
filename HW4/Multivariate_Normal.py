import numpy as np
import scipy
from numpy import exp
from numpy import trace
from numpy import pi
from numpy.linalg import det
from numpy.linalg import pinv
from numpy.linalg import inv

class Multivariate_Normal:
    def __init__(self, k, mu, sigma):
	""" Initializes the Multivariate Normal object

	    
	"""
	self.k = k
	self.mu = mu
	self.sigma = sigma
	# input assertion statements checking for proper dimensionality
	assert self.k == mu.shape[0], "mean vector mu is not k-dimensional"
	assert self.k == sigma.shape[0], "covariance matrix sigma is not k-by-k"
	assert self.k == sigma.shape[1], "covariance matrix sigma is not k-by-k"

    def pdf(self, x):
	""" Evaluates the pdf of the instantiated Multivariate Normal object 

	"""
	Z = (2*pi)**(-self.k/2.0)*det(self.sigma)**(0.5) #normalization term
	pdf = Z*exp(-0.5*(x-self.mu).T.dot(pinv(self.sigma)).dot(x-self.mu)) 
	return pdf


'''
import numpy as np
from Multivariate_Normal import Multivariate_Normal
k = 3
mu = np.zeros(k)
sigma = np.identity(k)
mvn = Multivariate_Normal(k, mu, sigma)
x = np.array([-0.2, 0.1, 0.4])
mvn.pdf(x)
'''

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
	self.k = k
	self.mu = mu
	self.sigma = sigma

	assert self.k = mu.shape[0]
	assert.self.k = sigma.shape[0]
	assert.self.k = sigma.shape[1]

    def evaluate_pdf(x, k, mu, sigma):
		Z = (2*pi)**(-k/2.0)*det(sigma)**(0.5) #normalization term
	    pdf = Z*exp(-0.5*(x-mu).T.dot(pinv(sigma)).dot(x-mu))
		return pdf



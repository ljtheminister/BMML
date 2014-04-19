import numpy as np
import pandas as pd
import random
from numpy.linalg import norm
from numpy import exp
from numpy import inf
from numpy import sqrt
from numpy import prod
from numpy import diag
from numpy.linalg import inv
from scipy.stats import multivariate_normal

'''
pi = np.ones(K)/float(K)
GMM = GMM_EM(X,K, 


'''


class GMM_EM:
    def __init__(self, X, K, pi, max_iter):
	self.X = X
	self.N, self.p = X.shape
	self.K = K
	self.pi = pi
	self.max_iter = max_iter
	self.means = {}
	self.covs = {}
	self.assignments = np.zeros((self.N, self.K))
	self.N_K = {}
	# instantiate cluster data structures
	for i in xrange(self.K):
	    self.clusters[i] = np.zeros((self.p,self.p))
	    self.N_K[i] = 0

    def find_nearest_cluster(self, x):
	min_dist = inf
	min_cluster = inf
	for k in xrange(self.K):
	    dist = norm(self.clusters[k]-x)
	    if dist < min_dist:
		min_dist = dist	
		min_cluster = k
	return min_cluster

    def initialize_clusters(self):	
	idx = [i for i in xrange(self.N)]	 
	random.shuffle(idx)
	# randomly initialize K clusters (hard assignment)
	for i in xrange(self.K):
	    self.means[i] = self.X[idx[i],:]
	# perform K-means to complete initialization
	for i,x in enumerate(X):
	    c = self.find_nearest_cluster(x)
	    self.assignments[i,c] = 1 # hard assignment

    # M-step
    def M_step(self):
	for k in xrange(self.K):
	    # compute N_k for each cluster
	    self.N_K[k] = sum(self.assignments[:,k])
	    # re-estimate means 
	    mean_sum = np.zeros(k)
	    for i,x in enumerate(self.X):
		mean_sum += self.assignments[i,k]*x 
	    # update means
	    self.means[k] = mean_sum/float(self.N_K[k])

	for k in xrange(self,K):
	    # re-estimate covariances
	    cov_sum = np.zeros((self.p,self.p))
	    for i,x in enumerate(self.X):
		cov_sum += self.assignments[i,k]*(x - self.means[k]).dot((x - self.means[k]).transpose())
	    # update covariances	
	    self.covs[k] = cov_sum/float(self.N_K[k])
	    # update mixing proportions
	    self.pi[k] = self.N_K[k]/float(self.N)

    def E_step(self):
	for i,x in enumerate(self.X):
	    row_sum = 0.0
	    for k in xrange(self.K):		
		responsibility = self.pi[k]*multivariate_normal(x, self.means[k], self.covs[k])
		self.assignments[i,k] = responsibility
		row_sum += responsibility
	    self.assignments[i,:] /= float(row_sum)


    def multivariate_normal(self, d, mean, cov, x):
	detDiagCovMatrix = sqrt(prod(diag(cov)))
	frac = (2*np.pi)**(-d/2.0)*(1/detDiagCovMatrix)
	return frac*exp(-.5*(x-mean).transpose().dot(inv(cov)).dot(x-mean))
	    















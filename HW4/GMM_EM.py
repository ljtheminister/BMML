import numpy as np
import pandas as pd
import random
from numpy.linalg import norm

X = np.loadtxt('data.txt', delimiter=',').transpose()
n, p = X.shape
K = [2,4,6,8,10] # set of K's on which to run experiment 
N_iter = 100 # number of iterations to run Expectation Maximization


K = 2
n_range = [x for x in xrange(n)]
random.shuffle(n_range)

#random initialization of Gaussian mixture means
mu = dict()
for i in xrange(K):
    mu[i] = X[n_range[i], :]
    
# uniform initialization of cluster mixing probabilities
pi = dict()
for i in xrange(K):
    pi[i] = float(1.0/K)

# initialize counts of clusters
N_k = np.zeros(K)
# do K-means to initialize means and covariance matrices



def compute_








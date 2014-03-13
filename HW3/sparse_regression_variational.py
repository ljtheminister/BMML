import numpy as np
import pandas as pd
from numpy.random import multivariate_normal
from numpy.random import gamma
from numpy.linalg import pinv
from scipy.linalg import pinv2
from numpy import identity


for i in [x+1 for x in xrange(3)]:
    X = np.loadtxt('X_set%s.csv'%(str(i)), delimiter=',')
    y = np.loadtxt('y_set%s.csv'%(str(i)), delimiter=',')
    z = np.loadtxt('y_set%s.csv'%(str(i)), delimiter=',')  


'''
X1 = pd.read_csv('X_set1.csv', header=None)
y1 = pd.read_csv('y_set1.csv', header=None)
z1 = pd.read_csv('z_set1.csv', header=None)


X2 = pd.read_csv('X_set2.csv', header=None)
y2 = pd.read_csv('y_set2.csv', header=None)
z2 = pd.read_csv('z_set2.csv', header=None)

X3 = pd.read_csv('X_set3.csv', header=None)
y3 = pd.read_csv('y_set3.csv', header=None)
z3 = pd.read_csv('z_set3.csv', header=None)
'''

N, K = X.shape
N_iter = 1000
a0 = 1e-16
b0 = 1e-16
e0 = 1
f0 = 1

alpha = gamma(a0, b0**-1, (K,))
lamb = gamma(e0, f0**-1)

for i in xrange(N_iter):
    # q(w)
    sigma = inv(lamb*X.T.dot(X) + alpha*(identity(K)))
    mu = lamb*sigma.dot(X.T).dot(y)
    w = multivariate_normal(mu, sigma)
    # q(alpha)
    a = a0 + .5
    for k in xrange(K):
	b_k = b0 + .5*w[k]**2
	alpha[k] = gamma(a, b_k**-1)

    #q(lambda)
    e = e0 + N/2.
    f = f0 + .5*y.T.dot(y) - y.T.dot(X.dot(w)) + .5*(w.T.dot(X.T)).dot(X.dot(w))
    lamb = gamma(e, f**-1)





lamb = 1e10

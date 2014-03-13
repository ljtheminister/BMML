import numpy as np
import pandas as pd
from numpy.random import normal
from numpy.random import gamma
from numpy.linalg import pinv
from numpy import identity

X1 = pd.read_csv('X_set1.csv', header=None)
y1 = pd.read_csv('y_set1.csv', header=None)
z1 = pd.read_csv('z_set1.csv', header=None)


X2 = pd.read_csv('X_set2.csv', header=None)
y2 = pd.read_csv('y_set2.csv', header=None)
z2 = pd.read_csv('z_set2.csv', header=None)

X3 = pd.read_csv('X_set3.csv', header=None)
y3 = pd.read_csv('y_set3.csv', header=None)
z3 = pd.read_csv('z_set3.csv', header=None)

N_iter = 1000
a0 = 1e-16
b0 = 1e-16
e0 = 1
f0 = 1

alpha = gamma(a0,b0**-1, (1,K))
lamb = gamma(e0, f0**-1)

w = normal(0, (alpha**-1)*identity(K))


N, K = X.shape
for i in xrange(N_iter):

sigma = pinv(lamb*X.transpose()*X + alpha*identity(K))
mu = lamb*sigma*X.transpose()*X*y
w = normal(mu, sigma)





lamb = 1e10

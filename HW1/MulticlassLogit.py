import numpy as np
import pandas as pd
import os

def readMNISTdata(base_dir, type):
    X = []
    Y = []
    digits = range(10)
    for digit in digits:
	path = base_dir + type + str(digit) + '.csv'
	digit_data = pd.read_csv(path, header=None)
	p,n = digit_data.shape
	idx = range(n)
	for i in idx:
	    X.append(digit_data.ix[:,i].values)
	    Y.append(digit)	
	N = np.shape(X)[0]
	X = np.append(np.ones((N,1)), X, axis=1)
    return X, Y

def getPCA(base_dir):
    princomps = pd.read_csv(base_dir + 'Q.csv', header=None)
    return princomps

def update(w, X, N, K, alpha, lamb):

    gradient = np.zeros(K)
    for i in range(N):
		

    w_new = w - alpha * gradient 
    return w_new


def gradient(X, y, alpha, eps):
    # first iteration
    N, K = np.shape(X)
    idx = range(N)    
    w = np.zeros(N)
    w_new = update(w, X, N, K, alpha)

    # repeat until convergence
    while np.linalg.norm(max(np.abs(w_new - w) > eps)):
	w = w_new
	w_new = update(w, X, N, K, alpha)

    return w

def main():
    base_dir = '/Users/LJ/BMML/HW1/mnist_csv/'
    X_train, y_train = readMNISTdata(base_dir, 'train')
    X_test, y_test = readMNISTdata(base_dir, 'test')
    alpha = 1
    eps = 1e-2

    w = gradient(X_train, y_train, alpha, eps) 
    
    pred_probs = np.dot(X_test.transpose(), w)


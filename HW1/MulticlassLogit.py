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

def update_batch(w, X, y, N, K, alpha, lamb):

    I = range(N)
    J = range(K)
    gradient = np.zeros(K)

    for i in I:	
	indicator = np.zeros(K)
	C = y[i]
	indicator[C] = 1
	
	denom_sum = 0
	num = np.zeros(K)
	for j in J:
	    num[j] = np.exp(w[j]*X[i])
	    denom_sum += num[j] 

	gradient += np.dot(X[i], indicator- num/denom_sum)

    w_new = w - alpha * gradient 
    return w_new

def update_stochastic(w, K, x_i, y_i, alpha, lamb):

    J = range(K)
    indicator = np.zeros(K)
    C = y_i
    indicator[C] = 1

    denom_sum = 0
    num = np.zeros(K)
    for j in J:
	num[j] = np.exp(w[j]*X[i])
	denom_sum += num[j] 
    

def gradient_batch(X, y, alpha, eps):
    # first iteration
    N, K = np.shape(X)
    idx = range(N)    
    w = np.zeros(K)
    w_new = update(w, X, N, K, alpha)

    # repeat until convergence
    while np.linalg.norm(max(np.abs(w_new - w) > eps)):
	w = w_new
	w_new = update(w, X, N, K, alpha)

    return w

def gradient_stochastic(X, y, alpha, eps):
    N, K = np.shape(X)
    idx = range(N)
    w = np.zeros(K)
    



def confusion_matrix(w_opt, X, y):
    conf_mat = np.zeros((10,10))

    return conf_mat

def main():
    base_dir = '/Users/LJ/BMML/HW1/mnist_csv/'
    X_train, y_train = readMNISTdata(base_dir, 'train')
    X_test, y_test = readMNISTdata(base_dir, 'test')
    alpha = 1
    eps = 1e-2

    w = gradient(X_train, y_train, alpha, eps) 
    
    pred_probs = np.dot(X_test.transpose(), w)


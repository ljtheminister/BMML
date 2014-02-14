import numpy as np
import pandas as pd
import os

def read_MNISTdata(base_dir, type):
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

def get_PCA(base_dir):
    princomps = pd.read_csv(base_dir + 'Q.csv', header=None)
    return princomps

def reconstruct_image(x_i, Q, y_i):
    X = np.dot(Q, x_i[1:]) 


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
    # Indicator term
    indicator = np.zeros(K)
    C = y_i
    indicator[C] = 1
    # Fraction term
    denom_sum = 0
    num = np.zeros(K)
    J = range(K)
    for j in J:
	num[j] = np.exp(w[j]*X[i])
	denom_sum += num[j] 
    # Compute gradient
    gradient = x_i * (indicator - num/denom_sum)
    # Update weight vector
    w_new = w - alpha * gradient
    return w_new
    

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
    # convergence of w's
    return w_new

def gradient_stochastic(X, y, alpha, eps):
    N, K = np.shape(X)
    I = np.random.permutation(N) #shuffle examples
    w = np.zeros(K) #initialize weight vector
 
    for i in I:
	w_new = update_stochastic(w, K, X[i], y[i], alpha)

    while np.linalg.norm(max(np.abs(w_new - w) > eps)):
	w = w_new
	I = np.random.permutation(N) #keep shuffling
	 for i in I:
	w_new = update_stochastic(w, K, X[i], y[i], alpha)

def gradient_ascent(X, y, alpha, eps, algo_type=None)
    if algo_type == None:
	raise ValueError('Invalid Arguments Given')	
    elif algo_type == 'batch':
	return gradient_batch(X, y, alpha, eps)
    elif algo_type == 'stochastic':
	return gradient_stochastic(X, y, alpha, eps)
    elif algo_type == 'mini':
	return gradient_mini(X, y, alpha, eps)
    

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


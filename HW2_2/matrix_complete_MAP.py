import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
	
def load_data_dict(filename):
    data = dict()
    with open(filename, 'rb') as f:
	for line in f:
	    row = line.split(',')
	    user, movie, rating = int(row[0])-1, int(row[1])-1, int(row[2])
	    data[(user, movie)] = rating
    return data

def metadata(data, base_dir):
    users = list()
    movies = list()
    for user, movie in data.keys():
	users.append(user)
	movies.append(movie)
    users = sorted(list(set(users)))
    movies = sorted(list(set(movies)))
    N = len(users)
    M = int(subprocess.Popen('wc -l %s/movies.txt'%(base_dir), 
	stdout=subprocess.PIPE, shell=True).communicate()[0].split()[0])
    #M = 1682
    return users, movies, N,M

def user_movie_dictionary(data, N, M):
    user_movie_dict = dict()
    for user_id in xrange(N):
	user_movie_dict[user_id] = list()
	for movie_id in xrange(M):
	    try:
		data[(user_id, movie_id)]
		user_movie_dict[user_id].append(movie_id)
	    except:
		pass
    return user_movie_dict

def movie_user_dictionary(data, N, M):
    movie_user_dict = dict()
    for movie_id in xrange(M):
	movie_user_dict[movie_id] = list()
	for user_id in xrange(N):
	    try:
		data[(user_id, movie_id)]
		movie_user_dict[movie_id].append(user_id)
	    except:
		pass
    return movie_user_dict

def dict_to_matrix(data_dict, N, M):
    mat = np.zeros((N,M))
    for user, movie in data_dict.keys():
	mat[user, movie] = data_dict[(user, movie)]
    return mat

#initialization
def initialize_factorization(N, M, d, sigma, lamb):
    U = np.zeros((N,d))
    V = np.zeros((d,M))
    I = np.identity(d)
    mean = np.zeros(d)
    cov = np.power(float(lamb),-1)*I
    for i in xrange(N):
	U[i,:] = np.random.multivariate_normal(mean, cov)
    for j in xrange(M):
	V[:,j] = np.random.multivariate_normal(mean, cov)
    return U, V, I

# user optimization
def update_user_MAP(M_train, U, V, N, I, user_movie_dict):
    for i in xrange(N):
	# compute V_i
	V_i = V[:, user_movie_dict[i]].transpose()
	# compute m_u
	m_u = M_train[i, user_movie_dict[i]]
	# compute u_MAP
	U[i,:] = np.dot(np.linalg.inv(lamb*np.power(float(sigma),2)*I + 
	    np.dot(V_i.transpose(), V_i)), np.dot(V_i.transpose(), m_u))
    return U

# movie optimizaiton
def update_movie_MAP(M_train, U, V, M, I, movie_user_dict):
    for j in xrange(M):
	# compute U
	U_j = U[movie_user_dict[j], :].transpose()
	# compute m_v
	m_v = M_train[movie_user_dict[j], j]
	# compute v_MAP
	V[:,j] = np.dot(np.linalg.inv(lamb*np.power(float(sigma),2)*I + 
	    np.dot(U_j, U_j.transpose())), np.dot(U_j, m_v))
    return V

def predict_ratings(U, V):
    M_pred = np.dot(U,V)      
    return M_pred      

def compute_RMSE(M_pred, test):
    MSE_sum = 0
    N = len(test.keys())
    for i, j in test.keys():
	y_pred = round(M_pred[i,j])
	y = M_test[i,j]
	MSE_sum += np.power(y - y_pred, 2)
    RMSE = np.sqrt(MSE_sum/float(N)) 
    return RMSE

def log_likelihood(train, U, V, sigma, lamb, N, M):
    data_sum = 0
    u_sum = 0
    v_sum = 0
    for i, j in train.keys():
	data_sum += np.power(train[i,j] - np.dot(U[i,:].transpose(), V[:,j]), 2)
    for i in xrange(N):
	u_sum += np.dot(U[i,:].transpose(), U[i,:])
    for j in xrange(M):
	v_sum += np.dot(V[:,j].transpose(), V[:,j])
    joint_LL = np.power(sigma,-2)*-.5*data_sum - .5*lamb*u_sum - .5*lamb*v_sum
    return joint_LL

# coordinate ascent 
def coordinate_ascent(M_train, test, N, M, d, N_iterations, user_movie_dict, movie_user_dict, sigma, lamb):
    U, V, I = initialize_factorization(N, M, d, sigma, lamb)
    rmse_list = list()
    log_likelihood_list = list()
    for iter in xrange(N_iterations):
	U = update_user_MAP(M_train, U, V, N, I, user_movie_dict)
	V = update_movie_MAP(M_train, U, V, M, I, movie_user_dict)
	M_pred = predict_ratings(U, V)
	rmse = compute_RMSE(M_pred, test)
	rmse_list.append(rmse)
	joint_LL = log_likelihood(train, U, V, sigma, lamb, N, M)
	log_likelihood_list.append(joint_LL)
    return U, V, rmse_list, log_likelihood_list

def plot_RMSE(N_iterations, rmse):
    X = [x+1 for x in xrange(N_iterations)]
    plt.plot(X, rmse)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.show()

## GIBBS SAMPLING
def update_user_Gibbs(M_train, U, V, N, I, user_movie_dict):
    for i in xrange(N):
    # compute V_i
	V_i = V[:, user_movie_dict[i]].transpose()
	# compute m_u
	m_u = M_train[i, user_movie_dict[i]]
	# compute u_MAP
	mean_user = np.dot(np.linalg.inv(lamb*np.power(sigma,2)*I + 
	    np.dot(V_i.transpose(), V_i)), np.dot(V_i.transpose(), m_u))
	cov_user = np.linalg.inv(lamb*I + np.power(sigma,-2)*np.dot(V_i.transpose(), V_i))
	U[i,:]= np.random.multivariate_normal(mean_user, cov_user)
    return U

def update_movie_Gibbs(M_train, U, V, M, I, movie_user_dict):
    for j in xrange(M):
	# compute U_j
	U_j = U[movie_user_dict[j], :].transpose()
	# compute m_v
	m_v = M_train[movie_user_dict[j], j]
	# compute v_MAP
	mean_movie = np.dot(np.linalg.inv(lamb*np.power(sigma,2)*I +
	     np.dot(U_j, U_j.transpose())), np.dot(U_j, m_v))
	cov_movie= np.linalg.inv(lamb*I + np.power(sigma,-2)*np.dot(U_j, U_j.transpose()))
	V[:,j] = np.random.multivariate_normal(mean_movie, cov_movie)
    return V

def initialize_gibbs_dict(test):
    gibbs_dict = dict()
    for i,j in test.keys():
	gibbs_dict[(i,j)] = list()
    return gibbs_dict

def sample_gibbs(train, gibbs_dict, U, V):
    for i, j in train.keys():
	gibbs_dict[i,j].append(np.dot(U[i,:],V[:,j]))		
    return gibbs_dict

def compute_RMSE_Gibbs(gibbs_dict, test):
    MSE_sum = 0
    N = len(test.keys())
    for i, j in test.keys():
	y_pred = round(np.mean(gibbs_dict[i,j]))
	y = M_test[i,j]
	MSE_sum += np.power(y - y_pred, 2)
    RMSE = np.sqrt(MSE_sum/float(N)) 
    return RMSE

def gibbs(M_train, train, test, N, M, d, sigma, lamb, N_gibbs, burn_in, thinning):
    U, V, I = initialize_factorization(N, M, d, sigma, lamb)
    gibbs_dict = initialize_gibbs_dict(test)
    log_likelihood_list = list() 
    for iter in xrange(burn_in):
	U = update_user_Gibbs(M_train, U, V, N, I, user_movie_dict)
	V = update_movie_Gibbs(M_train, U, V, N, I, movie_user_dict)
	if iter%10 == 0:
	    joint_LL = log_likelihood(train, U, V, sigma, lamb, N, M)
	    log_likelihood_list.append(joint_LL)
    for iter in xrange(N_gibbs - burn_in):
	if iter%10 == 0:
	    joint_LL = log_likelihood(train, U, V, sigma, lamb, N, M)
	    log_likelihood_list.append(joint_LL)
	if iter%thinning == 0:
	    gibbs_dict = sample_gibbs(test, gibbs_dict, U, V)
    rmse = compute_RMSE_Gibbs(gibbs_dict, test)    
    return rmse, log_likelihood_list


if __name__ == "__main__":

# parameterization
sigma = np.sqrt(0.25)
lamb = 10
d_list = [10, 20, 30]
d = 10

base_dir = os.path.join(os.getcwd(), 'movie_ratings')
train = load_data_dict(os.path.join(base_dir, 'ratings.txt'))
test = load_data_dict(os.path.join(base_dir, 'ratings_test.txt'))
users, movies, N, M = metadata(train, base_dir)

user_movie_dict = user_movie_dictionary(train, N, M)
movie_user_dict = movie_user_dictionary(train, N, M)

M_train = dict_to_matrix(train, N, M)
M_test = dict_to_matrix(test, N, M)

N_iterations = 100

N_gibbs = 500
burn_in = 250
thinning = 25


d=5
U5, V5, rmse_5, LL_5 = coordinate_ascent(
    M_train, test, N, M, d, N_iterations, user_movie_dict, movie_user_dict, sigma, lamb)
d=10
U10, V10, rmse_10, LL_10 = coordinate_ascent(
    M_train, test, N, M, d, N_iterations, user_movie_dict, movie_user_dict, sigma, lamb)
d=20
U20, V20, rmse_20, LL_20 = coordinate_ascent(
    M_train, test, N, M, d, N_iterations, user_movie_dict, movie_user_dict, sigma, lamb)
d=30
U30, V30, rmse_30, LL_30 = coordinate_ascent(
    M_train, test, N, M, d, N_iterations, user_movie_dict, movie_user_dict, sigma, lamb)
rmse_gibbs, LL_gibbs = gibbs(
    M_train, train, test, N, M, d, sigma, lamb, N_gibbs, burn_in, thinning)

X_coord = [x+1 for x in xrange(N_iterations)]
X_gibbs = [10*x+1 for x in xrange(len(LL_gibbs))]

plt.plot(X_coord, rmse_5, label='d=5')
plt.plot(X_coord, rmse_10, label='d=10')     
plt.plot(X_coord, rmse_20, label='d=20')     
plt.plot(X_coord, rmse_30, label='d=30')
plt.axhline(y=rmse_gibbs, color='black', label='Gibbs')
plt.xlabel('Iteration')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title ('RMSE convergence by d (dimensionality of latent space)')
plt.legend(loc=2)
plt.savefig('RMSE.png')
plt.show()

plt.plot(X_coord, LL_5, label='d=5')
plt.plot(X_coord, LL_10, label='d=10')
plt.plot(X_coord, LL_20, label='d=20')
plt.plot(X_coord, LL_30, label='d=30')
plt.xlabel('Iteration')
plt.ylabel('Joint Log Likelihood')
plt.title('Joint Log Likelihood MAP convergence by d')
plt.legend(loc=5)
plt.savefig('JLL.png')
plt.show()

plt.plot(X_gibbs, LL_gibbs)
plt.xlabel('Iteration')
plt.ylabel('Joint Log Likelihood')
plt.title('Joint Log Likelihood using Gibbs')
plt.save('JLL_Gibbs.png')
plt.show()

U = U10
V = V10

f = open(base_dir+'/movies.txt')
i = 0
movies = dict()
for line in f:
    movie_name = line.split('\n')[0]
    movies[i] = movie_name
    i += 1

import pandas as pd
import random

n = range(M)
random.shuffle(n)

movie_list = n[:3]

sim_movie_dict = dict()
for m in movie_list:
    sim_movie_dict[m] = list()
    print 'Base Movie: ', movies[m]
    col = range(M)
    col.remove(m)
    dist_Euc = []
    for j in col:
        d = np.linalg.norm(V[:,j] - V[:,m])
        dist_Euc.append(d)
    dist_Euc = pd.Series(dist_Euc)
    dist_Euc.sort() #inplace sort
    for idx in dist_Euc.index[:5]:
        sim_movie_dict[m].append(col[idx])
    print '5 most similar movies:'
    for x in sim_movie_dict[m]:
        print movies[x]








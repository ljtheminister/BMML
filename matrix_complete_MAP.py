import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
	
def map_to_ratings_N(x, ratings_range):
    idx = (np.abs(ratings_range - x)).argmin()
    return ratings_range[idx]

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
    M = int(subprocess.Popen('wc -l %s/movies.txt'%(base_dir), stdout=subprocess.PIPE, shell=True).communicate()[0].split()[0])
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
def initialize_factorization(N, M, d):
    U = np.zeros((N,d))
    V = np.zeros((d,M))
    mean = np.zeros(d)
    cov = np.power(sigma,2)*I
    for i in xrange(N):
	U[i,:] = np.random.multivariate_normal(mean, cov)
    for j in xrange(M):
	V[:,j] = np.random.multivariate_normal(mean, cov)
    return U, V

# user optimization
def update_user_MAP(M_train, U, V, N):
    for i in xrange(N):
	# compute V_i
	V_i = V[:, user_movie_dict[i]].transpose()
	# compute m_u
	m_u = M_train[i, user_movie_dict[i]]
	# compute u_MAP
	U[i,:] = np.dot(np.linalg.inv(lamb*np.power(sigma,2)*I + np.dot(V_i.transpose(), V_i)), np.dot(V_i.transpose(), m_u))
    return U

# movie optimizaiton
def update_movie_MAP(M_train, U, V, M):
    for j in xrange(M):
	# compute U
	U_j = U[movie_user_dict[j], :].transpose()
	# compute m_v
	m_v = M_train[movie_user_dict[j], j]
	# compute v_MAP
	V[:,j] = np.dot(np.linalg.inv(lamb*np.power(sigma,2)*I + np.dot(U_j, U_j.transpose())), np.dot(U_j, m_v))
    return V

def predict_ratings(U, V):
    M_pred = np.dot(U,V)      
    return M_pred      

def compute_RMSE(M_pred, test, ratings_range):  
    MSE_sum = 0
    N = len(test.keys())
    for i, j in test.keys():
	y_pred = map_to_ratings_N(M_pred[i,j], ratings_range)
	y = M_test[i,j]
	MSE_sum += np.power(y - y_pred, 2)
    RMSE = np.sqrt(MSE_sum/float(N)) 
    return RMSE

# coordinate ascent 
def coordinate_ascent(M_train, test, N, M, d, N_iterations, ratings_range):
    U, V = initialize_factorization(N, M, d)
    rmse_list = list()
    for iter in xrange(N_iterations):
	U = update_user_MAP(M_train, U, V, N)    
	V = update_movie_MAP(M_train, U, V, M)
	M_pred = predict_ratings(U, V)
	rmse = compute_RMSE(M_pred, test, ratings_range)
	rmse_list.append(rmse)
    return U, V, rmse_list

def plot_RMSE(N_iterations, rmse):
    X = [x+1 for x in xrange(N_iterations)]
    plt.plot(X, rmse)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.show()

   

if __name__ == "__main__":

# parameterization
sigma = np.sqrt(0.25)
lamb = 10
d_list = [10, 20, 30]
d = 10
I = np.identity(d)
N_ratings = 5
ratings_range = np.array([1 + x for x in xrange(N)])


base_dir = os.path.join(os.getcwd(), 'movie_ratings')
train = load_data_dict(os.path.join(base_dir, 'ratings.txt'))
test = load_data_dict(os.path.join(base_dir, 'ratings_test.txt'))
users, movies, N, M = metadata(train, base_dir)

user_movie_dict = user_movie_dictionary(train, N, M)
movie_user_dict = movie_user_dictionary(train, N, M)

M_train = dict_to_matrix(train, N, M)
M_test = dict_to_matrix(test, N, M)

N_iterations = 100
U,V, rmse = coordinate_ascent(M_train, test, N, M, d, N_iterations, ratings_range)
plot_RMSE(N_iterations, rmse)

## GIBBS SAMPLING
N_iterations = 100

N_gibbs = 500
burn_in = 250
thinning = 25

def gibbs(M_train, test, N, M, d, N_gibbs, burn_in, thinning):

U, V = initialize_factorization(N, M, d)
rmse_list = list()
for iter in xrange(N_gibbs):
    




# initialize model parameters U, V

# 
for t in xrange(N_iterations):
    # sample hyperparameters


    # sample user features


    # sample movie features
























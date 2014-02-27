import numpy as np
import subprocess
import os
	
# parameterization
sigma = np.sqrt(0.25)
lamb = 10

d_list = [10, 20, 30]
d = 10
I = np.identity(d)

N = 5
ratings_range = np.array([1 + x for x in xrange(N)])

def map_to_ratings_N(x, ratings_range):
    idx = (np.abs(ratings_range - x)).argmin()
    return ratings_range[idx]

'''
def load_data(filename):
    X, y = [], [] 
    with open(filename, 'rb') as f:
	for line in f:
	    row = line.split(',')
	    X.append((int(row[0])-1, int(row[1])-1))
	    y.append(int(row[2]))
    return X, y
'''

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
def update_user_MAP(U, V, N):
    for i in xrange(N):
	# compute V_i
	V_i = V[:, user_movie_dict[i]].transpose()
	# compute m_u
	m_u = mat[i, user_movie_dict[i]]
	# compute u_MAP
	U[i,:] = np.dot(np.linalg.inv(lamb*np.power(sigma,2)*I + np.dot(V_i.transpose(), V_i)), np.dot(V_i.transpose(), m_u))
    return U

# movie optimizaiton
def update_movie_MAP(U, V, M):
    for j in xrange(M):
	# compute U
	U_j = U[movie_user_dict[j], :].transpose()
	# compute m_v
	m_v = mat[movie_user_dict[j], j]
	# compute v_MAP
	V[:,j] = np.dot(np.linalg.inv(lamb*np.power(sigma,2)*I + np.dot(U_j, U_j.transpose())), np.dot(U_j, m_v))
    return V

# coordinate ascent 
def coordinate_ascent(N, M, d, N_iterations):
    U, V = initialize_factorization(N, M, d)
    for iter in xrange(N_iterations):
	U = update_user_MAP(U, V, N)    
	V = update_user_MAP(U, V, M)
    return U, V

def predict_ratings(U, V):
    M_pred = np.dot(U,V)      
    return M_pred      

def RMSE(M_pred, M_test, ratings_range):  
    MSE_sum = 0
    N = len(M_test.keys())
    for i, j in M_test.keys():
	y_pred = map_to_ratings_N(M_pred[i,j], ratings_range)
	y = M_test[i,j]
	MSE_sum += np.power(y - y_pred, 2)
    RMSE = np.sqrt(MSE_sum/float(N)) 
    return RMSE









if __name__ == "__main__":

base_dir = os.path.join(os.getcwd(), 'movie_ratings')

'''
X_train, y_train = load_data(base_dir + 'ratings.txt')
X_test, y_test = load_data(base_dir + 'ratings_test.txt')
'''

M_train = load_data_dict(os.path.join(base_dir, 'ratings.txt'))
M_test = load_data_dict(os.path.join(base_dir, 'ratings_test.txt'))


mat = np.zeros((N,M))
for user, movie in M_train.keys():
    mat[user, movie] = M_train[(user, movie)]


user_movie_dict = user_movie_dictionary(train, N, M)
movie_user_dict = movie_user_dictionary(train, N, M)

N_iterations = 100


## GIBBS SAMPLING
N_iterations = 100

N_Gibbs = 500
burn_in = 250
thinning = 25


# initialize model parameters U, V

# 
for t in xrange(N_iterations):
    # sample hyperparameters


    # sample user features


    # sample movie features
























import numpy as np
import os


# parameterization
sigma = 0.25
lamb = 10

d_list = [10, 20, 30]
d = 10

N = 5
ratings_range = np.array([1 + x for x in xrange(N)])

def map_to_ratings_N(x, ratings_range):
    idx = (np.abs(ratings_range - x)).argmin()
    return ratings_range[idx]


def load_data(filename):
    X, y = [], [] 
    with open(filename, 'rb') as f:
	for line in f:
	    row = line.split(',')
	    X.append((int(row[0]), int(row[1])))
	    y.append(int(row[2]))
    return X, y


def load_data_dict(filename):
    data = dict()
    with open(filename, 'rb') as f:
	for line in f:
	    row = line.split(',')
	    user, movie, rating = int(row[0]), int(row[1]), int(row[2])
	    data[(user, movie)] = rating
    return data






if __name__ == "__main__":

base_dir = '../movie_ratings/'
X_train, y_train = load_data(base_dir + 'ratings.txt')
X_test, y_test = load_data(base_dir + 'ratings_test.txt')

train = load_data_dict(base_dir + 'ratings.txt')
test = load_data_dict(base_dir + 'ratings_test.txt')

users = list()
movies = list()

for user, movie in train.keys():
    users.append(user)
    movies.append(movie)

users = sorted(list(set(users)))
movies = sorted(list(set(movies)))

N = len(users)
M = int(subprocess.Popen('wc -l movies.txt', stdout=subprocess.PIPE, shell=True).communicate()[0].split()[0])
M = 1682

mat = np.zeros((N,M))
for user, movie in train.keys():
    mat[user-1][movie-1] = train[(user, movie)]






















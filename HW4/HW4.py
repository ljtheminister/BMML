import numpy as np
from GMM_EM import GMM_EM
from sklearn.mixture import GMM



X = np.loadtxt('data.txt', delimiter=',').T

np.random.seed(1)
gmm = GMM(n_components=2)
gmm.fit(X)



gmm = GMM_EM(X, 4, max_iter=100)
gmm.initialize_clusters()
gmm.EM()










gmm.M_step()
gmm.E_step()

for i in xrange(28):
    gmm.M_step()
    gmm.E_step()


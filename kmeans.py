import numpy as np
import itertools

class kMeans:
    def __init__(self, n_clusters, max_iter=1000, random_seed=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)
    
    def fit(self, X):
        cycle = itertools.cycle(range(self.n_clusters))
        self.labels_ = np.fromiter(itertools.islice(cycle, X.shape[0]), dtype=np.int) #0,1,2,3,4 ... n_clusters, 0,1,...
        self.random_state.shuffle(self.labels_)
        labels_prev = np.zeros(X.shape[0])
        count = 0
        self.cluster_clusters_ = np.zeros((self.n_clusters, X.shape[1]))

        while (not (self.labels_ == labels_prev).all() and count < self.max_iter):
            for i in range(self.n_clusters):
                XX = X[self.labels_ == i,:]
                self.cluster_clusters_[i,:] = XX.mean(axis=0)
            dist = ((X[:,:,np.newaxis] - self.cluster_clusters_.T[np.newaxis,:,:])**2).sum(axis=1)
            labels_prev = self.labels_
            self.labels_ = dist.argmin(axis=1)
            count += 1
    
    def predict(self, X):
        dist = ((X[:,:,np.newaxis] - self.cluster_clusters_.T[np.newaxis,:,:])**2).sum(axis=1)
        labels = dist.argmin(axis=1)
        return labels


         















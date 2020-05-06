import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
class CustomKMeans:
    def __init__(self,n_clusters,random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        center_indices = np.random.choice(X.shape[0], size = self.n_clusters)
        centers = X[center_indices]
        
        while True:
            cluster_assignments, distances = pairwise_distances_argmin_min(X, centers,metric = 'euclidean')
        
            # Find new center by taking mean of cluster points
            new_centers = []
            for i in range(self.n_clusters):
                cluster_points_index = cluster_assignments == i
                #print('Cluster',i)
                cluster_points = X[cluster_points_index]
                new_center = cluster_points.mean(axis = 0)
                new_centers.append(new_center)
            new_centers = np.array(new_centers)
            # new_centers = np.array([np.mean(X[cluster_assignments == i],axis = 0) for i in range(k)])
            
            # Test for convergence
            if np.all(new_centers == centers):
                break

            centers = new_centers
        self.labels_ = cluster_assignments
        self.cluster_centers_ = centers
        self.inertia_ = np.sum(np.square(distances))
        return self
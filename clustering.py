from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors


class CustomDBSCAN(BaseEstimator, ClusterMixin):
    def __init__(self, eps=7.5, min_pts=3, metric="euclidean"):
        self.metric = metric
        self.eps = eps
        self.min_pts = min_pts
        self.clusters_ = None
        self.visited_ = None
        self.neighbors = None

    def fit(self, X):
        # Initialize all points with cluster id as index of points in data
        self.clusters_ = [-1] * X.shape[0]
        cluster_id = 0
        # Neighborhood creation
        nbrs = NearestNeighbors(metric=self.metric, n_jobs=-1, radius=self.eps, algorithm='auto').fit(X)
        self.neighbors = nbrs.radius_neighbors(X, return_distance=False)
        self.visited_ = [False] * X.shape[0]
        # find neighbourhood of each point x.
        # this code works like BFS, start with a point and then find its neighborhood if size of nx >= minPts then
        # assign each point in nx as part of cluster x_id and then repeat this with points in nx
        for i in range(X.shape[0]):
            if not self.visited_[i]:
                self.visited_[i] = True
                nx = self.neighbors[i]
                if len(nx) >= self.min_pts:
                    self.expand_cluster(i, nx, cluster_id)
                    cluster_id += 1
        return self

    def expand_cluster(self, x_id, neighbors, cluster_id):
        queue = list(neighbors)
        while queue:
            pt = queue.pop(0)
            if not self.visited_[pt]:
                self.visited_[pt] = True
                nx = self.neighbors[pt]
                if len(nx) >= self.min_pts:
                    queue.extend(nx)
                    # assign cluster to point in neighborhood as x_id
                if self.clusters_[pt] == -1:
                    self.clusters_[pt] = cluster_id


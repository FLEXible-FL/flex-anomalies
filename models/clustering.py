from models import BaseModel
from sklearn.cluster import KMeans
class ClusterAnomaly(BaseModel):
    def __init__(
        self,
        n_clusters,
        seed,
        init_centroids='random',
        max_iter=100,
        tol=0.0001,
        n_init=1,
        verbose=True,
        precompute_distances=True,
        algorithm='full'
    )-> None:
        super(ClusterAnomaly, self).__init__()
        self.n_clusters = n_clusters
        self.seed = seed
        self.init_centroids = init_centroids
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.n_init = n_init
        self.precompute_distances = precompute_distances
        self.algorithm = algorithm
        self.model = self._build_model()
def _build_model(self):
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state= self.seed,
            init=self.init_centroids,  # 'random', 'k-means++', ndarray (n_clusters, n_features)
            max_iter=self.max_iter,
            tol= self.tol,
            n_init= self.n_init,
            # verbose=True,
            precompute_distances= self.precompute_distances,
            algorithm=self.algorithm,
        )
        return kmeans

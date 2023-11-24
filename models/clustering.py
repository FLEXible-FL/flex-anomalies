from models import BaseModel
from sklearn.cluster import KMeans
import numpy as np

class ClusterAnomaly(BaseModel):
    def __init__(
        self,
        n_clusters,
        seed = None,
        init_centroids ='random',
        max_iter =100,
        tol = 0.0001,
        n_init = 1,
        verbose = True,
        algorithm ='full',
        contamination = 0.1
    )-> None:
        super(ClusterAnomaly, self).__init__(contamination = contamination)
        self.n_clusters = n_clusters
        self.seed = seed
        self.init_centroids = init_centroids
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.n_init = n_init
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
            algorithm=self.algorithm
        )
        return kmeans
        
    def fit(self, X, y=None):
        """
        X : numpy array of shape (samples, features)
        """
        self.model.fit(X=X, y=y)  
        
        # attributes from the sklearn cluster KMeans
        self.cluster_centers_ = self.model.cluster_centers_
        

    def predict(self, X):
        self.d_scores_ = self.decision_function(X)
        self.process_scores()    
        return self 
   
    def decision_function(self,X):
        """
         X : numpy array of shape (n_samples, n_features)

         Returns  anomaly scores : numpy array of shape (n_samples,)
                 The anomaly score of the input samples.
        """
        # distance to the cluster centers
        dist = self.model.transform(X)  # shape (n_samples, n_clusters)
        scores = np.min(dist, axis = 1) 
        return scores

                           
          # # memory efficient
        # sq_dist = np.zeros((len(X), self.n_clusters))
        # for i in range(self.n_clusters):
        #     sq_dist[:, i] = np.sum(np.square(x - self.cluster_centers_[i, :]), axis=1)
        # labels = np.argmin(sq_dist, axis=1)
        # return labels
        
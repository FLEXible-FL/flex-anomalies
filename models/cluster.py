from models import BaseModel
from sklearn.cluster import KMeans
import numpy as np
from datetime import datetime
import pickle

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
        algorithm ='lloyd',
        model_path = '',
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
        self.model_path = model_path
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

    def load_model(self, model_path=''):
        self.model = pickle.load(open(f"{model_path}/model.pkl" if model_path else f"{self.model_path}/model.pkl", 'rb'))

    def save_model(self, model_path=''):
        if not model_path and not self.model_path:
            raise "You must provide a path to save model"
        pickle.dump(self.model, open(f"{model_path}/model.pkl" if model_path else f"{self.model_path}/model.pkl",'wb'))                
     
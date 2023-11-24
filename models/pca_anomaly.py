from models import BaseModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import cdist

class PCA_Anomaly(BaseModel):
    def __init__(
        self,
        n_components=None,
        n_selected_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
        scaler = True,
        contamination = 0.1,
        preprocess = True,
        weighted = True
    ) -> None:
        super(PCA_Anomaly, self).__init__(contamination = contamination)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.weighted = weighted,
        self.preprocess = preprocess
        self.scaler = StandardScaler()

        self.model = self._build_model()

    def _build_model(self):
        model = PCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.random_state,
        )

        return model

    def fit(self, X, y=None):
        """
        X : numpy array of shape (samples, features)
        y:  Ignored in unsupervised methods
        """
        
        # Standardize data 
        if self.preprocess:
             Xscaler = self.scaler.fit_transform(X)
        else:
             Xscaler = np.copy(X)
        
        self.model.fit(X=Xscaler, y=y)



        # attributes from the sklearn PCA 
        self.n_components_ = self.model.n_components_
        self.components_ = self.model.components_

        # selected num components 
        if self.n_selected_components is None:
            self.n_selected_components_ = self.n_components_
        else:
            self.n_selected_components_ = self.n_selected_components

        self.w_components_ = np.ones([self.n_components_, ])
        if self.weighted:
            self.w_components_ = self.model.explained_variance_ratio_

        self.selected_components_ = self.components_[
                                    -1 * self.n_selected_components_:, :]
        self.selected_w_components_ = self.w_components_[
                                      -1 * self.n_selected_components_:]

        self.decision_scores_ = np.sum(
            cdist(X, self.selected_components_) / self.selected_w_components_,
            axis=1).ravel()

        self.process_scores()
        return self         
        
        
    def decision_funcion(self, X):
         """
         X : numpy array of shape (n_samples, n_features)

         Returns  anomaly scores : numpy array of shape (n_samples,)
                 The anomaly score of the input samples.
         """
         return np.sum(
            cdist(X, self.selected_components_) / self.selected_w_components_,
            axis=1).ravel()



       

    
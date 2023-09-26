from models import BaseModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
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
        scaler = True
    ) -> None:
        super(PCA_Anomaly, self).__init__()
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.scaler = scaler
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
        if self.scaler:
             X_scaler = StandardScaler().fit_transform(X)
        else:
             X_scaler = np.copy(X)

        self.model.fit(X=X_scaler, y=y)
        
    def predict_outlier(self, X):
        pass

    def evaluate(self, X):
        pass 
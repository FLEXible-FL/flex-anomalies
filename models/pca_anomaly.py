from models import BaseModel
from sklearn.decomposition import PCA as sklearn_PCA


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
        self.model = self._build_model()

    def _build_model(self):
        model = sklearn_PCA(
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
        self.model.fit(X=X, y=y)

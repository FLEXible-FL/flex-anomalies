from flexanomalies.models import BaseModel
from sklearn.ensemble import IsolationForest as sklearn_IF
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


class IsolationForest(BaseModel):
    """
    IsolationForest with scikit-learn

    random_state : int, RandomState instance, optional (default=None)
    n_estimators : int, optional (default=100)
    max_samples : float or int, optional (default="auto")
    bootstrap : bool, optional (default=False)
    max_features : float or int, optional (default=1.0)
    n_jobs : int, optional (default=1)
    contamination : float in (0., 0.5), optional (default=0.1)
                    Contamination of the data set, the proportion of outliers in the data set.

    """

    def __init__(
        self,
        n_estimators=100,
        max_samples=1000,
        contamination=0.1,
        max_features=1.0,
        bootstrap=False,
        n_jobs=1,
        behaviour="old",
        random_state=None,
        model_path="",
        verbose=0,
    ):
        super(IsolationForest, self).__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.behaviour = behaviour
        self.random_state = random_state
        self.verbose = verbose
        self.model_path = model_path
        self.model = self._build_model()

    def _build_model(self):
        model = sklearn_IF(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        return model

    def fit(self, X, y=None):
        """
        X : numpy array of shape (samples, features)
        y:  Ignored in unsupervised methods

        """
        self.model.fit(X=X, y=y)

    def predict(self, X,y = None):
        self.d_scores_ = self.model.decision_function(X)
        self.d_scores_ = (
            self.d_scores_ * -1
        )  # Invert order scores. Outliers have higher scores
        self.process_scores()
        return self

    def decision_function(self, X,y = None):

        """
        X : numpy array of shape (n_samples, n_features)

        Returns  anomaly scores : numpy array of shape (n_samples,)
                The anomaly score of the input samples.
        """
        return (self.model.decision_function(X)) * -1

    def load_model(self, model_path=""):
        self.model = pickle.load(
            open(
                f"{model_path}/model.pkl"
                if model_path
                else f"{self.model_path}/model.pkl",
                "rb",
            )
        )

    def save_model(self, model_path=""):
        if not model_path and not self.model_path:
            raise Exception("You must provide a path to save model")
        pickle.dump(
            self.model,
            open(
                f"{model_path}/model.pkl"
                if model_path
                else f"{self.model_path}/model.pkl",
                "wb",
            ),
        )

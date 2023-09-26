from models import BaseModel

class IsolationForest(BaseModel):
    """
      IsolationForest with scikit-learn

      random_state : int, RandomState instance, optional (default=None)
      n_estimators : int, optional (default=100)
      max_samples : float or int, optional (default="auto")
      bootstrap : bool, optional (default=False)
      max_features : float or int, optional (default=1.0)   
      n_jobs : int, optional (default=1)
      contamination : float, optional (default=0.1)
    """
    def __init__(self, n_estimators=100,
                 max_samples="auto",
                 contamination = 0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 behaviour='old',
                 random_state=None,
                 verbose=0):
        super(IsolationForest, self).__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.behaviour = behaviour
        self.random_state = random_state
        self.verbose = verbose
        self.model = self._build_model() 


    def _build_model(self):
        model = IsolationForest(n_estimators=self.n_estimators,
                                         max_samples=self.max_samples,
                                         contamination=self.contamination,
                                         max_features=self.max_features,
                                         bootstrap=self.bootstrap,
                                         n_jobs=self.n_jobs,
                                         random_state=self.random_state,
                                         verbose=self.verbose)
        return model
    def fit(self, X, y=None):
        self.model.fit(X=X, y=y)   
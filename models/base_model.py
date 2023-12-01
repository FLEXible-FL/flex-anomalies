import abc
import six
import numpy as np
from utils.metrics import *

"""
contamination : float in (0., 0.5), optional (default=0.1)
                Contamination of the data set, the proportion of outliers in the data set.
"""
@six.add_metaclass(abc.ABCMeta)
class BaseModel(object):
    @abc.abstractmethod
    def __init__(self,contamination = 0.1) -> None:
        if (isinstance(contamination, (float, int))):

            if not (0. < contamination <= 0.5):
                  raise ValueError("contamination must be in (0, 0.5], "
                                   "got: %f" % contamination)

        self.contamination = contamination

    @abc.abstractmethod
    def _build_model(self):
        pass

    @abc.abstractmethod
    def fit(self, X, y=None):
        """
        X : numpy array of shape (samples, features)
        """
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def decision_function():
        pass

    def evaluate(self, X, y =None, metrics = ["Accuracy", "Precision", "F1", "Recall", "AUC_ROC"]):
        self.predict(X)
        self.result_metrics_= print_metrics(metrics,y,self.labels_)


    def process_scores(self):

        if isinstance(self.contamination, (float, int)):
            # threshold is num anomalies
            num_anomalies = int(self.contamination * (len(self.d_scores_)))  
            ind_anomalies = np.flip(np.argsort(self.d_scores_))[:num_anomalies]   # Revisar el orden 
            self.labels_ = np. zeros(len(self.d_scores_))                            
            self.labels_[ind_anomalies] = 1
        
        return self 
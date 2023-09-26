import abc
import six


@six.add_metaclass(abc.ABCMeta)
class BaseModel(object):
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

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
    def predict_outlier(self, X):
        pass

    @abc.abstractmethod
    def evaluate(self, X):
        pass

# Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import abc
import six
import numpy as np
from flexanomalies.utils.metrics import *
from numpy import percentile

"""
Base model
contamination : float in (0., 0.5), optional (default=0.1)
                Contamination of the data set, the proportion of outliers in the data set.
"""


@six.add_metaclass(abc.ABCMeta)
class BaseModel(object):
    @abc.abstractmethod
    def __init__(self, contamination=0.1) -> None:
        if isinstance(contamination, (float, int)):

            if not (0.0 < contamination <= 0.5):
                raise ValueError(
                    "contamination must be in (0, 0.5], " "got: %f" % contamination
                )

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

    def evaluate(
        self, X, y=None, metrics=["Accuracy", "Precision", "F1", "Recall", "AUC_ROC"]
    ):
        self.predict(X, y)
        self.result_metrics_ = print_metrics(metrics, y, self.labels_)

    def process_scores(self):

        if isinstance(self.contamination, (float, int)):
            # threshold is num anomalies
            num_anomalies = int(self.contamination * (len(self.d_scores_)))
            ind_anomalies = np.flip(np.argsort(self.d_scores_))[
                :num_anomalies
            ]  # Revisar el orden
            self.labels_ = np.zeros(len(self.d_scores_))
            self.labels_[ind_anomalies] = 1

        return self

    def process_scores_with_percentile(self):
        self.threshold_ = percentile(
            np.mean(self.d_scores_), 100 * (1 - self.contamination)
        )
        self.labels_ = (self.d_scores_ > self.threshold_).astype("int").ravel()

        return self

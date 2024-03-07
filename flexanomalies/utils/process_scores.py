#Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)
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


import numpy as np
from numpy import percentile


def process_scores(d_scores, contamination):

    """function to calculate threshold used to decide the binary label:
       labels_: binary labels of training data
    Args:
    ----
       d_scores: decision score
       contamination:  float in (0., 0.5), optional (default=0.1)
                       Contamination of the data set, the proportion of outliers in the data set.
    Returns:
    -------
       labels_

    """
    num_anomalies = int(contamination * (len(d_scores)))
    ind_anomalies = np.flip(np.argsort(d_scores))[:num_anomalies]  # Revisar el orden
    labels_ = np.zeros(len(d_scores))
    labels_[ind_anomalies] = 1
    return labels_


def process_scores_with_percentile(d_scores, contamination):

    """function to calculate threshold used to decide the binary label

    Args:
    ----
       d_scores: decision score
       contamination:  float in (0., 0.5), optional (default=0.1)
                       Contamination of the data set, the proportion of outliers in the data set.
    Returns:
    -------
           threshold

    """
    threshold_ = percentile(np.mean(d_scores), 100 * (1 - contamination))
    return threshold_


def process_scores_with_threshold(d_scores):
    """function to calculate threshold used to decide the binary label

    Args:
    ----
       d_scores: decision score
    Returns:
    -------
           threshold

    """
    threshold_ = np.mean(d_scores) + 2 * np.std(d_scores)
    return threshold_

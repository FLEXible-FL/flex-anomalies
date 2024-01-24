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

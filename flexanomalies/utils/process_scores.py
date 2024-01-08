import numpy as np
from numpy import percentile


def process_scores(d_scores, contamination):

    num_anomalies = int(contamination * (len(d_scores)))
    ind_anomalies = np.flip(np.argsort(d_scores))[:num_anomalies]  # Revisar el orden
    labels_ = np.zeros(len(d_scores))
    labels_[ind_anomalies] = 1
    return labels_


def process_scores_with_percentile(d_scores, contamination):
    threshold_ = percentile(np.mean(d_scores), 100 * (1 - contamination))
    return threshold_


def process_scores_with_threshold(d_scores):
    threshold_ = np.mean(d_scores) + 2 * np.std(d_scores)
    return threshold_

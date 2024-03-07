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


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np


def metric_precision(y, y_prediction):
    TP, FP, TN, FN = measure(y, y_prediction)
    precision = TP / (TP + FP)
    return precision


def metric_recall(y, y_prediction):
    TP, FP, TN, FN = measure(y, y_prediction)
    recall = TP / (TP + FN)
    return recall


def metric_accuracy(y, y_prediction):
    TP, FP, TN, FN = measure(y, y_prediction)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc * 100


def metric_F1score(y, y_prediction):
    precision = metric_precision(y, y_prediction)
    recall = metric_recall(y, y_prediction)
    F1score = 2 * ((precision * recall) / (precision + recall))
    return F1score


def metric_AUC_ROC(y, y_prediction):
    return roc_auc_score(y, y_prediction)


def measure(y, y_prediction):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_prediction)):
        if y[i] == y_prediction[i] == 1:
            TP += 1
        if y_prediction[i] == 1 and y[i] != y_prediction[i]:
            FP += 1
        if y[i] == y_prediction[i] == 0:
            TN += 1
        if y_prediction[i] == 0 and y[i] != y_prediction[i]:
            FN += 1
    return (TP, FP, TN, FN)


def print_metrics(list_metrics, y, y_prediction):
    """function to print the metrics
    Args:
    ----
    y: labels
    y_prediction: predicted labels
    list_metrics: list, containing list of metrics to display, metrics name (Accuracy, Precision, F1, Recall, AUC_ROC, ConfusionMatrix).

    """
    metrics = {}

    for i in list_metrics:
        if i == "Accuracy":
            acc = metric_accuracy(y, y_prediction)
            metrics["Accuracy"] = "%.3f%%" % acc
            print("Acc: %.3f%% \n" % acc)

        if i == "Recall":
            r = metric_recall(y, y_prediction)
            metrics["Recall"] = "%.3f" % r
            print("Recall: %.3f \n" % r)

        if i == "F1":
            f1 = metric_F1score(y, y_prediction)
            metrics["F1"] = "%.3f" % f1
            print("F1score: %.3f \n" % f1)

        if i == "Precision":
            p = metric_precision(y, y_prediction)
            metrics["Precision"] = "%.3f" % p
            print("Precision: %.3f \n" % p)

        if i == "AUC_ROC":
            auc = metric_AUC_ROC(y, y_prediction)
            metrics["AUC_ROC"] = "%.3f" % auc
            print("AUC_ROC: %.3f \n" % auc)

    return metrics


def distances(y, y_prediction):
    return np.linalg.norm(y - y_prediction, axis=1)

from flex.pool import init_server_model
from flex.model import FlexModel
from flex.data import Dataset
from flex.pool import deploy_server_model
from flex.pool.decorators import collect_clients_weights
from flex.pool.decorators import set_aggregated_weights
from flexanomalies.utils.process_scores import process_scores_with_percentile
from flexanomalies.utils.metrics import print_metrics
import numpy as np
import tensorflow as tf
from copy import deepcopy


@init_server_model
def build_server_model_ae(model):
    """
    Function to initialize the server model
    """
    flex_model = FlexModel()
    flex_model["model"] = model
    return flex_model


@deploy_server_model
def copy_model_to_clients_ae(server_flex_model):
    client_flex_model = deepcopy(server_flex_model)
    return client_flex_model


@collect_clients_weights
def weights_collector_ae(client_model):
    return client_model["model"].model.get_weights()


@set_aggregated_weights
def set_aggregated_weights_ae(server_flex_model, aggregated_weights, *args, **kwargs):
    server_flex_model["model"].model.set_weights(aggregated_weights)


def train_ae(client_model, client_data, **kwargs):
    print("Training model at client.")
    model = client_model["model"]
    slice = int(0.9 * len(client_data.to_numpy()[0]))
    X_data = client_data.to_numpy()[0][:slice]
    y_data = client_data.to_numpy()[1][:slice]

    model.fit(X_data, y_data)


def evaluate_global_model(
    model,
    X,
    y,
    labels,
    metrics=["Accuracy", "Precision", "F1", "Recall", "AUC_ROC"],
    threshold=None,
):
    """Evaluate global model on the server with a global test set.

    Args:
        server_flex_model (FlexModel): Server Flex Model.
        X : Array with the data to evaluate.
        y:Output data matrix. Depending on the chosen window model (it can be the set X).
        labels: Labels of the data to evaluate.
        metrics: Metrics to evaluate.
        threshold: Anomaly threshold to evaluate.

    """
    prediction = model.predict(X, y)
    d_scores = np.mean((y - prediction), axis=2).flatten()
    if threshold is None:
        threshold = process_scores_with_percentile(d_scores, 0.1)

    l = (d_scores > threshold).astype("int").ravel()
    model.result_metrics_ = print_metrics(metrics, labels, l)


def evaluate_global_model_clients(client_flex_model, client_data, *args, **kwargs):
    """Evaluate global model on the client.

    Args:
        client_flex_model (FlexModel): Client Flex Model.
        client_data: Array with the data to evaluate.

    """
    slice = int(0.9 * len(client_data.to_numpy()[0]))
    X_test = client_data.to_numpy()[0][slice:]
    y_test = client_data.to_numpy()[1][slice:]
    p = client_flex_model["model"].predict(X_test, y_test)

    d_scores = np.mean((y_test - p), axis=2).flatten()
    threshold = process_scores_with_percentile(d_scores, 0.1)

    client_flex_model["threshold"] = threshold


def threshold_collector_ae(client_model, client_data):
    return client_model["threshold"]

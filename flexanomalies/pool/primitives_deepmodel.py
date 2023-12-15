from flex.pool import init_server_model
from flex.model import FlexModel
from flex.data import Dataset
from flex.pool import deploy_server_model
from flex.pool.decorators import collect_clients_weights
from flex.pool.decorators import set_aggregated_weights
from flexanomalies.utils.process_scores import process_scores_with_percentile
import numpy as np
import tensorflow as tf
from copy import deepcopy


@init_server_model
def build_server_model_ae(model):
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


def train_ae(client_model, client_data):
    print("Training model at client.")
    model = client_model["model"]
    X_data, y_data = client_data.to_numpy()
    model.fit(X_data, y_data)
    
def evaluate_global_model_clients(
    client_flex_model,
    client_data,
    *args, **kwargs
):

    X_test, y_test = client_data.to_numpy()
    p = client_flex_model['model'].predict(X_test, y_test)
    d_scores = (np.linalg.norm(y_test - p, axis = 2))
    threshold = process_scores_with_percentile(d_scores, 0.1)
    return threshold
from flex.pool import init_server_model
from flex.model import FlexModel
from flex.pool import deploy_server_model
from flex.pool.decorators import collect_clients_weights
from flex.pool.decorators import aggregate_weights
from flex.pool.decorators import set_aggregated_weights
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


@aggregate_weights
def aggregate_ae(agg_model):
    return np.mean(np.array(agg_model, dtype=object), axis=0)


@set_aggregated_weights
def set_aggregated_weights_ae(server_flex_model, aggregated_weights, *args, **kwargs):
    server_flex_model["model"].model.set_weights(aggregated_weights)


def train_ae(client_model, client_data):
    print("Training model at client.")
    model = client_model["model"]
    X_data, y_data = client_data.to_numpy()
    model.fit(X_data, y_data)
    
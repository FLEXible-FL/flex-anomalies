from flex.pool import init_server_model
from flex.model import FlexModel
from flex.pool import deploy_server_model
from flex.pool.decorators import collect_clients_weights
from flex.pool.decorators import set_aggregated_weights
import numpy as np
from copy import deepcopy


@init_server_model
def build_server_model_cl(model):
    flex_model = FlexModel()
    flex_model["model"] = model
    return flex_model


@deploy_server_model
def copy_model_to_clients_cl(server_flex_model):
    client_flex_model = deepcopy(server_flex_model)
    return client_flex_model


@collect_clients_weights
def get_clients_weights_cl(client_model):
    return client_model["model"].model.cluster_centers_


@set_aggregated_weights
def set_aggregated_weights_cl(server_flex_model, aggregated_weights, *args, **kwargs):
    server_flex_model["model"].model.cluster_centers_ = aggregated_weights


def train_cl(client_model, client_data):
    print("Training model at client.")
    model = client_model["model"]
    X_data, y_data = client_data.to_numpy()
    model.fit(X_data, y_data)
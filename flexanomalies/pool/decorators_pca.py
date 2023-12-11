from flex.pool import init_server_model
from flex.model import FlexModel
from flex.pool import deploy_server_model
from flex.pool.decorators import collect_clients_weights
from flex.pool.decorators import aggregate_weights
from flex.pool.decorators import set_aggregated_weights
import numpy as np
from copy import deepcopy


@init_server_model
def build_server_model_pca(model):
    flex_model = FlexModel()
    flex_model["model"] = model
    return flex_model


@deploy_server_model
def copy_model_to_clients_pca(server_flex_model):
    client_flex_model = deepcopy(server_flex_model)
    return client_flex_model


@collect_clients_weights
def get_clients_weights_pca(client_model):
    return client_model["model"].model


@aggregate_weights
def aggregate_pca(agg_model):
    components = [model.components_ for model in agg_model]
    n_components = agg_model[0].n_components_
    ratio = [model.explained_variance_ratio_ for model in agg_model]
    return {'components': np.mean(np.array(components), axis=0), 'n_components':n_components, 'ratio':np.mean(np.array(ratio), axis=0)}



@set_aggregated_weights
def set_aggregated_weights_pca(server_flex_model, aggregated_weights, *args, **kwargs):
    server_flex_model["model"].model.components_ = aggregated_weights['components']
    server_flex_model["model"].model.n_components_ = aggregated_weights['n_components']
    server_flex_model["model"].model.explained_variance_ratio_ = aggregated_weights['ratio']


def train_pca(client_model, client_data):
    print("Training model at client.")
    model = client_model["model"]
    X_data, y_data = client_data.to_numpy()
    model.fit(X_data, y_data)
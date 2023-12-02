from models import AutoEncoder
from flex.pool import init_server_model
from flex.pool import FlexPool
from flex.model import FlexModel
from flex.pool import deploy_server_model
from flex.pool.decorators import collect_clients_weights
from flex.pool.decorators import aggregate_weights
from flex.pool.decorators import set_aggregated_weights
import numpy as np
import tensorflow as tf
from utils.load_data import federate_data
from copy import deepcopy


@init_server_model
def build_server_model(model):
    flex_model = FlexModel()
    flex_model["model"] = model
    return flex_model


@deploy_server_model
def copy_model_to_clients(server_flex_model):
    client_flex_model = deepcopy(server_flex_model)
    return client_flex_model


@collect_clients_weights
def tensorflow_weights_collector(client_model):
    return client_model["model"].model.get_weights()


@aggregate_weights
def fed_avg(agg_model):
    return np.mean(np.array(agg_model, dtype=object), axis=0)


@set_aggregated_weights
def set_aggregated_weights_tf(server_flex_model, aggregated_weights, *args, **kwargs):
    server_flex_model["model"].model.set_weights(aggregated_weights)


def train(client_model, client_data):
    print("Training model at client.")
    model = client_model["model"]
    X_data, y_data = client_data.to_numpy()
    model.fit(X_data, y_data)


def test_autoencoder(
    model_params, X_train, X_test, y_train, y_test, n_clients=3, n_rounds=5
):
    model = AutoEncoder(**model_params)
    flex_dataset = federate_data(n_clients, X_train, y_train)
    pool = FlexPool.client_server_pool(
        fed_dataset=flex_dataset,
        server_id="autoencoder_server",
        init_func=build_server_model,
        model=model,
    )

    for i in range(n_rounds):
        print(f"\nRunning round: {i}\n")
        pool.servers.map(copy_model_to_clients, pool.clients)
        pool.clients.map(train)
        pool.aggregators.map(tensorflow_weights_collector, pool.clients)
        pool.aggregators.map(fed_avg)
        pool.aggregators.map(set_aggregated_weights_tf, pool.servers)
    output_model = pool.servers._models["autoencoder_server"]["model"]
    output_model.evaluate(X_test, y_test)
    return output_model

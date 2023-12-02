from models import ClusterAnomaly
from flex.pool import init_server_model
from flex.pool import FlexPool
from flex.model import FlexModel
from flex.pool import deploy_server_model
from flex.pool.decorators import collect_clients_weights
from flex.pool.decorators import aggregate_weights
from flex.pool.decorators import set_aggregated_weights
import numpy as np
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
def get_clients_weights(client_model):
    return client_model["model"].model.cluster_centers_


@aggregate_weights
def aggregate(list_of_weights: list, model):
    weight_arr = np.concatenate(list_of_weights)
    model.fit(weight_arr)
    return model.cluster_centers_


@set_aggregated_weights
def set_aggregated_weights(server_flex_model, aggregated_weights, *args, **kwargs):
    server_flex_model["model"].model.cluster_centers_ = aggregated_weights


def train(client_model, client_data):
    print("Training model at client.")
    model = client_model["model"]
    X_data, y_data = client_data.to_numpy()
    model.fit(X_data, y_data)


def test_cluster(
    model_params, X_train, X_test, y_train, y_test, n_clients=3, n_rounds=5
):
    model = ClusterAnomaly(**model_params)
    flex_dataset = federate_data(n_clients, X_train, y_train)
    pool = FlexPool.client_server_pool(
        fed_dataset=flex_dataset,
        server_id="cluster_server",
        init_func=build_server_model,
        model=model,
    )

    for i in range(n_rounds):
        print(f"\nRunning round: {i}\n")
        pool.servers.map(copy_model_to_clients, pool.clients)
        pool.clients.map(train)
        pool.aggregators.map(get_clients_weights, pool.clients)
        pool.aggregators.map(aggregate,model = model)
        pool.aggregators.map(set_aggregated_weights, pool.servers)
    output_model = pool.servers._models["cluster_server"]["model"]
    output_model.evaluate(X_test, y_test)
    return output_model

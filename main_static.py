from flexanomalies.models import (
    AutoEncoder,
    ClusterAnomaly,
    IsolationForest,
    PCA_Anomaly,
)
from flexanomalies.utils.load_data import load_and_split_dot_mat, federate_data
from flexanomalies.pool.primitives_deepmodel import (
    build_server_model_ae,
    copy_model_to_clients_ae,
    train_ae,
    set_aggregated_weights_ae,
    weights_collector_ae
)
from flexanomalies.pool.aggregators_favg import aggregate_ae
from flexanomalies.pool.aggregators_cl import aggregate_cl
from flexanomalies.pool.aggregators_pca import aggregate_pca
from flexanomalies.pool.primitives_cluster import (
    build_server_model_cl,
    copy_model_to_clients_cl,
    train_cl,
    set_aggregated_weights_cl,
    get_clients_weights_cl,
)
from flexanomalies.pool.primitives_iforest import (
    build_server_model_if,
    copy_model_to_clients_if,
    train_if,
    aggregate_if,
    set_aggregated_weights_if,
    get_clients_weights_if,
)
from flexanomalies.pool.primitives_pca import (
    build_server_model_pca,
    copy_model_to_clients_pca,
    train_pca,
    set_aggregated_weights_pca,
    get_clients_weights_pca,
)
from flexanomalies.utils.save_results import save_experiments_results
from flex.pool import FlexPool
import os
import json


def test_autoencoder(
    model_params, X_train, X_test, y_train, y_test, n_clients=3, n_rounds=5
):
    model = AutoEncoder(**model_params)
    flex_dataset = federate_data(n_clients, X_train, y_train)
    pool = FlexPool.client_server_pool(
        fed_dataset=flex_dataset,
        server_id="autoencoder_server",
        init_func=build_server_model_ae,
        model=model,
    )

    for i in range(n_rounds):
        print(f"\nRunning round: {i}\n")
        pool.servers.map(copy_model_to_clients_ae, pool.clients)
        pool.clients.map(train_ae)
        pool.aggregators.map(weights_collector_ae, pool.clients)
        pool.aggregators.map(aggregate_ae)
        pool.aggregators.map(set_aggregated_weights_ae, pool.servers)
    output_model = pool.servers._models["autoencoder_server"]["model"]
    output_model.evaluate(X_test, y_test)
    return output_model


def test_cluster(
    model_params, X_train, X_test, y_train, y_test, n_clients=3, n_rounds=5
):
    model = ClusterAnomaly(**model_params)
    flex_dataset = federate_data(n_clients, X_train, y_train)
    pool = FlexPool.client_server_pool(
        fed_dataset=flex_dataset,
        server_id="cluster_server",
        init_func=build_server_model_cl,
        model=model,
    )

    for i in range(n_rounds):
        print(f"\nRunning round: {i}\n")
        pool.servers.map(copy_model_to_clients_cl, pool.clients)
        pool.clients.map(train_cl)
        pool.aggregators.map(get_clients_weights_cl, pool.clients)
        pool.aggregators.map(aggregate_cl, model=model)
        pool.aggregators.map(set_aggregated_weights_cl, pool.servers)
    output_model = pool.servers._models["cluster_server"]["model"]
    output_model.evaluate(X_test, y_test)
    return output_model


def test_iforest(
    model_params, X_train, X_test, y_train, y_test, n_clients=3, n_rounds=5
):
    model = IsolationForest(**model_params)
    flex_dataset = federate_data(n_clients, X_train, y_train)
    pool = FlexPool.client_server_pool(
        fed_dataset=flex_dataset,
        server_id="iforest_server",
        init_func=build_server_model_if,
        model=model,
    )

    pool.servers.map(copy_model_to_clients_if, pool.clients)
    pool.clients.map(train_if)
    pool.aggregators.map(get_clients_weights_if, pool.clients)
    pool.aggregators.map(aggregate_if)
    pool.aggregators.map(set_aggregated_weights_if, pool.servers)
    output_model = pool.servers._models["iforest_server"]["model"]
    output_model.evaluate(X_test, y_test)
    return output_model


def test_pca(model_params, X_train, X_test, y_train, y_test, n_clients=3, n_rounds=5):
    model = PCA_Anomaly(**model_params)
    flex_dataset = federate_data(n_clients, X_train, y_train)
    pool = FlexPool.client_server_pool(
        fed_dataset=flex_dataset,
        server_id="pca_server",
        init_func=build_server_model_pca,
        model=model,
    )

    for i in range(n_rounds):
        print(f"\nRunning round: {i}\n")
        pool.servers.map(copy_model_to_clients_pca, pool.clients)
        pool.clients.map(train_pca)
        pool.aggregators.map(get_clients_weights_pca, pool.clients)
        pool.aggregators.map(aggregate_pca)
        pool.aggregators.map(set_aggregated_weights_pca, pool.servers)
    output_model = pool.servers._models["pca_server"]["model"]
    output_model.evaluate(X_test, y_test)
    return output_model


model_tests = {
    "autoencoder": test_autoencoder,
    "cluster": test_cluster,
    "pca_anomaly": test_pca,
    "iforest": test_iforest,
}


def hub(
    json_file="",
    model_params="",
    model="",
    dataset="",
    split_size=0.3,
    n_clients=3,
    n_rounds=5,
    output_name="",
):
    if json_file:
        json_params = json.load(open(json_file, "r"))
        model = json_params["model"] if "model" in json_params else model
        dataset = json_params["dataset"] if "dataset" in json_params else dataset
        model_params = (
            json_params["params"]
            if "params" in json_params
            else json.loads(model_params)
        )
        split_size = (
            json_params["split_size"] if "split_size" in json_params else split_size
        )
        n_clients = (
            json_params["n_clients"] if "n_clients" in json_params else n_clients
        )
        n_rounds = json_params["n_rounds"] if "n_rounds" in json_params else n_rounds
        output_name = (
            json_params["output_name"]
            if "output_name" in json_params
            else output_name
            if output_name
            else json_file.split("/")[-1].split(".")[0]
        )
    elif not model_params or not model or not dataset:
        raise Exception(
            "Must provide json_file or `model_params`, `model`, `dataset` and `output_name`"
        )
    else:
        model_params = json.loads(model_params)

    X_train, X_test, y_train, y_test = load_and_split_dot_mat(dataset, split_size)
    output_model = model_tests[model](
        model_params,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_clients=n_clients,
        n_rounds=n_rounds,
    )
    save_experiments_results(
        model,
        output_model,
        output_name,
        model_params,
        dataset,
        n_clients,
        n_rounds,
        split_size,
    )


def save_experiments_results(
    model_name,
    model,
    output_name,
    model_params,
    dataset,
    n_clients,
    n_rounds,
    split_size,
):
    exp_result = {}
    exp_result["model_name"] = model_name
    exp_result["dataset"] = dataset
    exp_result["model_params"] = model_params
    exp_result["output_metrics"] = model.result_metrics_
    exp_result["n_clients"] = n_clients
    exp_result["n_rounds"] = n_rounds
    exp_result["split_size"] = split_size
    output_path = f"experiments/results/{model_name}/{output_name}"
    os.makedirs(output_path, exist_ok=True)
    json.dump(
        exp_result,
        open(f"{output_path}/results.json", "w"),
        indent=4,
        ensure_ascii=False,
    )
    model_path = f"{output_path}/model/"
    os.makedirs(model_path, exist_ok=True)
    model.save_model(model_path)

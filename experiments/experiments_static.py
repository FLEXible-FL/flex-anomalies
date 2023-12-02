import json
from utils.load_data import load_and_split_dot_mat
from experiments import test_autoencoder,test_cluster,test_pca
import os

model_tests = {"autoencoder": test_autoencoder, "cluster": test_cluster, "pca_anomaly":test_pca}


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
        raise "Must provide json_file or `model_params`, `model`, `dataset` and `output_name`"
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
    model_path = f'{output_path}/model/'
    os.makedirs(model_path, exist_ok=True)
    model.save_model(model_path)

import os
import json

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
    output_path = f"results/{model_name}/{output_name}"
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
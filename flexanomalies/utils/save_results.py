#Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
    """function to save the experiment results ( model parameters, model and experimental parameters)
    Args:
    ----
        model parameters, model and experimental parameters
    """
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
    model_path = f"{output_path}/model/"
    os.makedirs(model_path, exist_ok=True)
    model.save_model(model_path)

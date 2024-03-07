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


from flex.pool import init_server_model
from flex.model import FlexModel
from flex.pool import deploy_server_model
from flex.pool.decorators import collect_clients_weights
from flex.pool.decorators import set_aggregated_weights
from copy import deepcopy


@init_server_model
def build_server_model_pca(model):
    """
    Function to initialize the server model
    Args:
    ----
        model: A model initialized.
    Returns:
    -------
        FlexModel: A FlexModel that will be assigned to the server.

    """
    flex_model = FlexModel()
    flex_model["model"] = model
    return flex_model


@deploy_server_model
def copy_model_to_clients_pca(server_flex_model):
    """Function to deploy a TensorFlow model from the server to a client.

    The function will make a deepcopy for a TensorFlow model, as it needs
    a special method of copying. Also, it compiles the model for being able
    to train the model.

    This function uses the decorator @deploy_server_model to deploy the
    server_flex_model to the all the clients, so we only need to create
    the steps for 1 client.

    Args:
    ----
        server_flex_model (FlexModel): Server FlexModel

    Returns:
    -------
        FlexModel: The client's FlexModel
    """
    client_flex_model = deepcopy(server_flex_model)
    return client_flex_model


@collect_clients_weights
def get_clients_weights_pca(client_model):
    """Function that collect the model of the client's.

    Args:
    ----
        client_model (FlexModel): A client's FlexModel

    Returns:
    -------
           A client's FlexModel
    """
    return client_model["model"].model


@set_aggregated_weights
def set_aggregated_weights_pca(server_flex_model, aggregated_params, *args, **kwargs):
    """Function that replaces the parameters of the server model by the aggregated parameters.

    Args:
    ----
        server_flex_model (FlexModel): The server's FlexModel
        aggregated_params (np.array): An array with the aggregate parameters of the models.
    """
    server_flex_model["model"].model.components_ = aggregated_params["components"]
    server_flex_model["model"].model.n_components_ = aggregated_params["n_components"]
    server_flex_model["model"].model.explained_variance_ratio_ = aggregated_params[
        "ratio"
    ]


def train_pca(client_model, client_data):
    """Function of general purpose to train a model

    Args:
    ----
        client_model (FlexModel): client's FlexModel
        client_data (FedDataset): client's FedDataset

    """
    print("Training model at client.")
    model = client_model["model"]
    X_data, y_data = client_data.to_numpy()
    model.fit(X_data, y_data)

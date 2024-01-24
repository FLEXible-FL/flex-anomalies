from flex.pool import init_server_model
from flex.model import FlexModel
from flex.pool import deploy_server_model
from flex.pool.decorators import collect_clients_weights
from flex.pool.decorators import set_aggregated_weights
import numpy as np
from copy import deepcopy


@init_server_model
def build_server_model_cl(model):
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
def copy_model_to_clients_cl(server_flex_model):
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
def get_clients_weights_cl(client_model):
    """Function that collect the clusters centers of the client's.

    Args:
    ----
        client_flex_model (FlexModel): A client's FlexModel

    Returns:
    -------
        np.array: An array with all the clusters centers of the client's.
    """
    return client_model["model"].model.cluster_centers_


@set_aggregated_weights
def set_aggregated_weights_cl(server_flex_model, aggregated_weights, *args, **kwargs):
    """Function that replaces the clusters centers  of the server with the aggregated clusters centers  of the aggregator.

    Args:
    ----
        server_flex_model (FlexModel): The server's FlexModel
        aggregated_weights (np.array): An array with the aggregated
        clusters centers of the models.
    """
    server_flex_model["model"].model.cluster_centers_ = aggregated_weights


def train_cl(client_model, client_data):
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

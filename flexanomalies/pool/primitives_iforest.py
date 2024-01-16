from flex.pool import init_server_model
from flex.model import FlexModel
from flex.pool import deploy_server_model
from flex.pool.decorators import collect_clients_weights
from flex.pool.decorators import aggregate_weights
from flex.pool.decorators import set_aggregated_weights
from copy import deepcopy


@init_server_model
def build_server_model_if(model):
    """
    Function to initialize the server model
    """
    flex_model = FlexModel()
    flex_model["model"] = model
    return flex_model


@deploy_server_model
def copy_model_to_clients_if(server_flex_model):
    client_flex_model = deepcopy(server_flex_model)
    return client_flex_model

@collect_clients_weights
def get_clients_weights_if(client_model):
    return client_model["model"].model



@aggregate_weights
def aggregate_if(agg_model):
    return agg_model



@set_aggregated_weights
def set_aggregated_weights_if(server_flex_model, aggregated_weights, *args, **kwargs):
    server_flex_model["model"].model.estimators_ = [estimator for model in aggregated_weights for estimator in model.estimators_]
    server_flex_model["model"].model.n_estimators_ = len(server_flex_model['model'].model.estimators_)
    server_flex_model["model"].model._max_features = server_flex_model['model'].max_features
    server_flex_model["model"].model._max_samples = server_flex_model['model'].max_samples
    server_flex_model["model"].model.estimators_features_= aggregated_weights[0].estimators_features_
    server_flex_model['model'].model._decision_path_lengths = [des for model in aggregated_weights for des in  model._decision_path_lengths ]
    server_flex_model['model'].model._average_path_length_per_tree = [des for model in aggregated_weights for des in  model._average_path_length_per_tree ]
    server_flex_model['model'].model.offset_ = sum([model.offset_ for model in aggregated_weights])/len(aggregated_weights)
    

def train_if(client_model, client_data):
    print("Training model at client.")
    model = client_model["model"]
    X_data, y_data = client_data.to_numpy()
    model.fit(X_data, y_data)
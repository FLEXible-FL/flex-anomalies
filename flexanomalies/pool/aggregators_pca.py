from flex.pool.decorators import aggregate_weights
import numpy as np


@aggregate_weights
def aggregate_pca(agg_model):
    """Function to aggregate the components of the clients.

    Args:
        agg_model: List of Client Flex Model.

    Returns:
        Dict: Dictionary with the added parameters.
    """
    components = [model.components_ for model in agg_model]
    n_components = agg_model[0].n_components_
    ratio = [model.explained_variance_ratio_ for model in agg_model]
    return {
        "components": np.mean(np.array(components), axis=0),
        "n_components": n_components,
        "ratio": np.mean(np.array(ratio), axis=0),
    }

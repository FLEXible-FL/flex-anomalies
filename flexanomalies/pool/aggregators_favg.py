from flex.pool.decorators import aggregate_weights
import numpy as np


@aggregate_weights
def aggregate_ae(agg_model):
    """
        Function to aggregate the weights

    Args:
        agg_model : List of weights to aggregate.

    Returns:
        List: List with the aggregated weights.
    """
    return np.mean(np.array(agg_model, dtype=object), axis=0)

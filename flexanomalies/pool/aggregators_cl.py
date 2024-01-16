from flex.pool.decorators import aggregate_weights
import numpy as np


@aggregate_weights
def aggregate_cl(list_of_weights: list, model):
    """Function to aggregate the cluster centers of the clients.

    Args:
       list_of_weights : List of cluster centers.

    Returns:
        List: Aggregate cluster centers
    """

    weight_arr = np.concatenate(list_of_weights)
    model.fit(weight_arr)
    return model.cluster_centers_

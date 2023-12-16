from flex.pool.decorators import aggregate_weights
import numpy as np


@aggregate_weights
def aggregate_cl(list_of_weights: list, model):
    weight_arr = np.concatenate(list_of_weights)
    model.fit(weight_arr)
    return model.cluster_centers_

from flex.pool.decorators import aggregate_weights
import numpy as np



@aggregate_weights
def aggregate_stats(list_of_stast: list):
    return np.mean(list_of_stast)

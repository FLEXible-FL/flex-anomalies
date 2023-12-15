from flex.pool.decorators import aggregate_weights
import numpy as np


@aggregate_weights
def aggregate_pca(agg_model):
    components = [model.components_ for model in agg_model]
    n_components = agg_model[0].n_components_
    ratio = [model.explained_variance_ratio_ for model in agg_model]
    return {
        "components": np.mean(np.array(components), axis=0),
        "n_components": n_components,
        "ratio": np.mean(np.array(ratio), axis=0),
    }

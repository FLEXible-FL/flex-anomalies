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

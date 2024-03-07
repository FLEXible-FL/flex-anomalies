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

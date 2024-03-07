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


import numpy as np


def aggregate_stats_mean(list_of_stats):
    """
        Function to aggregate the stats

    Args:
        list_of_stats : List of stats to aggregate.

    Returns:
        List: List with the aggregated stats.
    """
    return np.mean(list_of_stats)


def aggregate_stats_min(list_of_stats):
    """
        Function to aggregate the stats

    Args:
        list_of_stats : List of stats to aggregate.

    Returns:
        List: List with the aggregated stats.
    """
    return np.min(list_of_stats)


def aggregate_stats_max(list_of_stats):
    """
       Function to aggregate the stats

    Args:
        list_of_stats : List of stats to aggregate.

    Returns:
        List: List with the aggregated stats.
    """
    return np.max(list_of_stats)

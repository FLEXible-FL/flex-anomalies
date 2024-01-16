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

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


import scipy.io
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from flex.data import FedDatasetConfig, FedDataDistribution, Dataset


def split_data(X, y, split_size=0.30):
    """function to generate train-test splits with sklearn

    Returns:
    -------
           sklearn train-test splits

    """
    return train_test_split(X, y, test_size=split_size, random_state=42)


def load_and_split_dot_mat(file_path, split_size=0.3):
    """Function that loads .mat datasets and generate train-test splits with sklearn

    Returns:
    -------
           sklearn train-test splits

    """
    mat = scipy.io.loadmat(file_path)
    data = {"X": mat["X"].tolist(), "y": mat["y"].tolist()}

    df = pd.DataFrame.from_dict(data)
    df = df.sample(frac=1).reset_index(drop=True)
    X = np.array(df["X"].tolist())
    y = np.array(df["y"].tolist())
    return split_data(X, y, split_size)


def load_and_split_csv(file_path, input_dim, split_size=0.3):
    """Function that loads csv datasets and generate train-test splits with sklearn.
       Make sure labels are at the end
    Args:
    ----
            input_dim: number of attributes
    Returns:
    -------
            sklearn train-test splits

    """
    df = pd.read_csv(file_path)
    X = np.array(df.iloc[:, :(input_dim)])
    y = np.array(df.iloc[:, -1])
    return split_data(X, y, split_size)


def federate_data(n_clients, x, y):
    """function to federate a centralized dataset using FLEXible with FedDatasetConfig. Review FLEXible
    Args:
    ----
           n_clients: number of clients
    Returns:
    -------
            flex_dataset
    """
    data = Dataset.from_array(x, y)
    config = FedDatasetConfig(seed=0)
    config.n_nodes = n_clients
    config.replacement = False  # ensure that clients do not share any data
    config.node_ids = ["client" + str(i + 1) for i in range(n_clients)]  # Optional
    flex_dataset = FedDataDistribution.from_config(centralized_data=data, config=config)
    return flex_dataset

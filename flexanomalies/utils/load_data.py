import scipy.io
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from flex.data import FedDatasetConfig, FedDataDistribution, Dataset


def split_data(X, y, split_size=0.30):
    # Generate train-test splits
    return train_test_split(X, y, test_size=split_size, random_state=42)


def load_and_split_dot_mat(file_path, split_size=0.3):
    mat = scipy.io.loadmat(file_path)
    data = {"X": mat["X"].tolist(), "y": mat["y"].tolist()}

    df = pd.DataFrame.from_dict(data)
    df = df.sample(frac=1).reset_index(drop=True)
    X = np.array(df["X"].tolist())
    y = np.array(df["y"].tolist())
    return split_data(X, y, split_size)


def load_and_split_csv(file_path, input_dim, split_size=0.3):
    df = pd.read_csv(file_path)
    X = np.array(df.iloc[:, :(input_dim)])
    y = np.array(df.iloc[:, -1])

    return split_data(X, y, split_size)


def federate_data(n_clients, x, y):
    data = Dataset.from_array(x, y)
    config = FedDatasetConfig(seed=0)
    config.n_nodes = n_clients
    config.replacement = False  # ensure that clients do not share any data
    config.node_ids = ["client" + str(i + 1) for i in range(n_clients)]  # Optional
    flex_dataset = FedDataDistribution.from_config(centralized_data=data, config=config)
    return flex_dataset

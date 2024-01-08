import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd


def create_windows(w_size, n_pred, X_train, X_test, l_train, l_test):
    X_train_windows = []
    y_train_windows = []
    X_test_windows = []
    y_test_windows = []
    l_test_windows = []

    for i in range(0, len(X_train), n_pred):
        temp_xtrain = X_train[i : w_size + i, :]
        temp_ytrain = X_train[w_size + i : w_size + i + n_pred, :]
        if len(temp_xtrain) < w_size or len(temp_ytrain) < n_pred:
            break
        X_train_windows.append(temp_xtrain)

        y_train_windows.append(temp_ytrain)

    for i in range(0, len(X_test), n_pred):

        temp_xtest = X_test[i : w_size + i, :]
        temp_ytest = X_test[w_size + i : w_size + i + n_pred, :]
        temp_ltest = l_test[w_size + i : w_size + i + n_pred]
        if (
            len(temp_xtest) < w_size
            or len(temp_ytest) < n_pred
            or len(temp_ltest) < n_pred
        ):
            break

        X_test_windows.append(temp_xtest)

        y_test_windows.append(temp_ytest)

        l_test_windows.extend(temp_ltest)

    return (
        np.array(X_train_windows),
        np.array(y_train_windows),
        np.array(X_test_windows),
        np.array(y_test_windows),
        np.array(l_test_windows),
    )


def scaling(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return res


def impute_lost_values(
    df, feature_to_impute, n_neighbors=5, metric="nan_euclidean", weights="uniform"
):

    # Building the model
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric)

    # Fit the model and impute missing values
    imputer.fit(df[[feature_to_impute]])
    df[feature_to_impute] = imputer.transform(df[[feature_to_impute]]).ravel()

    print(
        " Missing values in "
        + feature_to_impute
        + str(df[feature_to_impute].isnull().sum())
    )


def rolling_mean(df, feature, windows=2, min_periods=1):
    df["rolling_mean"] = df.rolling(window=windows, min_periods=min_periods)[
        feature
    ].mean()
    return df

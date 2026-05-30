import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def load_data():
    """
    Load Boston Housing dataset
    """

    data_url = "http://lib.stat.cmu.edu/datasets/boston"

    raw_df = pd.read_csv(
        data_url,
        sep=r"\s+",
        skiprows=22,
        header=None
    )

    data = np.hstack([
        raw_df.values[::2, :],
        raw_df.values[1::2, :2]
    ])

    target = raw_df.values[1::2, 2]

    feature_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT"
    ]

    df = pd.DataFrame(
        data,
        columns=feature_names
    )

    df["MEDV"] = target

    return df


def preprocess_data(df):

    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    return mse


def cross_validate_model(model, X_train, y_train):

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=5
    )

    mse_scores = -scores

    return mse_scores.mean()

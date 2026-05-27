"""
Shared ML workflow helpers — g25ai1025 — ML Ops Assignment 1.

Keeps experiments model-agnostic: pass any sklearn regression estimator into the same pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data() -> pd.DataFrame:
    """Fetch Boston Housing from CMU mirror and return a labelled DataFrame (target: MEDV)."""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None, engine="python")
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
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
        "LSTAT",
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df["MEDV"] = target
    return df


def preprocess_data(
    df: pd.DataFrame,
    target_column: str = "MEDV",
    test_size: float = 0.2,
    random_state: int = 1025,
    scale_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    """
    Split features / target; optionally standardize X (useful for Kernel Ridge, etc.).

    Returns (X_train, X_test, y_train, y_test, scaler_or_none).
    """
    if target_column not in df.columns:
        raise ValueError(f"Missing target column {target_column!r} in DataFrame.")

    feature_cols = [c for c in df.columns if c != target_column]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    scaler: StandardScaler | None = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """Fit any sklearn regressor with a standard fit API."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Mean squared error on the held-out test partition."""
    y_pred = model.predict(X_test)
    return float(mean_squared_error(y_test, y_pred))


def run_experiment(
    model,
    model_label: str,
    *,
    scale_features: bool = False,
    test_size: float = 0.2,
    random_state: int = 1025,
) -> dict:
    """
    End-to-end: load → preprocess → train → test MSE.

    Prints a clear banner for CI / reports (g25ai1025).
    """
    df = load_data()
    X_train, X_test, y_train, y_test, _scaler = preprocess_data(
        df,
        target_column="MEDV",
        test_size=test_size,
        random_state=random_state,
        scale_features=scale_features,
    )
    trained = train_model(model, X_train, y_train)
    mse = evaluate_model(trained, X_test, y_test)

    print(f"student=g25ai1025 model={model_label} average_mse_test={mse:.6f}")
    return {
        "model": model_label,
        "mse_test": mse,
        "random_state": random_state,
    }

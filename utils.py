"""
utils.py — Shared utility functions for the Jena Climate RNN Activity.
These are complete (no TODOs). Import them in each step file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load the climate CSV, parse datetime, and set it as the index.

    Args:
        path: Absolute or relative path to climate_sample.csv.

    Returns:
        DataFrame with DatetimeIndex and columns T (degC), p (mbar), rh (%).
    """
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_series(dates, values, title: str = "Time Series", xlabel: str = "Time",
                ylabel: str = "Value", figsize=(12, 4)) -> None:
    """Plot a single time series line chart.

    Args:
        dates:  Array-like of datetime values (x-axis).
        values: Array-like of numeric values (y-axis).
        title:  Chart title.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        figsize: Matplotlib figure size tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, values, linewidth=0.8, color="#4c9be8")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions(actual: np.ndarray, predicted: np.ndarray,
                     title: str = "Actual vs Predicted Temperature (°C)",
                     figsize=(12, 5)) -> None:
    """Overlay actual and predicted value arrays on one chart.

    Args:
        actual:    Ground-truth values (inverse-transformed, in °C).
        predicted: Model predictions (inverse-transformed, in °C).
        title:     Chart title.
        figsize:   Matplotlib figure size tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(actual,    label="Actual",    linewidth=1.0, color="#4c9be8")
    ax.plot(predicted, label="Predicted", linewidth=1.0, color="#e87b4c", alpha=0.85)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    RMSE = sqrt( (1/n) * sum( (y_pred_i - y_true_i)^2 ) )

    Args:
        y_true: Ground-truth values, shape (n,).
        y_pred: Predicted values, shape (n,).

    Returns:
        Scalar RMSE value (same units as the target, e.g. °C).
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ---------------------------------------------------------------------------
# Inverse transform
# ---------------------------------------------------------------------------

def inverse_transform_predictions(scaler, preds: np.ndarray,
                                   n_features: int = 3,
                                   target_col: int = 0) -> np.ndarray:
    """Undo MinMaxScaler normalisation on the temperature prediction column.

    Because the scaler was fit on all 3 features together, we must reconstruct
    a full (n, 3) array with dummy values for the non-target columns, call
    inverse_transform, then extract the target column.

    Args:
        scaler:     Fitted MinMaxScaler instance.
        preds:      Predictions array, shape (n, 1) or (n,).
        n_features: Total number of features the scaler was fit on (default 3).
        target_col: Column index of the target feature (0 = T degC).

    Returns:
        1-D numpy array of predictions in original °C units.
    """
    preds = np.array(preds).flatten()
    dummy = np.zeros((len(preds), n_features))
    dummy[:, target_col] = preds
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, target_col]

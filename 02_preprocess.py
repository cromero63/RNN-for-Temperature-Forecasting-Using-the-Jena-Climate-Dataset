"""
02_preprocess.py — Step 2: Normalize Data + Create Sliding Window Sequences
===========================================================================
Connection to Final Project:
  This step maps to the "Data Preprocessing" section of your report.
  In the project you will normalize all 14 Jena features; here we practice
  on 3 features with the same workflow.

Theory — Why Normalize?
  Neural networks learn by computing gradients. If one feature (e.g. pressure
  ~1013 mbar) has values 100× larger than another (e.g. temperature ~10°C),
  the gradient update will be dominated by the large-magnitude feature.
  MinMax scaling maps every feature to [0, 1], giving equal gradient influence.

  MinMax formula:  x' = (x - x_min) / (x_max - x_min)

  CRITICAL — fit the scaler on TRAINING data only:
  If you fit on the full dataset, the scaler "sees" future data, which leaks
  information about the test set into training. This inflates metrics and
  would be cheating in a real forecasting system.

Theory — Sliding Window for Time Series:
  Unlike text (integer word IDs), climate data is continuous floats.
  We group consecutive timesteps into windows:

    X[t] = rows t … t+119   shape (120, 3)   ← 20 hours of 3 features
    y[t] = T (degC) at row t+120             ← next temperature to predict

  Sliding the window by 1 step at a time creates many (X, y) training pairs.

Run:  python 02_preprocess.py
Expected output (approximate):
  X_train shape: (1280, 120, 3)
  y_train shape: (1280,)
  X_test  shape: (320, 120, 3)
  y_test  shape: (320,)
"""

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils import load_data

DATA_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "climate_sample.csv")

WINDOW_SIZE = 120    # 120 timesteps × 10 min = 20 hours of history
FEATURES    = ["T (degC)", "p (mbar)", "rh (%)"]
TRAIN_FRAC  = 0.80   # 80 % train, 20 % test (chronological split)

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
df = load_data(DATA_PATH)
data = df[FEATURES].values          # numpy array, shape (2000, 3)

# ===========================================================================
# TODO 1 ─ Chronological train/test split
# ===========================================================================
# Compute split_idx = int(len(data) * TRAIN_FRAC)
# Then slice:  train_data = data[:split_idx]
#              test_data  = data[split_idx:]
#
# Do NOT shuffle — time order must be preserved for time-series forecasting.
#
# YOUR CODE HERE:
split_idx  = None   # replace
train_data = None   # replace
test_data  = None   # replace


# ===========================================================================
# TODO 2 ─ Fit MinMaxScaler on training data ONLY, then transform both splits
# ===========================================================================
# scaler = MinMaxScaler()
# train_scaled = scaler.fit_transform(train_data)
# test_scaled  = scaler.transform(test_data)
#
# ⚠️ NEVER call fit_transform on test_data — that would leak future statistics!
#
# YOUR CODE HERE:
scaler       = None  # replace
train_scaled = None  # replace
test_scaled  = None  # replace


# ===========================================================================
# TODO 3 ─ Build sliding window sequences
# ===========================================================================
# Implement the helper function below, then call it for both splits.
#
# def make_sequences(data_array, window_size):
#     X, y = [], []
#     for i in range(len(data_array) - window_size):
#         X.append(data_array[i : i + window_size])        # shape (window_size, 3)
#         y.append(data_array[i + window_size, 0])         # col 0 = T (degC)
#     return np.array(X), np.array(y)
#
# X_train, y_train = make_sequences(train_scaled, WINDOW_SIZE)
# X_test,  y_test  = make_sequences(test_scaled,  WINDOW_SIZE)
#
# YOUR CODE HERE:
X_train = y_train = X_test = y_test = None  # replace


# ===========================================================================
# TODO 4 ─ Print shapes
# ===========================================================================
# Expected (approximate):
#   X_train: (1480, 120, 3)   y_train: (1480,)
#   X_test:  (280,  120, 3)   y_test:  (280,)
# (exact numbers depend on split_idx and window; numbers in docstring are approx)
#
# YOUR CODE HERE:


# ---------------------------------------------------------------------------
# Save for use in later steps (optional convenience)
# ---------------------------------------------------------------------------
# Uncomment to persist arrays to disk so 04_train_evaluate.py can load them:
# np.save("X_train.npy", X_train)
# np.save("y_train.npy", y_train)
# np.save("X_test.npy",  X_test)
# np.save("y_test.npy",  y_test)
# import joblib; joblib.dump(scaler, "scaler.pkl")

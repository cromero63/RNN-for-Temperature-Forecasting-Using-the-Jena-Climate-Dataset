"""
05_backend.py — Step 5: Backend Forecasting Class
==================================================
Connection to Final Project:
  Steps 5–6 are optional extensions. If you implement them, document the
  class design in your report's "Deployment" section.

Theory — Why normalize input?
  The LSTM was trained on MinMax-scaled data in [0, 1]. If we feed raw values
  (e.g. temperature ~10 °C, pressure ~1013 mbar) directly to the model,
  it will produce garbage predictions because the weights were tuned for
  scaled inputs. We MUST apply the SAME scaler (with the SAME fit parameters)
  before calling model inference.
"""

import os
import numpy as np
import joblib

import torch
import torch.nn as nn

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "climate_lstm.pt")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

WINDOW_SIZE = 120
N_FEATURES  = 3


# ---------------------------------------------------------------------------
# Model definition (mirrors 04_train_evaluate.py)
# ---------------------------------------------------------------------------
class ClimateLSTM(nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.lstm1   = nn.LSTM(n_features, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2   = nn.LSTM(64, 32, batch_first=True)
        self.fc      = nn.Linear(32, 1)

    def forward(self, x):
        out, _      = self.lstm1(x)
        out         = self.dropout(out)
        out, (h, _) = self.lstm2(out)
        return self.fc(h.squeeze(0))


# ---------------------------------------------------------------------------
# Forecaster class
# ---------------------------------------------------------------------------
class ClimateForecaster:
    def __init__(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        self.ready   = False
        self._model  = None
        self._scaler = None
        self._load(model_path, scaler_path)

    def _load(self, model_path: str, scaler_path: str):
        """Load weights and scaler from disk."""
        if not os.path.exists(model_path):
            print(f"WARNING: model not found at {model_path}. Train Step 4 first.")
            return
        if not os.path.exists(scaler_path):
            print(f"WARNING: scaler not found at {scaler_path}.")
            return

        self._model = ClimateLSTM()
        self._model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self._model.eval()

        self._scaler = joblib.load(scaler_path)
        self.ready   = True

    def forecast(self, window: np.ndarray) -> float:
        """
        Accepts a raw (unscaled) window of shape (WINDOW_SIZE, N_FEATURES)
        and returns the predicted temperature in °C.
        """
        if not self.ready:
            raise RuntimeError("Forecaster is not ready — model or scaler missing.")
        if window.shape != (WINDOW_SIZE, N_FEATURES):
            raise ValueError(
                f"Expected window shape ({WINDOW_SIZE}, {N_FEATURES}), got {window.shape}"
            )

        # 1. Normalize with the SAME scaler used in training
        window_scaled = self._scaler.transform(window)                          # (120, 3)

        # 2. Convert to tensor and add batch dimension
        x = torch.tensor(window_scaled[np.newaxis, ...], dtype=torch.float32)  # (1, 120, 3)

        # 3. Predict (still scaled)
        with torch.no_grad():
            pred_scaled = self._model(x).item()

        # 4. Inverse transform to Celsius
        dummy       = np.zeros((1, N_FEATURES))
        dummy[0, 0] = pred_scaled
        return round(float(self._scaler.inverse_transform(dummy)[0, 0]), 4)

    def model_info(self) -> dict:
        """Returns metadata about the loaded model."""
        if not self.ready:
            raise RuntimeError("Forecaster is not ready — model or scaler missing.")
        return {
            "architecture": str(self._model),
            "total_params": sum(p.numel() for p in self._model.parameters()),
            "window_size":  WINDOW_SIZE,
            "n_features":   N_FEATURES,
            "features":     ["T (degC)", "p (mbar)", "rh (%)"],
        }
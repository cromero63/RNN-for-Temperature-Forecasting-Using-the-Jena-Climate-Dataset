"""
05_api_backend.py — Step 5: FastAPI Backend for Temperature Forecasting
=======================================================================
Connection to Final Project:
  Steps 5–6 are optional extensions. If you implement them, document the API
  design in your report's "Deployment" section.

Theory — Why normalize API input?
  The LSTM was trained on MinMax-scaled data in [0, 1]. If we feed raw values
  (e.g. temperature ~10 °C, pressure ~1013 mbar) directly to the model,
  it will produce garbage predictions because the weights were tuned for
  scaled inputs. We MUST apply the SAME scaler (with the SAME fit parameters)
  before calling model inference.

Endpoints:
  POST /forecast   — accepts 120 × 3 window, returns predicted temperature
  GET  /model/info — returns model metadata (input shape, parameters, etc.)

Run:
  uvicorn 05_api_backend:app --reload
  Then open http://127.0.0.1:8000/docs for interactive docs.
"""

import os
import numpy as np
import joblib
from contextlib import asynccontextmanager
from typing import List

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

# Paths — adjust if you saved model/scaler in a different location
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "climate_lstm.pt")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

WINDOW_SIZE = 120
N_FEATURES  = 3

# ---------------------------------------------------------------------------
# Re-define the same model class used in 04_train_evaluate.py
# (In a real project this would live in a shared models.py module)
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
# Globals — loaded once at startup
# ---------------------------------------------------------------------------
model  = None
scaler = None


# ===========================================================================
# TODO 1 ─ Define the Pydantic request model
# ===========================================================================
# The client sends a JSON body with a 2-D list: 120 rows × 3 columns.
# Each inner list is [T (degC), p (mbar), rh (%)] for one timestep.
#
# class ForecastRequest(BaseModel):
#     window: List[List[float]]   # shape: (120, 3)
#
# YOUR CODE HERE:


# ===========================================================================
# TODO 2 ─ Load model and scaler on startup using lifespan
# ===========================================================================
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global model, scaler
#     device = torch.device('cpu')
#     if os.path.exists(MODEL_PATH):
#         model = ClimateLSTM()
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#         model.eval()
#         print(f"Model loaded from {MODEL_PATH}")
#     else:
#         print(f"WARNING: model not found at {MODEL_PATH}. Train Step 4 first.")
#     if os.path.exists(SCALER_PATH):
#         scaler = joblib.load(SCALER_PATH)
#         print(f"Scaler loaded from {SCALER_PATH}")
#     else:
#         print(f"WARNING: scaler not found at {SCALER_PATH}.")
#     yield   # ← application runs here
#
# app = FastAPI(title="Jena Climate Forecast API", lifespan=lifespan)
#
# YOUR CODE HERE:
app = FastAPI(title="Jena Climate Forecast API")   # placeholder — replace with lifespan version


# ===========================================================================
# TODO 3 ─ Add CORS middleware
# ===========================================================================
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# YOUR CODE HERE:


# ===========================================================================
# TODO 4 ─ POST /forecast endpoint
# ===========================================================================
# @app.post("/forecast")
# async def forecast(request: ForecastRequest):
#     if model is None or scaler is None:
#         raise HTTPException(status_code=503, detail="Model or scaler not loaded.")
#
#     window = np.array(request.window, dtype=np.float32)   # (120, 3)
#     if window.shape != (WINDOW_SIZE, N_FEATURES):
#         raise HTTPException(
#             status_code=422,
#             detail=f"Expected window shape ({WINDOW_SIZE}, {N_FEATURES}), got {window.shape}"
#         )
#
#     # 1. Normalize with the SAME scaler used in training
#     window_scaled = scaler.transform(window)              # (120, 3)
#
#     # 2. Convert to tensor and add batch dimension
#     x = torch.tensor(window_scaled[np.newaxis, ...], dtype=torch.float32)   # (1, 120, 3)
#
#     # 3. Predict (still scaled)
#     model.eval()
#     with torch.no_grad():
#         pred_scaled = model(x).item()                     # scalar float in [0, 1]
#
#     # 4. Inverse transform to Celsius
#     dummy = np.zeros((1, N_FEATURES))
#     dummy[0, 0] = pred_scaled
#     pred_celsius = scaler.inverse_transform(dummy)[0, 0]
#
#     return {"predicted_temperature_celsius": round(float(pred_celsius), 4)}
#
# YOUR CODE HERE:


# ===========================================================================
# TODO 5 ─ GET /model/info endpoint
# ===========================================================================
# @app.get("/model/info")
# async def model_info():
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded.")
#     total_params = sum(p.numel() for p in model.parameters())
#     return {
#         "architecture":  str(model),
#         "total_params":  total_params,
#         "window_size":   WINDOW_SIZE,
#         "n_features":    N_FEATURES,
#         "features":      ["T (degC)", "p (mbar)", "rh (%)"],
#     }
#
# YOUR CODE HERE:

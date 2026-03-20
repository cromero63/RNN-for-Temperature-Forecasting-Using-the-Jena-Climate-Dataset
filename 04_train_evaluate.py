"""
04_train_evaluate.py — Step 4: Train the Model, Evaluate with RMSE, Plot Results
==================================================================================
Connection to Final Project:
  This step maps to the "Training & Visualization" deliverable in your project.
  You will produce the same plots (loss curve, actual vs predicted) for the
  full 14-feature Jena dataset. The RMSE figure goes directly in your report.

Theory — RMSE:
  RMSE = √( (1/n) Σ (ŷᵢ − yᵢ)² )

  • Same units as the target (°C) → interpretable
  • Example: RMSE = 1.5 °C means the model is off by ~1.5 °C on average

Run:  python 04_train_evaluate.py
  (Runs 20 epochs; expect ~30–60 s on CPU. Loss should decrease.)
"""

import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from utils import (load_data, plot_series, compute_rmse,
                   inverse_transform_predictions, plot_predictions)

DATA_PATH   = os.path.join(os.path.dirname(__file__), "sample_data", "climate_sample.csv")
WINDOW_SIZE = 120
FEATURES    = ["T (degC)", "p (mbar)", "rh (%)"]
TRAIN_FRAC  = 0.80
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 0.001

# ---------------------------------------------------------------------------
# Reproduce preprocessing from Step 2 (self-contained for easy running)
# ---------------------------------------------------------------------------
df   = load_data(DATA_PATH)
data = df[FEATURES].values

split_idx  = int(len(data) * TRAIN_FRAC)
train_data = data[:split_idx]
test_data  = data[split_idx:]

scaler       = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled  = scaler.transform(test_data)

def make_sequences(arr, ws):
    X, y = [], []
    for i in range(len(arr) - ws):
        X.append(arr[i : i + ws])
        y.append(arr[i + ws, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = make_sequences(train_scaled, WINDOW_SIZE)
X_test,  y_test  = make_sequences(test_scaled,  WINDOW_SIZE)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)   # (N, 1) for MSELoss
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test).unsqueeze(1)

# ---------------------------------------------------------------------------
# Reproduce model from Step 3
# ---------------------------------------------------------------------------
class ClimateLSTM(nn.Module):
    def __init__(self, n_features=len(FEATURES)):
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

model     = ClimateLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# DataLoader for batching
train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ===========================================================================
# TODO 1 ─ Train the model (manual training loop)
# ===========================================================================
# Run EPOCHS training passes over the data.
# For each epoch:
#   a) Set model to training mode: model.train()
#   b) Loop over batches from train_dl
#   c) Zero gradients:  optimizer.zero_grad()
#   d) Forward pass:    preds = model(X_batch)
#   e) Compute loss:    loss  = criterion(preds, y_batch)
#   f) Backward pass:   loss.backward()
#   g) Update weights:  optimizer.step()
#   h) Accumulate epoch loss for history tracking
#
# Store average loss per epoch in a list called `history_loss`.
history_loss = [] 
for epoch in range(EPOCHS):
  model.train()
  epoch_loss = 0.0
  for X_batch, y_batch in train_dl:
    optimizer.zero_grad()
    preds = model(X_batch)
    loss  = criterion(preds, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item() * len(y_batch)
  avg_loss = epoch_loss / len(train_ds)
  history_loss.append(avg_loss)
  print(f"Epoch {epoch+1:2d}/{EPOCHS}  loss: {avg_loss:.6f}")


# ===========================================================================
# TODO 2 ─ Plot training loss curve
# ===========================================================================
# Use plot_series() from utils to visualise loss over epochs.
#
plot_series(
    dates  = range(1, EPOCHS + 1),
    values = history_loss,
    title  = "Training Loss (MSE) Over Epochs",
    xlabel = "Epoch",
    ylabel = "MSE Loss",
)


# ===========================================================================
# TODO 3 ─ Generate predictions on the test set
# ===========================================================================
# In PyTorch, disable gradient computation during inference:

preds_scaled = None
model.eval()
with torch.no_grad():
    preds_scaled = model(X_test_t)   # (N_test, 1)
preds_scaled = preds_scaled.numpy()  # convert to numpy for inverse transform


# ===========================================================================
# TODO 4 ─ Inverse transform predictions back to Celsius
# ===========================================================================
# Use utils.inverse_transform_predictions() — it reconstructs the dummy
# full-feature array so the scaler can undo normalisation correctly.
preds_celsius = inverse_transform_predictions(scaler, preds_scaled)
dummy_y = np.zeros((len(y_test), 3))
dummy_y[:, 0] = y_test
actual_celsius = scaler.inverse_transform(dummy_y)[:, 0]


# ===========================================================================
# TODO 5 ─ Compute RMSE
# ===========================================================================
rmse = compute_rmse(actual_celsius, preds_celsius)
print(f"\nTest RMSE: {rmse:.4f} °C")



# ===========================================================================
# TODO 6 ─ Plot actual vs predicted temperatures
# ===========================================================================
plot_predictions(actual_celsius, preds_celsius)


# ---------------------------------------------------------------------------
# Optional: save model and scaler for use in Steps 5–6
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), "climate_lstm.pt")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")

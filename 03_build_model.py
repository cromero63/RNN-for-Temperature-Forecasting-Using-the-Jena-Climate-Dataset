"""
03_build_model.py — Step 3: Build the LSTM Regression Model (PyTorch)
=======================================================================
Connection to Final Project:
  This step maps to the "Model Adaptation" section of your report.
  In the project you will adapt this model to accept 14 input features
  instead of 3. The architecture (stacked LSTM → Linear output) stays
  the same; only the input dimension changes.

Theory — Regression vs Classification with RNNs:
  ┌──────────────────────┬────────────────────────────────────────────────┐
  │                      │ Classification (text gen)  │ Regression (temp) │
  ├──────────────────────┼───────────────────────────┼───────────────────┤
  │ Output layer         │ nn.Linear(units, vocab)    │ nn.Linear(units,1)│
  │ Activation           │ softmax                    │ none (linear)     │
  │ Loss function        │ CrossEntropyLoss           │ MSELoss           │
  │ Output interpretation│ probability distribution   │ continuous value  │
  └──────────────────────┴───────────────────────────┴───────────────────┘

  MSE  = (1/n) Σ (ŷᵢ − yᵢ)²
  RMSE = √MSE   ← same units as °C, easy to interpret

Run:  python 03_build_model.py
Expected output: model printed with ~20 K–25 K parameters
"""

import torch
import torch.nn as nn

# Input dimensions — must match what 02_preprocess.py produces
WINDOW_SIZE = 120    # timesteps per sequence
N_FEATURES  = 3     # T (degC), p (mbar), rh (%)

# ===========================================================================
# TODO 1 ─ Define the model class
# ===========================================================================
# In PyTorch, a model is a class that inherits from nn.Module.
# You define the layers in __init__() and the forward pass in forward().
#
# Architecture:
#   Layer 1: nn.LSTM(N_FEATURES, 64, batch_first=True)
#              input dim = N_FEATURES, hidden dim = 64
#              Returns: (output, (h_n, c_n))
#              output shape: (batch, 120, 64)  — all timesteps
#   Layer 2: nn.Dropout(0.2)
#   Layer 3: nn.LSTM(64, 32, batch_first=True)
#              input dim = 64, hidden dim = 32
#              h_n shape: (1, batch, 32)  — final hidden state only
#   Layer 4: nn.Linear(32, 1)            ← regression: single continuous output
#
# Note: nn.Linear(32, 1) is the equivalent of Dense(1) with no activation —
# NOT Dense(vocab_size) — because we are predicting temperature, not a word.
#
# class ClimateLSTM(nn.Module):
#     def __init__(self, n_features=N_FEATURES):
#         super().__init__()
#         self.lstm1   = nn.LSTM(n_features, 64, batch_first=True)
#         self.dropout = nn.Dropout(0.2)
#         self.lstm2   = nn.LSTM(64, 32, batch_first=True)
#         self.fc      = nn.Linear(32, 1)
#
#     def forward(self, x):
#         # x: (batch, WINDOW_SIZE, N_FEATURES)
#         out, _      = self.lstm1(x)          # out: (batch, 120, 64)
#         out         = self.dropout(out)
#         out, (h, _) = self.lstm2(out)        # h:   (1, batch, 32)
#         return self.fc(h.squeeze(0))         # return: (batch, 1)
#
# YOUR CODE HERE:
# class ClimateLSTM(nn.Module):
#     ...


# ===========================================================================
# TODO 2 ─ Instantiate the model and print it
# ===========================================================================
# model = ClimateLSTM()
# total_params = sum(p.numel() for p in model.parameters())
# print(model)
# print(f"\nTotal parameters: {total_params:,}")
#
# YOUR CODE HERE:
model = None  # replace


# ===========================================================================
# TODO 3 ─ Define loss function and optimizer
# ===========================================================================
# Use:  criterion = nn.MSELoss()            (Mean Squared Error — standard for regression)
#       optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# Note: nn.MSELoss computes (1/n) Σ (ŷ - y)² — no need to call backward manually
# on the loss formula; PyTorch handles the gradient graph automatically.
#
# YOUR CODE HERE:


# ---------------------------------------------------------------------------
# Optional: verify a single forward pass with dummy data
# ---------------------------------------------------------------------------
# dummy_input = torch.randn(8, WINDOW_SIZE, N_FEATURES)  # batch of 8 sequences
# output = model(dummy_input)
# print(f"\nDummy forward pass — output shape: {output.shape}")  # expect (8, 1)

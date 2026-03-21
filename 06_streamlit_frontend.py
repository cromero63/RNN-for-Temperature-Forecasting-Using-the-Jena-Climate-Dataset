"""
06_streamlit_frontend.py — Step 6: Streamlit Forecast Dashboard
================================================================
Connection to Final Project:
  Steps 5–6 are optional extensions. If you implement them, include
  a screenshot of your dashboard in the project report.

Theory — Auto-Regressive Multi-Step Forecasting:
  Our model predicts ONE step ahead (T at t+120).
  To forecast multiple future steps you can use an auto-regressive loop:
    1. Feed window [t-119 … t] → get prediction p̂₁ at t+1
    2. Shift window: drop oldest row, append [p̂₁, pressure_guess, humidity_guess]
    3. Repeat → p̂₂, p̂₃, …

  This accumulates error with each step. More advanced models (seq2seq,
  Temporal Fusion Transformers) mitigate this — but the simple approach
  works surprisingly well for short horizons (1–2 hours).

Run:
  streamlit run 06_streamlit_frontend.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

from backend_05 import ClimateForecaster

DATA_PATH   = os.path.join(os.path.dirname(__file__), "sample_data", "climate_sample.csv")
WINDOW_SIZE = 120
FEATURES    = ["T (degC)", "p (mbar)", "rh (%)"]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Jena Climate Forecast", layout="wide")
st.title("🌡️ Jena Climate — LSTM Temperature Forecast")

@st.cache_resource
def load_forecaster():
    return ClimateForecaster()

forecaster = load_forecaster()

# ===========================================================================
# Sidebar: feature sliders
# ===========================================================================
st.sidebar.header("Settings")

st.sidebar.subheader("Last observed reading")
last_temp     = st.sidebar.slider("Temperature (°C)",  -10.0, 40.0, 10.0, 0.1)
last_pressure = st.sidebar.slider("Pressure (mbar)",   970.0, 1050.0, 1013.0, 0.5)
last_humidity = st.sidebar.slider("Humidity (%)",        0.0,  100.0, 70.0, 1.0)


# ===========================================================================
# Load sample data (or accept user-uploaded CSV)
# ===========================================================================
uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["datetime"], index_col="datetime")
else:
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")

st.subheader("Raw Climate Data (last 200 rows)")
st.line_chart(df[FEATURES].tail(200))


# ===========================================================================
# Build the 120-step input window from the last WINDOW_SIZE rows
# ===========================================================================
# Scale the data with MinMaxScaler (same workflow as 02_preprocess.py).
# For a real deployment you would load the saved scaler; here we re-fit on
# whatever data is available (acceptable for a demo, NOT for production).
data = df[FEATURES].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
window_raw = data[-WINDOW_SIZE:]   # shape (120, 3) — ClimateForecaster scales internally


# ===========================================================================
# Call forecaster and display the result
# ===========================================================================
pred_temp = None
if st.button("Get Forecast"):
    try:
        pred_temp = forecaster.forecast(window_raw)
        st.metric(
            label="Predicted Temperature (next 10-min step)",
            value=f"{pred_temp:.2f} °C",
        )
        st.caption("Note: single-step forecast. See theory for multi-step extension.")
    except Exception as e:
        st.error(f"Forecast failed: {e}")


# ===========================================================================
# Plot actual (last N rows) vs forecasted value with Plotly
# ===========================================================================
if pred_temp is not None:
    actual_temps = df["T (degC)"].values[-144:]
    forecast_idx = len(actual_temps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual_temps, mode='lines', name='Actual',
                             line=dict(color='#4c9be8')))
    fig.add_trace(go.Scatter(x=[forecast_idx], y=[pred_temp], mode='markers+text',
                             name='Forecast', marker=dict(size=12, color='#e87b4c'),
                             text=[f"{pred_temp:.1f}°C"], textposition="top center"))
    fig.update_layout(title="Actual Temperature + 1-Step Forecast",
                      xaxis_title="Timestep (10-min intervals)",
                      yaxis_title="Temperature (°C)")
    st.plotly_chart(fig, use_container_width=True)
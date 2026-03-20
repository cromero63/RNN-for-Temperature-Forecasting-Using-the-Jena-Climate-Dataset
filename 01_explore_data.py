"""
01_explore_data.py — Step 1: Load and Explore the Climate Dataset
=================================================================
Connection to Final Project:
  This step maps to the "Data Loading & Exploration" section of your report.
  In the project you will load the full 400 MB Jena CSV; here we practice
  on a 2,000-row synthetic sample with the same column names.

Learning Goals:
  - Load a time-series CSV with a datetime index
  - Inspect shape, dtypes, missing values
  - Visualize raw feature trends before any preprocessing

Run:  python 01_explore_data.py
"""

import os
import matplotlib.pyplot as plt

# Import shared utilities (complete — no TODOs in utils.py)
from utils import load_data, plot_series

# ---------------------------------------------------------------------------
# Path to sample data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "climate_sample.csv")

# ===========================================================================
# TODO 1 ─ Load the CSV
# ===========================================================================
# Use the load_data() function from utils.py.
# It parses the 'datetime' column and sets it as the DataFrame index.
#
#   df = load_data(DATA_PATH)
#
# YOUR CODE HERE:
df = load_data(DATA_PATH)


# ===========================================================================
# TODO 2 ─ Basic inspection
# ===========================================================================
# Print the following to the terminal:
#   a) df.shape          — (rows, columns)
#   b) df.dtypes         — column data types
#   c) df.head(10)       — first 10 rows
#   d) df.describe()     — summary statistics (min, max, mean, std, ...)
#
# YOUR CODE HERE:
print(df.shape)
print(df.dtypes)
print(df.head(10))
print(df.describe())


# ===========================================================================
# TODO 3 ─ Check for missing values
# ===========================================================================
# Print df.isnull().sum() to see if any column has NaN values.
# In the Jena project you may need to handle missing values; here there
# should be none, but it is good practice to always check.
#
# YOUR CODE HERE:
print(df.isnull().sum())


# ===========================================================================
# TODO 4 ─ Plot Temperature over time
# ===========================================================================
# Use utils.plot_series() to draw a line chart of temperature.
#
#   plot_series(
#       dates  = df.index,
#       values = df["T (degC)"],
#       title  = "Temperature Over Time",
#       ylabel = "Temperature (°C)"
#   )
#
# YOUR CODE HERE:
plot_series(
    dates  = df.index,
    values = df["T (degC)"],
    title  = "Temperature Over Time",
    ylabel = "Temperature (°C)"
)


# ===========================================================================
# TODO 5 ─ 3-panel subplot of all features
# ===========================================================================
# Create a figure with 3 stacked subplots, one per column:
#   Row 0: T (degC)
#   Row 1: p (mbar)
#   Row 2: rh (%)
#
# Starter skeleton (fill in the blanks):
#
#   fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
#   columns = ["T (degC)", "p (mbar)", "rh (%)"]
#   ylabels = ["Temp (°C)", "Pressure (mbar)", "Humidity (%)"]
#
#   for ax, col, ylabel in zip(axes, columns, ylabels):
#       ax.plot(df.index, df[col], linewidth=0.8)
#       ax.set_ylabel(ylabel)
#       ax.grid(True, alpha=0.3)
#
#   axes[0].set_title("Climate Features Over Time")
#   axes[2].set_xlabel("Datetime")
#   plt.tight_layout()
#   plt.show()
#
# YOUR CODE HERE:
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
columns = ["T (degC)", "p (mbar)", "rh (%)"]
ylabels = ["Temp (°C)", "Pressure (mbar)", "Humidity (%)"]

for ax, col, ylabel in zip(axes, columns, ylabels):
    ax.plot(df.index, df[col], linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

axes[0].set_title("Climate Features Over Time")
axes[2].set_xlabel("Datetime")
plt.tight_layout()
plt.show()

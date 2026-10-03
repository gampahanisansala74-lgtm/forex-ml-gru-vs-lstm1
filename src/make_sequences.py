"""
make_sequences.py — Convert scaled Close series into sliding-window
sequences for GRU/LSTM training.

Each sample: X = 60 past scaled closes, y = next scaled close.

Produces (in data/):
  X_train.npy, y_train.npy
  X_val.npy,   y_val.npy
  X_test.npy,  y_test.npy

Run from the project root:
    python src/make_sequences.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------
LOOKBACK = 60              # how many past minutes to feed the model
FEATURE_COL = "Close_scaled"

TRAIN_SCALED = os.path.join("data", "train_scaled.csv")
VAL_SCALED   = os.path.join("data", "val_scaled.csv")
TEST_SCALED  = os.path.join("data", "test_scaled.csv")

# ---------------------------------------------------------------------
# 2. Load scaled series
# ---------------------------------------------------------------------
print("[INFO] Loading scaled CSVs")
train = pd.read_csv(TRAIN_SCALED, index_col="Datetime", parse_dates=True)
val   = pd.read_csv(VAL_SCALED,   index_col="Datetime", parse_dates=True)
test  = pd.read_csv(TEST_SCALED,  index_col="Datetime", parse_dates=True)

# ---------------------------------------------------------------------
# 3. Sliding-window function
# ---------------------------------------------------------------------
def make_sequences(series: np.ndarray, lookback: int):
    """
    series: 1-D numpy array of scaled values
    returns X shape (N - lookback, lookback, 1), y shape (N - lookback,)
    """
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    X = np.array(X, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.array(y, dtype=np.float32)
    return X, y

# ---------------------------------------------------------------------
# 4. Build sequences per split (independently!)
# ---------------------------------------------------------------------
train_vals = train[FEATURE_COL].values
val_vals   = val[FEATURE_COL].values
test_vals  = test[FEATURE_COL].values

X_train, y_train = make_sequences(train_vals, LOOKBACK)
X_val,   y_val   = make_sequences(val_vals,   LOOKBACK)
X_test,  y_test  = make_sequences(test_vals,  LOOKBACK)

# ---------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------
print(f"\n[INFO] Lookback = {LOOKBACK}")
print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}     y_val:   {y_val.shape}")
print(f"  X_test:  {X_test.shape}    y_test:  {y_test.shape}")

# Sanity check — first train sample should match the first 60 values
print("\n[INFO] Sanity check on train sample 0:")
print(f"  X_train[0, :3, 0] (first 3 timesteps): {X_train[0, :3, 0]}")
print(f"  X_train[0, -3:, 0] (last 3 timesteps): {X_train[0, -3:, 0]}")
print(f"  y_train[0] (target):                  {y_train[0]}")

# ---------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------
np.save(os.path.join("data", "X_train.npy"), X_train)
np.save(os.path.join("data", "y_train.npy"), y_train)
np.save(os.path.join("data", "X_val.npy"),   X_val)
np.save(os.path.join("data", "y_val.npy"),   y_val)
np.save(os.path.join("data", "X_test.npy"),  X_test)
np.save(os.path.join("data", "y_test.npy"),  y_test)

print("\n[INFO] Saved 6 .npy files to data/")
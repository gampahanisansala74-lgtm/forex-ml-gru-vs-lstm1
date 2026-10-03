"""
scale_data.py — Fit MinMaxScaler on TRAIN ONLY, then transform val/test.

Outputs:
  data/train_scaled.csv   (column: Close_scaled)
  data/val_scaled.csv
  data/test_scaled.csv
  models/scaler.pkl       (pickled MinMaxScaler for inverse_transform later)

Run from the project root:
    python src/scale_data.py
"""

import os
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------
TRAIN_PATH        = os.path.join("data", "train.csv")
VAL_PATH          = os.path.join("data", "val.csv")
TEST_PATH         = os.path.join("data", "test.csv")

TRAIN_SCALED_PATH = os.path.join("data", "train_scaled.csv")
VAL_SCALED_PATH   = os.path.join("data", "val_scaled.csv")
TEST_SCALED_PATH  = os.path.join("data", "test_scaled.csv")
SCALER_PATH       = os.path.join("models", "scaler.pkl")

FEATURE = "Close"

# ---------------------------------------------------------------------
# 2. Load splits
# ---------------------------------------------------------------------
print("[INFO] Loading train/val/test CSVs")
train = pd.read_csv(TRAIN_PATH, index_col="Datetime", parse_dates=True)
val   = pd.read_csv(VAL_PATH,   index_col="Datetime", parse_dates=True)
test  = pd.read_csv(TEST_PATH,  index_col="Datetime", parse_dates=True)

print(f"  Train rows: {len(train):,}")
print(f"  Val   rows: {len(val):,}")
print(f"  Test  rows: {len(test):,}")

# ---------------------------------------------------------------------
# 3. Fit scaler on TRAIN ONLY
# ---------------------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train[[FEATURE]])

print(f"\n[INFO] Scaler fit on TRAIN '{FEATURE}' column")
print(f"  data_min_: {scaler.data_min_[0]:.6f}")
print(f"  data_max_: {scaler.data_max_[0]:.6f}")
print(f"  range_:    {scaler.data_max_[0] - scaler.data_min_[0]:.6f}")

# ---------------------------------------------------------------------
# 4. Transform all three splits using the SAME scaler
# ---------------------------------------------------------------------
train_scaled = pd.DataFrame(
    scaler.transform(train[[FEATURE]]),
    columns=[f"{FEATURE}_scaled"],
    index=train.index,
)
val_scaled = pd.DataFrame(
    scaler.transform(val[[FEATURE]]),
    columns=[f"{FEATURE}_scaled"],
    index=val.index,
)
test_scaled = pd.DataFrame(
    scaler.transform(test[[FEATURE]]),
    columns=[f"{FEATURE}_scaled"],
    index=test.index,
)

# ---------------------------------------------------------------------
# 5. Report scaling stats
# ---------------------------------------------------------------------
def _stats(name, d):
    s = d[f"{FEATURE}_scaled"]
    print(f"  {name:5s}  min={s.min():.4f}  max={s.max():.4f}  "
          f"mean={s.mean():.4f}  std={s.std():.4f}")

print("\n[INFO] Scaled value ranges:")
_stats("Train", train_scaled)
_stats("Val  ", val_scaled)
_stats("Test ", test_scaled)

# ---------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------
train_scaled.to_csv(TRAIN_SCALED_PATH)
val_scaled.to_csv(VAL_SCALED_PATH)
test_scaled.to_csv(TEST_SCALED_PATH)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"\n[INFO] Saved: {TRAIN_SCALED_PATH}")
print(f"[INFO] Saved: {VAL_SCALED_PATH}")
print(f"[INFO] Saved: {TEST_SCALED_PATH}")
print(f"[INFO] Saved scaler: {SCALER_PATH}")
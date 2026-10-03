"""
split_data.py — Chronological train/validation/test split.

Loads data/EURUSD_clean.csv, splits it into three consecutive
non-overlapping time blocks, and saves each as a separate CSV.

Run from the project root:
    python src/split_data.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------
CLEAN_PATH = os.path.join("data", "EURUSD_clean.csv")
TRAIN_PATH = os.path.join("data", "train.csv")
VAL_PATH   = os.path.join("data", "val.csv")
TEST_PATH  = os.path.join("data", "test.csv")

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC is implicitly 0.15 (the remainder)

# ---------------------------------------------------------------------
# 2. Load cleaned data
# ---------------------------------------------------------------------
print(f"[INFO] Loading: {CLEAN_PATH}")
df = pd.read_csv(CLEAN_PATH, index_col="Datetime", parse_dates=True)
n = len(df)
print(f"[INFO] Total rows: {n:,}")

# ---------------------------------------------------------------------
# 3. Compute chronological cut points
# ---------------------------------------------------------------------
train_end = int(n * TRAIN_FRAC)
val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

# ---------------------------------------------------------------------
# 4. Slice (order preserved — NO shuffling)
# ---------------------------------------------------------------------
train_df = df.iloc[:train_end]
val_df   = df.iloc[train_end:val_end]
test_df  = df.iloc[val_end:]

# ---------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------
def _report(name, d):
    print(
        f"  {name:5s}  rows={len(d):>6,}  "
        f"{d.index.min()}  ->  {d.index.max()}"
    )

print("\n[INFO] Split boundaries (chronological):")
_report("Train", train_df)
_report("Val  ", val_df)
_report("Test ", test_df)

# ---------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------
train_df.to_csv(TRAIN_PATH)
val_df.to_csv(VAL_PATH)
test_df.to_csv(TEST_PATH)
print(f"\n[INFO] Saved: {TRAIN_PATH}")
print(f"[INFO] Saved: {VAL_PATH}")
print(f"[INFO] Saved: {TEST_PATH}")
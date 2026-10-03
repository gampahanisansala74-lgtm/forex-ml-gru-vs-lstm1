"""
preprocess.py — Load, clean, and save the EUR/USD minute dataset.

Run from the project root:
    python src/preprocess.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------
RAW_PATH = os.path.join("data", "EURUSD1.csv")
CLEAN_PATH = os.path.join("data", "EURUSD_clean.csv")

# ---------------------------------------------------------------------
# 2. Load raw data (tab-separated, no header)
# ---------------------------------------------------------------------
print(f"[INFO] Loading raw data from: {RAW_PATH}")
df = pd.read_csv(
    RAW_PATH,
    sep="\t",
    header=None,
    names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
)

# ---------------------------------------------------------------------
# 3. Parse Datetime + set as index + sort
# ---------------------------------------------------------------------
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime").set_index("Datetime")

# ---------------------------------------------------------------------
# 4. Defensive cleaning
# ---------------------------------------------------------------------
before = len(df)
df = df.dropna()
after = len(df)
if before != after:
    print(f"[INFO] Dropped {before - after} rows with NaNs")

# ---------------------------------------------------------------------
# 5. Sanity check
# ---------------------------------------------------------------------
assert df.index.is_monotonic_increasing, "Index is not chronological!"
assert df.index.is_unique, "Duplicate timestamps found!"

print("\n[INFO] Final dataset summary")
print("-" * 40)
print(f"Rows:         {len(df):,}")
print(f"Columns:      {list(df.columns)}")
print(f"Date range:   {df.index.min()}  ->  {df.index.max()}")
print(f"Index freq:   median gap = {df.index.to_series().diff().median()}")
print("\nFirst 3 rows:")
print(df.head(3))
print("\nLast 3 rows:")
print(df.tail(3))

# ---------------------------------------------------------------------
# 6. Save cleaned data
# ---------------------------------------------------------------------
df.to_csv(CLEAN_PATH)
print(f"\n[INFO] Saved cleaned data to: {CLEAN_PATH}")
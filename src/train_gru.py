"""
train_gru.py — Build, train, and evaluate a GRU model on the scaled
EUR/USD minute-close sequences.

Outputs:
  models/gru_model.keras      (trained Keras model)
  reports/gru_history.csv     (per-epoch loss/MSE history)

Run from the project root:
    python src/train_gru.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ---------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------
tf.keras.utils.set_random_seed(42)

# ---------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------
LOOKBACK       = 60
GRU_UNITS      = 50
DROPOUT        = 0.2
BATCH_SIZE     = 32
EPOCHS         = 50
LEARNING_RATE  = 1e-3
PATIENCE       = 5

MODEL_PATH   = os.path.join("models", "gru_model.keras")
HISTORY_PATH = os.path.join("reports", "gru_history.csv")

# ---------------------------------------------------------------------
# 2. Load sequences
# ---------------------------------------------------------------------
print("[INFO] Loading sequences from data/*.npy")
X_train = np.load(os.path.join("data", "X_train.npy"))
y_train = np.load(os.path.join("data", "y_train.npy"))
X_val   = np.load(os.path.join("data", "X_val.npy"))
y_val   = np.load(os.path.join("data", "y_val.npy"))

print(f"  X_train: {X_train.shape}")
print(f"  X_val:   {X_val.shape}")

# ---------------------------------------------------------------------
# 3. Build model
# ---------------------------------------------------------------------
model = models.Sequential(
    [
        layers.Input(shape=(LOOKBACK, 1)),
        layers.GRU(GRU_UNITS, return_sequences=False),
        layers.Dropout(DROPOUT),
        layers.Dense(1),
    ],
    name="gru_forex",
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"],
)

model.summary()

# ---------------------------------------------------------------------
# 4. Callbacks
# ---------------------------------------------------------------------
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1,
)

# ---------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------
print("\n[INFO] Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=2,
)

# ---------------------------------------------------------------------
# 6. Save model + history
# ---------------------------------------------------------------------
model.save(MODEL_PATH)
print(f"\n[INFO] Model saved: {MODEL_PATH}")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(HISTORY_PATH, index=False)
print(f"[INFO] History saved: {HISTORY_PATH}")

# ---------------------------------------------------------------------
# 7. Final validation metrics
# ---------------------------------------------------------------------
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
print(f"\n[RESULT] Best val_loss (MSE): {val_loss:.6f}")
print(f"[RESULT] Best val_MAE:        {val_mae:.6f}")
print(f"[RESULT] Epochs trained:      {len(hist_df)}")
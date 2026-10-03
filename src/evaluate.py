"""
evaluate.py — Evaluate GRU and LSTM on the test set.

Loads both trained models + the saved scaler, predicts on the test set,
inverse-transforms back to real EUR/USD prices, computes RMSE / MAE / MAPE,
and saves comparison plots + a metrics CSV to reports/.

Run from the project root:
    python src/evaluate.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------
X_TEST_PATH = os.path.join("data", "X_test.npy")
y_TEST_PATH = os.path.join("data", "y_test.npy")

GRU_MODEL_PATH  = os.path.join("models", "gru_model.keras")
LSTM_MODEL_PATH = os.path.join("models", "lstm_model.keras")
SCALER_PATH     = os.path.join("models", "scaler.pkl")

GRU_PREDS_PATH   = os.path.join("reports", "gru_test_preds.csv")
LSTM_PREDS_PATH  = os.path.join("reports", "lstm_test_preds.csv")
METRICS_PATH     = os.path.join("reports", "model_metrics.csv")
PRED_PLOT_PATH   = os.path.join("reports", "predictions_vs_actual.png")
LOSS_PLOT_PATH   = os.path.join("reports", "loss_curves.png")

# ---------------------------------------------------------------------
# 2. Load test data + scaler
# ---------------------------------------------------------------------
print("[INFO] Loading test set + scaler")
X_test = np.load(X_TEST_PATH)
y_test = np.load(y_TEST_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------------------------------------------
# 3. Inverse-transform helper
# ---------------------------------------------------------------------
def inverse(vals):
    """Convert scaled [0,1] values back to real EUR/USD prices."""
    return scaler.inverse_transform(vals.reshape(-1, 1)).flatten()

y_true_real = inverse(y_test)
print(f"[INFO] Test samples: {len(y_true_real):,}")
print(f"[INFO] Test price range (real): "
      f"{y_true_real.min():.5f} .. {y_true_real.max():.5f}")

# ---------------------------------------------------------------------
# 4. Predict with both models
# ---------------------------------------------------------------------
print("\n[INFO] Predicting with GRU...")
gru_model = tf.keras.models.load_model(GRU_MODEL_PATH)
gru_pred_scaled = gru_model.predict(X_test, verbose=0).flatten()
gru_pred_real   = inverse(gru_pred_scaled)

print("[INFO] Predicting with LSTM...")
lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
lstm_pred_scaled = lstm_model.predict(X_test, verbose=0).flatten()
lstm_pred_real   = inverse(lstm_pred_scaled)

# ---------------------------------------------------------------------
# 5. Metrics in real price terms
# ---------------------------------------------------------------------
def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

gru_rmse,  gru_mae,  gru_mape  = metrics(y_true_real, gru_pred_real)
lstm_rmse, lstm_mae, lstm_mape = metrics(y_true_real, lstm_pred_real)

# Convert absolute errors to "pips" (1 pip = 0.0001 for EUR/USD)
PIP = 1e-4
gru_rmse_pips  = gru_rmse  / PIP
gru_mae_pips   = gru_mae   / PIP
lstm_rmse_pips = lstm_rmse / PIP
lstm_mae_pips  = lstm_mae  / PIP

print("\n[RESULTS] Test set (real EUR/USD prices)")
print("-" * 60)
print(f"{'Model':<6}  {'RMSE (pips)':>12}  {'MAE (pips)':>12}  {'MAPE (%)':>10}")
print(f"{'GRU':<6}  {gru_rmse_pips:>12.3f}  {gru_mae_pips:>12.3f}  {gru_mape:>10.4f}")
print(f"{'LSTM':<6}  {lstm_rmse_pips:>12.3f}  {lstm_mae_pips:>12.3f}  {lstm_mape:>10.4f}")

# ---------------------------------------------------------------------
# 6. Save metrics CSV
# ---------------------------------------------------------------------
metrics_df = pd.DataFrame(
    [
        {"model": "GRU",  "rmse_pips": gru_rmse_pips,  "mae_pips": gru_mae_pips,  "mape_pct": gru_mape},
        {"model": "LSTM", "rmse_pips": lstm_rmse_pips, "mae_pips": lstm_mae_pips, "mape_pct": lstm_mape},
    ]
)
metrics_df.to_csv(METRICS_PATH, index=False)
print(f"\n[INFO] Metrics saved: {METRICS_PATH}")

# ---------------------------------------------------------------------
# 7. Save raw predictions
# ---------------------------------------------------------------------
pd.DataFrame({"actual": y_true_real, "gru_pred": gru_pred_real}).to_csv(GRU_PREDS_PATH, index=False)
pd.DataFrame({"actual": y_true_real, "lstm_pred": lstm_pred_real}).to_csv(LSTM_PREDS_PATH, index=False)

# ---------------------------------------------------------------------
# 8. Plot: predictions vs actual (last 500 test points for readability)
# ---------------------------------------------------------------------
N = 500
plt.figure(figsize=(14, 6))
plt.plot(y_true_real[-N:], label="Actual", linewidth=2, color="black")
plt.plot(gru_pred_real[-N:], label="GRU prediction", linewidth=1, alpha=0.85)
plt.plot(lstm_pred_real[-N:], label="LSTM prediction", linewidth=1, alpha=0.85)
plt.title(f"EUR/USD — Test set predictions (last {N} minutes)")
plt.xlabel("Time step (minutes)")
plt.ylabel("Price (EUR/USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PRED_PLOT_PATH, dpi=120)
plt.close()
print(f"[INFO] Saved plot: {PRED_PLOT_PATH}")

# ---------------------------------------------------------------------
# 9. Plot: loss curves from training histories
# ---------------------------------------------------------------------
gru_hist  = pd.read_csv(os.path.join("reports", "gru_history.csv"))
lstm_hist = pd.read_csv(os.path.join("reports", "lstm_history.csv"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(gru_hist["loss"], label="GRU train")
axes[0].plot(gru_hist["val_loss"], label="GRU val")
axes[0].plot(lstm_hist["loss"], label="LSTM train")
axes[0].plot(lstm_hist["val_loss"], label="LSTM val")
axes[0].set_title("Training loss (MSE, scaled)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(gru_hist["mae"], label="GRU train")
axes[1].plot(gru_hist["val_mae"], label="GRU val")
axes[1].plot(lstm_hist["mae"], label="LSTM train")
axes[1].plot(lstm_hist["val_mae"], label="LSTM val")
axes[1].set_title("Training MAE (scaled)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH, dpi=120)
plt.close()
print(f"[INFO] Saved plot: {LOSS_PLOT_PATH}")

print("\n[INFO] Evaluation complete.")
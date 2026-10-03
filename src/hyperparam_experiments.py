"""
hyperparam_experiments.py — Run a small hyperparameter sweep for
GRU and LSTM on the EUR/USD scaled Close series, logging every run
to the same MLflow experiment.

Each config differs in ONE dimension (lookback or units) so we can
attribute performance differences to a single cause.

Run from the project root:
    python src/hyperparam_experiments.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.keras

# ---------------------------------------------------------------------
# 0. MLflow setup — reuse the same experiment
# ---------------------------------------------------------------------
TRACKING_URI    = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "forex-gru-vs-lstm"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ---------------------------------------------------------------------
# 1. Fixed hyperparameters (identical to baseline)
# ---------------------------------------------------------------------
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 1e-3
DROPOUT       = 0.2
PATIENCE      = 5
FEATURE       = "Close_scaled"

# ---------------------------------------------------------------------
# 2. Experiment matrix — one variable changes at a time
# ---------------------------------------------------------------------
EXPERIMENTS = [
    # (lookback, units, run_label)
    (60,   50, "baseline_lb60_u50"),   # baseline (should match your existing runs)
    (30,   50, "short_lb30_u50"),
    (120,  50, "long_lb120_u50"),
    (60,  100, "wide_lb60_u100"),
]

# ---------------------------------------------------------------------
# 3. Load scaled series (once)
# ---------------------------------------------------------------------
print("[INFO] Loading scaled data")
train_scaled = pd.read_csv("data/train_scaled.csv", index_col="Datetime", parse_dates=True)
val_scaled   = pd.read_csv("data/val_scaled.csv",   index_col="Datetime", parse_dates=True)
test_scaled  = pd.read_csv("data/test_scaled.csv",  index_col="Datetime", parse_dates=True)

train_vals = train_scaled[FEATURE].values
val_vals   = val_scaled[FEATURE].values
test_vals  = test_scaled[FEATURE].values

# Load scaler for inverse-transform
with open(os.path.join("models", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

def inverse(vals):
    return scaler.inverse_transform(vals.reshape(-1, 1)).flatten()

# ---------------------------------------------------------------------
# 4. Sequence builder (parameterized by lookback)
# ---------------------------------------------------------------------
def make_sequences(series: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return (
        np.array(X, dtype=np.float32).reshape(-1, lookback, 1),
        np.array(y, dtype=np.float32),
    )

# ---------------------------------------------------------------------
# 5. Model builder
# ---------------------------------------------------------------------
def build_model(kind: str, lookback: int, units: int):
    assert kind in ("GRU", "LSTM")
    rnn_layer = layers.GRU if kind == "GRU" else layers.LSTM
    model = models.Sequential(
        [
            layers.Input(shape=(lookback, 1)),
            rnn_layer(units, return_sequences=False),
            layers.Dropout(DROPOUT),
            layers.Dense(1),
        ],
        name=f"{kind.lower()}_forex",
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model

# ---------------------------------------------------------------------
# 6. Run all experiments
# ---------------------------------------------------------------------
PIP = 1e-4
results = []

for lookback, units, label in EXPERIMENTS:
    print(f"\n{'='*70}")
    print(f"[CONFIG] lookback={lookback}  units={units}  ({label})")
    print(f"{'='*70}")

    # Build sequences for this lookback
    X_train, y_train = make_sequences(train_vals, lookback)
    X_val,   y_val   = make_sequences(val_vals,   lookback)
    X_test,  y_test  = make_sequences(test_vals,  lookback)

    print(f"  X_train: {X_train.shape}   X_val: {X_val.shape}   X_test: {X_test.shape}")

    # Train both architectures on the same config
    for kind in ("GRU", "LSTM"):
        run_name = f"{kind}_{label}"
        print(f"\n  → Training {run_name} ...")

        # Reproducibility per run
        tf.keras.utils.set_random_seed(42)

        model = build_model(kind, lookback, units)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=PATIENCE,
                    restore_best_weights=True,
                    verbose=0,
                )
            ],
            verbose=0,
        )

        hist_df = pd.DataFrame(history.history)

        # Predict + evaluate on TEST in real pips
        pred_scaled = model.predict(X_test, verbose=0).flatten()
        pred_real   = inverse(pred_scaled)
        y_true_real = inverse(y_test)

        rmse_pips = np.sqrt(mean_squared_error(y_true_real, pred_real)) / PIP
        mae_pips  = mean_absolute_error(y_true_real, pred_real) / PIP
        mape      = np.mean(np.abs((y_true_real - pred_real) / y_true_real)) * 100

        best_val_loss = float(hist_df["val_loss"].min())
        best_val_mae  = float(hist_df["val_mae"].min())
        epochs_run    = len(hist_df)

        # Log to MLflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model_type":  kind,
                "lookback":    lookback,
                "units":       units,
                "dropout":     DROPOUT,
                "batch_size":  BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "patience":    PATIENCE,
                "loss":        "mse",
                "optimizer":   "adam",
                "config_label": label,
            })
            mlflow.log_metrics({
                "best_val_loss":  best_val_loss,
                "best_val_mae":   best_val_mae,
                "epochs_trained": epochs_run,
                "test_rmse_pips": float(rmse_pips),
                "test_mae_pips":  float(mae_pips),
                "test_mape_pct":  float(mape),
            })
            mlflow.keras.log_model(model, artifact_path="model")

        print(f"     ✓ {run_name}: RMSE={rmse_pips:.3f} pips  "
              f"MAE={mae_pips:.3f} pips  epochs={epochs_run}  "
              f"best_val_loss={best_val_loss:.2e}")

        results.append({
            "run_name":      run_name,
            "model_type":    kind,
            "lookback":      lookback,
            "units":         units,
            "epochs_trained": epochs_run,
            "best_val_loss": best_val_loss,
            "test_rmse_pips": rmse_pips,
            "test_mae_pips":  mae_pips,
            "test_mape_pct":  mape,
        })

# ---------------------------------------------------------------------
# 7. Summary table
# ---------------------------------------------------------------------
print(f"\n{'='*70}")
print("[SUMMARY] All experiments")
print(f"{'='*70}")
summary = pd.DataFrame(results).sort_values("test_mae_pips")
print(summary.to_string(index=False))
summary.to_csv(os.path.join("reports", "hyperparam_summary.csv"), index=False)
print(f"\n[INFO] Summary saved: reports/hyperparam_summary.csv")
print("[INFO] All runs logged to MLflow experiment: forex-gru-vs-lstm")
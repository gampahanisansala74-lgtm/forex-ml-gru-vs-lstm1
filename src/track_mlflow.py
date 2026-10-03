"""
track_mlflow.py — Log both GRU and LSTM training runs + evaluation
metrics + artifacts into MLflow.

The MLflow tracking server uses a local SQLite DB in ./mlruns/,
so everything stays inside the project folder.

Run from the project root:
    python src/track_mlflow.py

Then launch the UI with:
    mlflow ui --backend-store-uri sqlite:///mlruns.db
"""

import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------
# 0. MLflow setup — local SQLite store inside the project
# ---------------------------------------------------------------------
TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "forex-gru-vs-lstm"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"[INFO] MLflow tracking URI: {TRACKING_URI}")
print(f"[INFO] Experiment: {EXPERIMENT_NAME}")

# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------
X_TEST_PATH = os.path.join("data", "X_test.npy")
y_TEST_PATH = os.path.join("data", "y_test.npy")
SCALER_PATH = os.path.join("models", "scaler.pkl")

MODELS = {
    "GRU":  {"path": os.path.join("models", "gru_model.keras"),
             "history": os.path.join("reports", "gru_history.csv")},
    "LSTM": {"path": os.path.join("models", "lstm_model.keras"),
             "history": os.path.join("reports", "lstm_history.csv")},
}

HYPERPARAMS = {
    "lookback": 60,
    "units": 50,
    "dropout": 0.2,
    "batch_size": 32,
    "epochs_max": 50,
    "learning_rate": 1e-3,
    "patience": 5,
    "loss": "mse",
    "optimizer": "adam",
}

# ---------------------------------------------------------------------
# 2. Load test data + scaler
# ---------------------------------------------------------------------
X_test = np.load(X_TEST_PATH)
y_test = np.load(y_TEST_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

def inverse(vals):
    return scaler.inverse_transform(vals.reshape(-1, 1)).flatten()

y_true_real = inverse(y_test)

# ---------------------------------------------------------------------
# 3. Log each model as a separate MLflow run
# ---------------------------------------------------------------------
for model_name, info in MODELS.items():
    print(f"\n[INFO] Logging {model_name} to MLflow...")

    # Load the trained model + history
    model = tf.keras.models.load_model(info["path"])
    history = pd.read_csv(info["history"])

    # Predict on test set
    pred_scaled = model.predict(X_test, verbose=0).flatten()
    pred_real   = inverse(pred_scaled)

    # Compute metrics in real pips
    PIP = 1e-4
    rmse_pips = np.sqrt(mean_squared_error(y_true_real, pred_real)) / PIP
    mae_pips  = mean_absolute_error(y_true_real, pred_real) / PIP
    mape      = np.mean(np.abs((y_true_real - pred_real) / y_true_real)) * 100

    # Best epoch metrics from history
    best_val_loss = float(history["val_loss"].min())
    best_val_mae  = float(history["val_mae"].min())
    epochs_trained = len(history)

    # Start a run
    with mlflow.start_run(run_name=model_name):
        # --- Params ---
        mlflow.log_params({**HYPERPARAMS, "model_type": model_name})

        # --- Metrics ---
        mlflow.log_metrics({
            "best_val_loss":  best_val_loss,
            "best_val_mae":   best_val_mae,
            "epochs_trained": epochs_trained,
            "test_rmse_pips": float(rmse_pips),
            "test_mae_pips":  float(mae_pips),
            "test_mape_pct":  float(mape),
        })

        # --- Artifacts ---
        mlflow.log_artifact(info["history"], artifact_path="history")
        mlflow.log_artifact(
            os.path.join("reports", "model_metrics.csv"),
            artifact_path="metrics",
        )
        mlflow.log_artifact(
            os.path.join("reports", "predictions_vs_actual.png"),
            artifact_path="plots",
        )
        mlflow.log_artifact(
            os.path.join("reports", "loss_curves.png"),
            artifact_path="plots",
        )

        # --- Model artifact (Keras) ---
        mlflow.keras.log_model(model, artifact_path="model")

        print(f"  ✓ {model_name} logged "
              f"(RMSE={rmse_pips:.3f} pips, MAE={mae_pips:.3f} pips)")

print("\n[INFO] All runs logged successfully.")
print("[INFO] Next step: launch the MLflow UI to view results.")
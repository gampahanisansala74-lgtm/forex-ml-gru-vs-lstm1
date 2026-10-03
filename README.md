# EUR/USD GRU vs LSTM — Deep Learning Forecasting with MLflow

A complete local machine learning pipeline for forecasting the next-minute
EUR/USD close price using two recurrent neural network architectures
(**GRU** and **LSTM**), with full experiment tracking via **MLflow**.

Built on Windows 11 with Python 3.11 + TensorFlow 2.21 (CPU).

---

## 📊 Results Summary

Test-set performance (real pips; lower is better).
Each model was trained on the same scaled data and evaluated on the
**same unseen chronological test split**.

| Rank | Run                    | Model | Lookback | Units | RMSE (pips) | MAE (pips) |
|------|------------------------|-------|----------|-------|-------------|------------|
| 🥇 1 | `GRU_short_lb30_u50`   | GRU   | 30       | 50    | 0.833       | **0.553**  |
| 🥈 2 | `GRU_baseline_lb60_u50`| GRU   | 60       | 50    | 0.877       | 0.597      |
| 🥉 3 | `LSTM_baseline_lb60_u50`| LSTM | 60       | 50    | 0.980       | 0.676      |
| 4    | `LSTM_short_lb30_u50`  | LSTM  | 30       | 50    | 1.238       | 0.830      |

### Key findings

- **GRU beat LSTM** on every comparable configuration. On this dataset
  (single-feature, minute-level, limited volatility), the simpler model
  generalizes better and trains faster.
- **Shorter lookback (30) beat longer (60)** for GRU. Intraday FX has very
  short memory — a 60-minute window adds noise more than signal.
- LSTM performed *worse* with the shorter window, suggesting its extra
  capacity needs more input context to be useful.

> **Note:** The dataset spans ~7 weeks of minute bars. Absolute error
> magnitudes are small because the test period was relatively calm. The
> methodology — chronological splits, scaler fit on train only, sliding
> windows, MLflow tracking — is the primary deliverable.

---

## 🗂️ Project Structure
forex-ml/
├── data/ # Raw + processed datasets (not committed)
├── models/ # Trained models + scaler
├── notebooks/ # (reserved for experimentation)
├── reports/ # Metrics, plots, comparison summaries
├── src/ # All pipeline scripts
│ ├── preprocess.py
│ ├── split_data.py
│ ├── scale_data.py
│ ├── make_sequences.py
│ ├── train_gru.py
│ ├── train_lstm.py
│ ├── evaluate.py
│ ├── track_mlflow.py
│ └── hyperparam_experiments.py
├── mlruns.db # MLflow tracking database
├── requirements.txt
├── .gitignore
└── README.md

---

## 🚀 How to Reproduce

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt# forex-ml-gru-vs-lstm1

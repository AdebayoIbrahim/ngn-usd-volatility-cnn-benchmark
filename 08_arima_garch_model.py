# ============================================================
# 08_arima_garch_model.py
# Project: Multivariate 2D-CNN for ₦/USD Volatility Forecasting
# Task: Walk-forward ARIMA-GARCH benchmark model
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# -----------------------------
# 1. File paths
# -----------------------------
BASE_DIR = Path(".")

cleaned_dir = BASE_DIR / "data_cleaned"
figures_dir = BASE_DIR / "figures"
results_dir = BASE_DIR / "results"

figures_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

dataset_file = cleaned_dir / "feature_engineered_fx_dataset_2006_2024.csv"


# -----------------------------
# 2. Load dataset
# -----------------------------
df = pd.read_csv(dataset_file)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("\nDataset preview:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isna().sum())


# -----------------------------
# 3. Select series
# -----------------------------
series = df["fx_return"].copy()
actual_spike = df["spike"].copy()
actual_volatility = df["volatility"].copy()
dates = df["Date"].copy()


# -----------------------------
# 4. Train-test split
# -----------------------------
split_index = int(len(df) * 0.80)

train_series = series.iloc[:split_index].copy()
test_series = series.iloc[split_index:].copy()

train_volatility = actual_volatility.iloc[:split_index].copy()

train_dates = dates.iloc[:split_index].copy()
test_dates = dates.iloc[split_index:].copy()

y_test_spike = actual_spike.iloc[split_index:].copy()

print("\nTrain size:", len(train_series))
print("Test size:", len(test_series))

print("\nTrain period:", train_dates.min(), "to", train_dates.max())
print("Test period:", test_dates.min(), "to", test_dates.max())


# ============================================================
# 5. Fit initial ARIMA-GARCH on training data
#    This is used to estimate a volatility threshold
# ============================================================

print("\nFitting initial ARIMA(1,0,1) on training data...")

initial_arima = ARIMA(train_series, order=(1, 0, 1))
initial_arima_result = initial_arima.fit()

initial_residuals = initial_arima_result.resid

print("\nFitting initial GARCH(1,1) on ARIMA residuals...")

initial_garch = arch_model(
    initial_residuals,
    vol="Garch",
    p=1,
    q=1,
    mean="Zero",
    dist="normal"
)

initial_garch_result = initial_garch.fit(disp="off")

# In-sample conditional volatility from training period
train_conditional_volatility = initial_garch_result.conditional_volatility

# Use 75th percentile of training GARCH volatility as threshold
vol_threshold = np.quantile(train_conditional_volatility, 0.75)

print("\nTraining GARCH volatility threshold:")
print(vol_threshold)


# ============================================================
# 6. Walk-forward one-step volatility forecasting
# ============================================================

predicted_volatility = []

print("\nStarting walk-forward ARIMA-GARCH forecasting...")

for i in range(len(test_series)):
    print(f"Forecasting test step {i + 1}/{len(test_series)}")

    # Expanding window: train data plus test observations before current point
    current_train = series.iloc[:split_index + i].copy()

    try:
        # Fit ARIMA on current training window
        arima_model = ARIMA(current_train, order=(1, 0, 1))
        arima_result = arima_model.fit()

        residuals = arima_result.resid

        # Fit GARCH on ARIMA residuals
        garch_model = arch_model(
            residuals,
            vol="Garch",
            p=1,
            q=1,
            mean="Zero",
            dist="normal"
        )

        garch_result = garch_model.fit(disp="off")

        # One-step-ahead variance forecast
        forecast = garch_result.forecast(horizon=1, reindex=False)
        variance_forecast = forecast.variance.values[-1, 0]

        vol_forecast = np.sqrt(variance_forecast)

    except Exception as e:
        print(f"Warning: model failed at step {i + 1}. Error: {e}")

        # fallback: use previous volatility forecast, or training threshold
        if len(predicted_volatility) > 0:
            vol_forecast = predicted_volatility[-1]
        else:
            vol_forecast = vol_threshold

    predicted_volatility.append(vol_forecast)


# Build results dataframe
arima_garch_results = pd.DataFrame({
    "Date": test_dates.values,
    "actual_fx_return": test_series.values,
    "actual_volatility": actual_volatility.iloc[split_index:].values,
    "actual_spike": y_test_spike.values,
    "predicted_volatility": predicted_volatility
})

print("\nARIMA-GARCH walk-forward results preview:")
print(arima_garch_results.head())

print("\nARIMA-GARCH walk-forward results tail:")
print(arima_garch_results.tail())


# ============================================================
# 7. Convert predicted volatility to spike predictions
# ============================================================

arima_garch_results["predicted_spike"] = np.where(
    arima_garch_results["predicted_volatility"] > vol_threshold,
    1,
    0
)

print("\nActual spike counts in test set:")
print(arima_garch_results["actual_spike"].value_counts())

print("\nPredicted spike counts in test set:")
print(arima_garch_results["predicted_spike"].value_counts())


# ============================================================
# 8. Evaluation metrics
# ============================================================

y_true = arima_garch_results["actual_spike"]
y_pred = arima_garch_results["predicted_spike"]
y_score = arima_garch_results["predicted_volatility"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

try:
    auc = roc_auc_score(y_true, y_score)
except ValueError:
    auc = np.nan

cm = confusion_matrix(y_true, y_pred)

print("\nARIMA-GARCH Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)

print("\nARIMA-GARCH Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC:", auc)


# ============================================================
# 9. Save metrics and predictions
# ============================================================

metrics_df = pd.DataFrame({
    "model": ["ARIMA-GARCH"],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1],
    "auc": [auc],
    "volatility_threshold": [vol_threshold]
})

metrics_path = results_dir / "arima_garch_metrics.csv"
results_path = results_dir / "arima_garch_predictions.csv"

metrics_df.to_csv(metrics_path, index=False)
arima_garch_results.to_csv(results_path, index=False)

print("\nSaved ARIMA-GARCH metrics:")
print(metrics_path)

print("\nSaved ARIMA-GARCH predictions:")
print(results_path)


# ============================================================
# 10. Plot actual vs predicted spike
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(
    arima_garch_results["Date"],
    arima_garch_results["actual_spike"],
    label="Actual Spike",
    marker="o"
)
plt.plot(
    arima_garch_results["Date"],
    arima_garch_results["predicted_spike"],
    label="Predicted Spike",
    marker="x"
)

plt.title("ARIMA-GARCH Actual vs Predicted Volatility Spikes")
plt.xlabel("Date")
plt.ylabel("Spike Class")
plt.legend()
plt.grid(True)
plt.tight_layout()

spike_plot_path = figures_dir / "arima_garch_actual_vs_predicted_spike.png"
plt.savefig(spike_plot_path, dpi=300)
plt.show()


# ============================================================
# 11. Plot predicted vs actual volatility
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(
    arima_garch_results["Date"],
    arima_garch_results["actual_volatility"],
    label="Actual Rolling Volatility"
)
plt.plot(
    arima_garch_results["Date"],
    arima_garch_results["predicted_volatility"],
    label="Predicted GARCH Volatility"
)

plt.axhline(
    vol_threshold,
    linestyle="--",
    label="GARCH Spike Threshold"
)

plt.title("ARIMA-GARCH Actual vs Predicted Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()

vol_plot_path = figures_dir / "arima_garch_actual_vs_predicted_volatility.png"
plt.savefig(vol_plot_path, dpi=300)
plt.show()


print("\nSaved figures:")
print(spike_plot_path)
print(vol_plot_path)

print("\nWalk-forward ARIMA-GARCH modelling completed successfully.")
# ============================================================
# 09_bayesian_var_model.py
# Project: Multivariate 2D-CNN for ₦/USD Volatility Forecasting
# Task: Bayesian VAR-style benchmark model for FX volatility spike detection
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
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


# ============================================================
# 3. Select variables for Bayesian VAR
# ============================================================

var_cols = [
    "fx_return",
    "oil_return",
    "reserves_change",
    "mpr_change"
]

target_col = "fx_return"

dates = df["Date"].copy()
actual_spike = df["spike"].copy()
actual_volatility = df["volatility"].copy()

data = df[var_cols].copy()


# ============================================================
# 4. Helper function to create VAR lag features
# ============================================================

def create_var_lag_features(dataframe, variables, lags=3):
    """
    Creates lagged features for VAR-style modelling.

    Example:
    fx_return_lag1, oil_return_lag1, reserves_change_lag1, mpr_change_lag1
    fx_return_lag2, oil_return_lag2, ...
    """
    lagged_df = dataframe.copy()

    for lag in range(1, lags + 1):
        for col in variables:
            lagged_df[f"{col}_lag{lag}"] = lagged_df[col].shift(lag)

    return lagged_df


# Number of monthly lags
# 3 means the model uses the previous 3 months to forecast the next month.
lags = 3

lagged_data = create_var_lag_features(df, var_cols, lags=lags)

feature_cols = []

for lag in range(1, lags + 1):
    for col in var_cols:
        feature_cols.append(f"{col}_lag{lag}")

# Drop rows with lag NaNs
model_data = lagged_data.dropna().copy().reset_index(drop=True)

print("\nBVAR-style model data preview:")
print(model_data[["Date"] + feature_cols + [target_col, "spike"]].head())

print("\nBVAR-style model data shape:")
print(model_data.shape)


# ============================================================
# 5. Train-test split
# ============================================================

split_index = int(len(model_data) * 0.80)

train_df = model_data.iloc[:split_index].copy()
test_df = model_data.iloc[split_index:].copy()

print("\nTrain size:", len(train_df))
print("Test size:", len(test_df))

print("\nTrain period:", train_df["Date"].min(), "to", train_df["Date"].max())
print("Test period:", test_df["Date"].min(), "to", test_df["Date"].max())


# ============================================================
# 6. Estimate training uncertainty threshold
# ============================================================

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

initial_model = BayesianRidge()
initial_model.fit(X_train_scaled, y_train)

# In-sample predictive uncertainty
train_pred_mean, train_pred_std = initial_model.predict(
    X_train_scaled,
    return_std=True
)

# Use 75th percentile of training predictive uncertainty as spike threshold
uncertainty_threshold = np.quantile(train_pred_std, 0.75)

print("\nTraining predictive uncertainty threshold:")
print(uncertainty_threshold)


# ============================================================
# 7. Walk-forward Bayesian VAR-style forecasting
# ============================================================

predicted_fx_return = []
predicted_uncertainty = []

print("\nStarting walk-forward Bayesian VAR-style forecasting...")

for i in range(len(test_df)):
    print(f"Forecasting test step {i + 1}/{len(test_df)}")

    # Expanding window
    current_train = model_data.iloc[:split_index + i].copy()
    current_test = test_df.iloc[[i]].copy()

    X_current_train = current_train[feature_cols].values
    y_current_train = current_train[target_col].values

    X_current_test = current_test[feature_cols].values

    # Scale using only current training data
    scaler = StandardScaler()
    X_current_train_scaled = scaler.fit_transform(X_current_train)
    X_current_test_scaled = scaler.transform(X_current_test)

    # Bayesian Ridge as BVAR-style equation for FX return
    bvar_model = BayesianRidge()
    bvar_model.fit(X_current_train_scaled, y_current_train)

    pred_mean, pred_std = bvar_model.predict(
        X_current_test_scaled,
        return_std=True
    )

    predicted_fx_return.append(pred_mean[0])
    predicted_uncertainty.append(pred_std[0])


# Build results dataframe
bvar_results = pd.DataFrame({
    "Date": test_df["Date"].values,
    "actual_fx_return": test_df["fx_return"].values,
    "predicted_fx_return": predicted_fx_return,
    "actual_volatility": test_df["volatility"].values,
    "actual_spike": test_df["spike"].values,
    "predicted_uncertainty": predicted_uncertainty
})

print("\nBayesian VAR-style results preview:")
print(bvar_results.head())

print("\nBayesian VAR-style results tail:")
print(bvar_results.tail())


# ============================================================
# 8. Convert predictive uncertainty to spike predictions
# ============================================================

bvar_results["predicted_spike"] = np.where(
    bvar_results["predicted_uncertainty"] > uncertainty_threshold,
    1,
    0
)

print("\nActual spike counts in test set:")
print(bvar_results["actual_spike"].value_counts())

print("\nPredicted spike counts in test set:")
print(bvar_results["predicted_spike"].value_counts())


# ============================================================
# 9. Evaluation metrics
# ============================================================

y_true = bvar_results["actual_spike"]
y_pred = bvar_results["predicted_spike"]
y_score = bvar_results["predicted_uncertainty"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

try:
    auc = roc_auc_score(y_true, y_score)
except ValueError:
    auc = np.nan

cm = confusion_matrix(y_true, y_pred)

print("\nBayesian VAR-style Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)

print("\nBayesian VAR-style Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC:", auc)


# ============================================================
# 10. Save metrics and predictions
# ============================================================

metrics_df = pd.DataFrame({
    "model": ["Bayesian VAR"],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1],
    "auc": [auc],
    "uncertainty_threshold": [uncertainty_threshold],
    "lags": [lags]
})

metrics_path = results_dir / "bayesian_var_metrics.csv"
predictions_path = results_dir / "bayesian_var_predictions.csv"

metrics_df.to_csv(metrics_path, index=False)
bvar_results.to_csv(predictions_path, index=False)

print("\nSaved Bayesian VAR metrics:")
print(metrics_path)

print("\nSaved Bayesian VAR predictions:")
print(predictions_path)


# ============================================================
# 11. Plot actual vs predicted spike
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(
    bvar_results["Date"],
    bvar_results["actual_spike"],
    label="Actual Spike",
    marker="o"
)
plt.plot(
    bvar_results["Date"],
    bvar_results["predicted_spike"],
    label="Predicted Spike",
    marker="x"
)

plt.title("Bayesian VAR Actual vs Predicted Volatility Spikes")
plt.xlabel("Date")
plt.ylabel("Spike Class")
plt.legend()
plt.grid(True)
plt.tight_layout()

spike_plot_path = figures_dir / "bayesian_var_actual_vs_predicted_spike.png"
plt.savefig(spike_plot_path, dpi=300)
plt.show()


# ============================================================
# 12. Plot predictive uncertainty
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(
    bvar_results["Date"],
    bvar_results["predicted_uncertainty"],
    label="Predictive Uncertainty"
)
plt.axhline(
    uncertainty_threshold,
    linestyle="--",
    label="Uncertainty Spike Threshold"
)

plt.title("Bayesian VAR Predictive Uncertainty")
plt.xlabel("Date")
plt.ylabel("Predictive Uncertainty")
plt.legend()
plt.grid(True)
plt.tight_layout()

uncertainty_plot_path = figures_dir / "bayesian_var_predictive_uncertainty.png"
plt.savefig(uncertainty_plot_path, dpi=300)
plt.show()


# ============================================================
# 13. Plot actual vs predicted FX return
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(
    bvar_results["Date"],
    bvar_results["actual_fx_return"],
    label="Actual FX Return"
)
plt.plot(
    bvar_results["Date"],
    bvar_results["predicted_fx_return"],
    label="Predicted FX Return"
)

plt.title("Bayesian VAR Actual vs Predicted FX Return")
plt.xlabel("Date")
plt.ylabel("FX Return (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

return_plot_path = figures_dir / "bayesian_var_actual_vs_predicted_fx_return.png"
plt.savefig(return_plot_path, dpi=300)
plt.show()


print("\nSaved figures:")
print(spike_plot_path)
print(uncertainty_plot_path)
print(return_plot_path)

print("\nBayesian VAR-style modelling completed successfully.")
# ============================================================
# 10_train_2d_cnn_v3_volatility_regression.py
# Project: Multivariate 2D-CNN for ₦/USD Volatility Forecasting
# Task: 2D-CNN regression model for volatility forecasting,
#       then spike detection using predicted volatility
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# 1. Reproducibility
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. File paths
# ============================================================

BASE_DIR = Path(".")

cleaned_dir = BASE_DIR / "data_cleaned"
figures_dir = BASE_DIR / "figures"
results_dir = BASE_DIR / "results"
models_dir = BASE_DIR / "models"

figures_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

dataset_file = cleaned_dir / "feature_engineered_fx_dataset_2006_2024.csv"


# ============================================================
# 3. Load dataset
# ============================================================

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
# 4. Select CNN input variables
# ============================================================

# We use macroeconomic levels + returns/changes.
# Target is volatility, not spike.
feature_cols = [
    "usd_ngn_rate",
    "oil_price",
    "external_reserves",
    "mpr",
    "fx_return",
    "oil_return",
    "reserves_change",
    "mpr_change"
]

target_col = "volatility"
spike_col = "spike"


# ============================================================
# 5. Train-test split by date
# ============================================================

test_start_date = pd.to_datetime("2021-04-01")

train_df_raw = df[df["Date"] < test_start_date].copy()
test_df_raw = df[df["Date"] >= test_start_date].copy()

print("\nRaw train period:", train_df_raw["Date"].min(), "to", train_df_raw["Date"].max())
print("Raw test period:", test_df_raw["Date"].min(), "to", test_df_raw["Date"].max())

print("\nRaw train size:", len(train_df_raw))
print("Raw test size:", len(test_df_raw))


# ============================================================
# 6. Scale features and target using training data only
# ============================================================

feature_scaler = StandardScaler()
target_scaler = StandardScaler()

df_scaled = df.copy()

feature_scaler.fit(train_df_raw[feature_cols])
df_scaled[feature_cols] = feature_scaler.transform(df[feature_cols])

target_scaler.fit(train_df_raw[[target_col]])
df_scaled[target_col] = target_scaler.transform(df[[target_col]])


# ============================================================
# 7. Create 2D windows
# ============================================================

def create_cnn_windows(dataframe, feature_columns, target_column, spike_column, window_size=12):
    """
    Converts multivariate time-series into 2D windows.

    X shape = samples × variables × months × channels
    y_reg = scaled volatility target
    y_spike = original spike label
    """
    X = []
    y_reg = []
    y_spike = []
    sample_dates = []

    values = dataframe[feature_columns].values
    targets = dataframe[target_column].values
    spikes = dataframe[spike_column].values
    date_values = dataframe["Date"].values

    for i in range(window_size, len(dataframe)):
        window = values[i - window_size:i].T

        X.append(window)
        y_reg.append(targets[i])
        y_spike.append(spikes[i])
        sample_dates.append(date_values[i])

    X = np.array(X)
    y_reg = np.array(y_reg)
    y_spike = np.array(y_spike)
    sample_dates = pd.to_datetime(np.array(sample_dates))

    X = X[..., np.newaxis]

    return X, y_reg, y_spike, sample_dates


window_size = 12

X, y_reg, y_spike, sample_dates = create_cnn_windows(
    df_scaled,
    feature_cols,
    target_col,
    spike_col,
    window_size=window_size
)

print("\nCNN input shape:")
print(X.shape)

print("\nRegression target shape:")
print(y_reg.shape)

print("\nSpike target shape:")
print(y_spike.shape)

print("\nFirst CNN sample date:", sample_dates.min())
print("Last CNN sample date:", sample_dates.max())


# ============================================================
# 8. Split CNN data into train, validation, and test
# ============================================================

test_mask = sample_dates >= test_start_date

X_train_full = X[~test_mask]
y_train_full = y_reg[~test_mask]
spike_train_full = y_spike[~test_mask]
dates_train_full = sample_dates[~test_mask]

X_test = X[test_mask]
y_test_scaled = y_reg[test_mask]
spike_test = y_spike[test_mask]
dates_test = sample_dates[test_mask]

# Validation split from the end of training period
val_size = int(len(X_train_full) * 0.20)

X_train = X_train_full[:-val_size]
y_train = y_train_full[:-val_size]
dates_train = dates_train_full[:-val_size]

X_val = X_train_full[-val_size:]
y_val = y_train_full[-val_size:]
spike_val = spike_train_full[-val_size:]
dates_val = dates_train_full[-val_size:]

print("\nCNN train size:", len(X_train))
print("CNN validation size:", len(X_val))
print("CNN test size:", len(X_test))

print("\nCNN train period:", dates_train.min(), "to", dates_train.max())
print("CNN validation period:", dates_val.min(), "to", dates_val.max())
print("CNN test period:", dates_test.min(), "to", dates_test.max())

print("\nValidation spike class counts:")
print(pd.Series(spike_val).value_counts())

print("\nTest spike class counts:")
print(pd.Series(spike_test).value_counts())


# ============================================================
# 9. Build 2D-CNN regression model
# ============================================================

input_shape = X_train.shape[1:]

model = Sequential([
    Conv2D(
        filters=8,
        kernel_size=(2, 3),
        activation="relu",
        padding="same",
        input_shape=input_shape
    ),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 2)),

    Conv2D(
        filters=16,
        kernel_size=(2, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 2)),

    Flatten(),

    Dense(16, activation="relu"),
    Dropout(0.20),

    Dense(1, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"]
)

print("\n2D-CNN V3 Regression Model Summary:")
model.summary()


# ============================================================
# 10. Train model
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)


# ============================================================
# 11. Predict validation and tune volatility threshold
# ============================================================

val_pred_scaled = model.predict(X_val).flatten()

# Convert scaled predicted volatility back to original volatility scale
val_pred_volatility = target_scaler.inverse_transform(
    val_pred_scaled.reshape(-1, 1)
).flatten()

val_actual_volatility = target_scaler.inverse_transform(
    y_val.reshape(-1, 1)
).flatten()

# Tune threshold based on validation F1-score
thresholds = np.linspace(
    val_pred_volatility.min(),
    val_pred_volatility.max(),
    100
)

threshold_results = []

for threshold in thresholds:
    val_pred_spike = np.where(val_pred_volatility >= threshold, 1, 0)

    accuracy = accuracy_score(spike_val, val_pred_spike)
    precision = precision_score(spike_val, val_pred_spike, zero_division=0)
    recall = recall_score(spike_val, val_pred_spike, zero_division=0)
    f1 = f1_score(spike_val, val_pred_spike, zero_division=0)

    threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

threshold_df = pd.DataFrame(threshold_results)

best_row = threshold_df.sort_values(
    by=["f1_score", "recall", "accuracy"],
    ascending=False
).iloc[0]

best_threshold = float(best_row["threshold"])

print("\nBest validation volatility threshold:")
print(best_row)

threshold_path = results_dir / "cnn_2d_v3_threshold_tuning.csv"
threshold_df.to_csv(threshold_path, index=False)


# ============================================================
# 12. Predict test volatility
# ============================================================

test_pred_scaled = model.predict(X_test).flatten()

predicted_volatility = target_scaler.inverse_transform(
    test_pred_scaled.reshape(-1, 1)
).flatten()

actual_volatility = target_scaler.inverse_transform(
    y_test_scaled.reshape(-1, 1)
).flatten()

predicted_spike = np.where(predicted_volatility >= best_threshold, 1, 0)


# ============================================================
# 13. Build results dataframe
# ============================================================

cnn_results = pd.DataFrame({
    "Date": dates_test,
    "actual_volatility": actual_volatility,
    "predicted_volatility": predicted_volatility,
    "actual_spike": spike_test,
    "predicted_spike": predicted_spike
})

print("\nCNN V3 prediction results preview:")
print(cnn_results.head())

print("\nCNN V3 prediction results tail:")
print(cnn_results.tail())

print("\nActual spike counts in test set:")
print(cnn_results["actual_spike"].value_counts())

print("\nPredicted spike counts in test set:")
print(cnn_results["predicted_spike"].value_counts())


# ============================================================
# 14. Regression metrics
# ============================================================

mae = mean_absolute_error(actual_volatility, predicted_volatility)
mse = mean_squared_error(actual_volatility, predicted_volatility)
rmse = np.sqrt(mse)

print("\n2D-CNN V3 Regression Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)


# ============================================================
# 15. Classification metrics
# ============================================================

y_true = cnn_results["actual_spike"]
y_pred = cnn_results["predicted_spike"]
y_score = cnn_results["predicted_volatility"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

try:
    auc = roc_auc_score(y_true, y_score)
except ValueError:
    auc = np.nan

cm = confusion_matrix(y_true, y_pred)

print("\n2D-CNN V3 Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)

print("\n2D-CNN V3 Classification Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC:", auc)
print("Best volatility threshold:", best_threshold)


# ============================================================
# 16. Save metrics and predictions
# ============================================================

metrics_df = pd.DataFrame({
    "model": ["Multivariate 2D-CNN V3 Volatility Regression"],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1],
    "auc": [auc],
    "mae": [mae],
    "mse": [mse],
    "rmse": [rmse],
    "volatility_threshold": [best_threshold],
    "window_size": [window_size],
    "epochs_trained": [len(history.history["loss"])],
    "num_features": [len(feature_cols)]
})

metrics_path = results_dir / "cnn_2d_v3_metrics.csv"
predictions_path = results_dir / "cnn_2d_v3_predictions.csv"

metrics_df.to_csv(metrics_path, index=False)
cnn_results.to_csv(predictions_path, index=False)

model_path = models_dir / "multivariate_2d_cnn_v3_volatility_model.keras"
model.save(model_path)

print("\nSaved 2D-CNN V3 metrics:")
print(metrics_path)

print("\nSaved 2D-CNN V3 predictions:")
print(predictions_path)

print("\nSaved 2D-CNN V3 model:")
print(model_path)

print("\nSaved threshold tuning results:")
print(threshold_path)


# ============================================================
# 17. Plot training history
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("2D-CNN V3 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

loss_plot_path = figures_dir / "cnn_2d_v3_training_loss.png"
plt.savefig(loss_plot_path, dpi=300)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.title("2D-CNN V3 Training and Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()

mae_plot_path = figures_dir / "cnn_2d_v3_training_mae.png"
plt.savefig(mae_plot_path, dpi=300)
plt.show()


# ============================================================
# 18. Plot threshold tuning
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(threshold_df["threshold"], threshold_df["f1_score"], label="Validation F1-score")
plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Validation Precision")
plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Validation Recall")
plt.axvline(best_threshold, linestyle="--", label="Best Threshold")
plt.title("2D-CNN V3 Volatility Threshold Tuning")
plt.xlabel("Predicted Volatility Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()

threshold_plot_path = figures_dir / "cnn_2d_v3_threshold_tuning.png"
plt.savefig(threshold_plot_path, dpi=300)
plt.show()


# ============================================================
# 19. Plot actual vs predicted volatility
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(
    cnn_results["Date"],
    cnn_results["actual_volatility"],
    label="Actual Volatility"
)
plt.plot(
    cnn_results["Date"],
    cnn_results["predicted_volatility"],
    label="Predicted Volatility"
)
plt.axhline(
    best_threshold,
    linestyle="--",
    label="Spike Threshold"
)

plt.title("2D-CNN V3 Actual vs Predicted Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()

volatility_plot_path = figures_dir / "cnn_2d_v3_actual_vs_predicted_volatility.png"
plt.savefig(volatility_plot_path, dpi=300)
plt.show()


# ============================================================
# 20. Plot actual vs predicted spike
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(
    cnn_results["Date"],
    cnn_results["actual_spike"],
    label="Actual Spike",
    marker="o"
)
plt.plot(
    cnn_results["Date"],
    cnn_results["predicted_spike"],
    label="Predicted Spike",
    marker="x"
)

plt.title("2D-CNN V3 Actual vs Predicted Volatility Spikes")
plt.xlabel("Date")
plt.ylabel("Spike Class")
plt.legend()
plt.grid(True)
plt.tight_layout()

spike_plot_path = figures_dir / "cnn_2d_v3_actual_vs_predicted_spike.png"
plt.savefig(spike_plot_path, dpi=300)
plt.show()


print("\nSaved figures:")
print(loss_plot_path)
print(mae_plot_path)
print(threshold_plot_path)
print(volatility_plot_path)
print(spike_plot_path)

print("\n2D-CNN V3 volatility regression modelling completed successfully.")
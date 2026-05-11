# ============================================================
# 06_feature_engineering.py
# Project: Multivariate 2D-CNN for ₦/USD Volatility Forecasting
# Task: Create returns, volatility, and spike target variable
# ============================================================

# -----------------------------
# 1. Necessary imports
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# 2. File paths
# -----------------------------
BASE_DIR = Path(".")

cleaned_dir = BASE_DIR / "data_cleaned"
figures_dir = BASE_DIR / "figures"
results_dir = BASE_DIR / "results"

figures_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

merged_file = cleaned_dir / "merged_macro_fx_dataset_2006_2024.csv"


# -----------------------------
# 3. Load merged dataset
# -----------------------------
df = pd.read_csv(merged_file)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print("\nMerged dataset preview:")
print(df.head())

print("\nMerged dataset tail:")
print(df.tail())

print("\nInitial shape:")
print(df.shape)

print("\nMissing values before feature engineering:")
print(df.isna().sum())


# ============================================================
# 4. Safety checks
# ============================================================

# Check for invalid exchange rate values
invalid_fx = df[df["usd_ngn_rate"] <= 0]

print("\nInvalid exchange-rate rows:")
print(invalid_fx)

if len(invalid_fx) > 0:
    print("\nFixing invalid exchange-rate values...")
    df["usd_ngn_rate"] = df["usd_ngn_rate"].replace(0, np.nan)
    df["usd_ngn_rate"] = df["usd_ngn_rate"].interpolate(method="linear").ffill().bfill()

# Check again
print("\nInvalid exchange-rate rows after safety fix:")
print(df[df["usd_ngn_rate"] <= 0])


# ============================================================
# 5. Create return/change variables
# ============================================================

# FX return: percentage change in ₦/USD exchange rate
df["fx_return"] = df["usd_ngn_rate"].pct_change() * 100

# Oil return: percentage change in Brent crude oil price
df["oil_return"] = df["oil_price"].pct_change() * 100

# External reserves change: percentage change in external reserves
df["reserves_change"] = df["external_reserves"].pct_change() * 100

# MPR change: month-to-month change in policy rate
df["mpr_change"] = df["mpr"].diff()


# ============================================================
# 6. Create volatility variable
# ============================================================

# Rolling volatility of exchange-rate returns.
# Since data is monthly, 3-month rolling volatility captures short-run instability.
rolling_window = 3

df["volatility"] = (
    df["fx_return"]
    .rolling(window=rolling_window)
    .std()
)

# Alternative possible window:
# 6 months can be used for smoother volatility.
# For now, we use 3 months because we are detecting spikes.


# ============================================================
# 7. Create volatility spike target
# ============================================================

# Use the 75th percentile of volatility as the spike threshold
spike_threshold = df["volatility"].quantile(0.75)

df["spike"] = np.where(df["volatility"] > spike_threshold, 1, 0)

print("\nVolatility spike threshold:")
print(spike_threshold)

print("\nSpike value counts:")
print(df["spike"].value_counts())


# ============================================================
# 8. Drop rows created as missing due to pct_change and rolling
# ============================================================

# The first few rows will have NaN because returns and rolling volatility need past values.
model_df = df.dropna().copy()

model_df = model_df.reset_index(drop=True)

print("\nFeature-engineered dataset preview:")
print(model_df.head())

print("\nFeature-engineered dataset tail:")
print(model_df.tail())

print("\nFinal shape after dropping NaN rows:")
print(model_df.shape)

print("\nMissing values after feature engineering:")
print(model_df.isna().sum())


# ============================================================
# 9. Save descriptive statistics
# ============================================================

desc_stats = model_df.describe()

print("\nDescriptive statistics after feature engineering:")
print(desc_stats)

desc_stats_path = results_dir / "feature_engineered_descriptive_statistics.csv"
desc_stats.to_csv(desc_stats_path)


# ============================================================
# 10. Plot FX return
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(model_df["Date"], model_df["fx_return"])
plt.title("Monthly ₦/USD Exchange Rate Return, 2006–2024")
plt.xlabel("Date")
plt.ylabel("FX Return (%)")
plt.grid(True)
plt.tight_layout()

fx_return_plot = figures_dir / "fx_return_2006_2024.png"
plt.savefig(fx_return_plot, dpi=300)
plt.show()


# ============================================================
# 11. Plot volatility
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(model_df["Date"], model_df["volatility"])
plt.axhline(spike_threshold, linestyle="--", label="Spike Threshold")
plt.title("Rolling Exchange Rate Volatility, 2006–2024")
plt.xlabel("Date")
plt.ylabel("Rolling Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()

volatility_plot = figures_dir / "fx_volatility_2006_2024.png"
plt.savefig(volatility_plot, dpi=300)
plt.show()


# ============================================================
# 12. Plot volatility spikes
# ============================================================

spike_dates = model_df[model_df["spike"] == 1]

plt.figure(figsize=(12, 6))
plt.plot(model_df["Date"], model_df["volatility"], label="Volatility")
plt.scatter(
    spike_dates["Date"],
    spike_dates["volatility"],
    label="Volatility Spike"
)
plt.axhline(spike_threshold, linestyle="--", label="Spike Threshold")
plt.title("Detected ₦/USD Volatility Spikes, 2006–2024")
plt.xlabel("Date")
plt.ylabel("Rolling Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()

spike_plot = figures_dir / "detected_fx_volatility_spikes_2006_2024.png"
plt.savefig(spike_plot, dpi=300)
plt.show()


# ============================================================
# 13. Plot all engineered variables
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(model_df["Date"], model_df["fx_return"], label="FX Return")
plt.plot(model_df["Date"], model_df["oil_return"], label="Oil Return")
plt.plot(model_df["Date"], model_df["reserves_change"], label="Reserves Change")
plt.title("Monthly Return/Change Variables, 2006–2024")
plt.xlabel("Date")
plt.ylabel("Percentage Change (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

returns_plot = figures_dir / "macro_return_variables_2006_2024.png"
plt.savefig(returns_plot, dpi=300)
plt.show()


# ============================================================
# 14. Save final feature-engineered dataset
# ============================================================

csv_output_path = cleaned_dir / "feature_engineered_fx_dataset_2006_2024.csv"
excel_output_path = cleaned_dir / "feature_engineered_fx_dataset_2006_2024.xlsx"

model_df.to_csv(csv_output_path, index=False)
model_df.to_excel(excel_output_path, index=False)

print("\nSaved feature-engineered dataset:")
print(csv_output_path)
print(excel_output_path)

print("\nSaved descriptive statistics:")
print(desc_stats_path)

print("\nSaved plots:")
print(fx_return_plot)
print(volatility_plot)
print(spike_plot)
print(returns_plot)

print("\nFeature engineering completed successfully.")

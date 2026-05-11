# ============================================================
# 07_eda_analysis.py
# Project: Multivariate 2D-CNN for ₦/USD Volatility Forecasting
# Task: Exploratory Data Analysis for final modelling dataset
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

dataset_file = cleaned_dir / "feature_engineered_fx_dataset_2006_2024.csv"


# -----------------------------
# 3. Load dataset
# -----------------------------
df = pd.read_csv(dataset_file)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print("\nDataset preview:")
print(df.head())

print("\nDataset tail:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())


# ============================================================
# 4. Descriptive statistics
# ============================================================

numeric_cols = [
    "usd_ngn_rate",
    "oil_price",
    "external_reserves",
    "mpr",
    "fx_return",
    "oil_return",
    "reserves_change",
    "mpr_change",
    "volatility",
    "spike"
]

desc_stats = df[numeric_cols].describe().T

desc_stats = desc_stats.rename(
    columns={
        "count": "Count",
        "mean": "Mean",
        "std": "Std Dev",
        "min": "Min",
        "25%": "25%",
        "50%": "Median",
        "75%": "75%",
        "max": "Max"
    }
)

print("\nDescriptive statistics:")
print(desc_stats)

desc_stats_path = results_dir / "eda_descriptive_statistics.csv"
desc_stats.to_csv(desc_stats_path)


# ============================================================
# 5. Correlation matrix
# ============================================================

corr_cols = [
    "usd_ngn_rate",
    "oil_price",
    "external_reserves",
    "mpr",
    "fx_return",
    "oil_return",
    "reserves_change",
    "mpr_change",
    "volatility"
]

corr_matrix = df[corr_cols].corr()

print("\nCorrelation matrix:")
print(corr_matrix)

corr_path = results_dir / "eda_correlation_matrix.csv"
corr_matrix.to_csv(corr_path)


# -----------------------------
# Correlation heatmap using matplotlib
# -----------------------------
plt.figure(figsize=(10, 8))

plt.imshow(corr_matrix, aspect="auto")
plt.colorbar(label="Correlation")

plt.xticks(
    ticks=np.arange(len(corr_matrix.columns)),
    labels=corr_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks=np.arange(len(corr_matrix.index)),
    labels=corr_matrix.index
)

# Add correlation values inside cells
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        plt.text(
            j,
            i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Correlation Matrix of Macroeconomic and FX Variables")
plt.tight_layout()

corr_plot_path = figures_dir / "eda_correlation_heatmap.png"
plt.savefig(corr_plot_path, dpi=300)
plt.show()


# ============================================================
# 6. Time-series plots
# ============================================================

# Exchange rate
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["usd_ngn_rate"])
plt.title("Monthly ₦/USD Exchange Rate, 2006–2024")
plt.xlabel("Date")
plt.ylabel("₦/USD Rate")
plt.grid(True)
plt.tight_layout()

exchange_plot_path = figures_dir / "eda_exchange_rate_trend.png"
plt.savefig(exchange_plot_path, dpi=300)
plt.show()


# Oil price
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["oil_price"])
plt.title("Monthly Brent Crude Oil Price, 2006–2024")
plt.xlabel("Date")
plt.ylabel("Oil Price ($/bbl)")
plt.grid(True)
plt.tight_layout()

oil_plot_path = figures_dir / "eda_oil_price_trend.png"
plt.savefig(oil_plot_path, dpi=300)
plt.show()


# External reserves
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["external_reserves"])
plt.title("Nigeria External Reserves, 2006–2024")
plt.xlabel("Date")
plt.ylabel("External Reserves (USD)")
plt.grid(True)
plt.tight_layout()

reserves_plot_path = figures_dir / "eda_external_reserves_trend.png"
plt.savefig(reserves_plot_path, dpi=300)
plt.show()


# MPR
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["mpr"])
plt.title("CBN Monetary Policy Rate, 2006–2024")
plt.xlabel("Date")
plt.ylabel("MPR (%)")
plt.grid(True)
plt.tight_layout()

mpr_plot_path = figures_dir / "eda_mpr_trend.png"
plt.savefig(mpr_plot_path, dpi=300)
plt.show()


# ============================================================
# 7. Return and volatility plots
# ============================================================

# FX return
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["fx_return"])
plt.title("Monthly ₦/USD Exchange Rate Return, 2006–2024")
plt.xlabel("Date")
plt.ylabel("FX Return (%)")
plt.grid(True)
plt.tight_layout()

fx_return_plot_path = figures_dir / "eda_fx_return.png"
plt.savefig(fx_return_plot_path, dpi=300)
plt.show()


# Oil return
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["oil_return"])
plt.title("Monthly Brent Crude Oil Return, 2006–2024")
plt.xlabel("Date")
plt.ylabel("Oil Return (%)")
plt.grid(True)
plt.tight_layout()

oil_return_plot_path = figures_dir / "eda_oil_return.png"
plt.savefig(oil_return_plot_path, dpi=300)
plt.show()


# Volatility
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["volatility"])
plt.title("Rolling ₦/USD Exchange Rate Volatility, 2006–2024")
plt.xlabel("Date")
plt.ylabel("Rolling Volatility")
plt.grid(True)
plt.tight_layout()

volatility_plot_path = figures_dir / "eda_fx_volatility.png"
plt.savefig(volatility_plot_path, dpi=300)
plt.show()


# ============================================================
# 8. Spike distribution
# ============================================================

spike_counts = df["spike"].value_counts().sort_index()

spike_summary = pd.DataFrame({
    "spike": spike_counts.index,
    "count": spike_counts.values,
    "percentage": (spike_counts.values / len(df)) * 100
})

print("\nSpike distribution:")
print(spike_summary)

spike_summary_path = results_dir / "eda_spike_distribution.csv"
spike_summary.to_csv(spike_summary_path, index=False)


plt.figure(figsize=(8, 5))
plt.bar(spike_summary["spike"].astype(str), spike_summary["count"])
plt.title("Distribution of Volatility Spike Classes")
plt.xlabel("Spike Class")
plt.ylabel("Count")
plt.xticks(
    ticks=[0, 1],
    labels=["Normal Periods (0)", "Spike Periods (1)"]
)
plt.grid(axis="y")
plt.tight_layout()

spike_bar_path = figures_dir / "eda_spike_distribution_bar.png"
plt.savefig(spike_bar_path, dpi=300)
plt.show()


# ============================================================
# 9. Volatility spike plot
# ============================================================

spike_dates = df[df["spike"] == 1]

spike_threshold = df["volatility"].quantile(0.75)

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["volatility"], label="Volatility")
plt.scatter(
    spike_dates["Date"],
    spike_dates["volatility"],
    label="Detected Spike"
)
plt.axhline(spike_threshold, linestyle="--", label="75th Percentile Threshold")

plt.title("Detected ₦/USD Volatility Spike Periods, 2006–2024")
plt.xlabel("Date")
plt.ylabel("Rolling Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()

spike_plot_path = figures_dir / "eda_detected_volatility_spikes.png"
plt.savefig(spike_plot_path, dpi=300)
plt.show()


# ============================================================
# 10. Save key spike periods
# ============================================================

spike_periods = df[df["spike"] == 1][
    [
        "Date",
        "year_month",
        "usd_ngn_rate",
        "fx_return",
        "volatility",
        "oil_price",
        "external_reserves",
        "mpr"
    ]
].copy()

spike_periods_path = results_dir / "eda_detected_spike_periods.csv"
spike_periods.to_csv(spike_periods_path, index=False)

print("\nDetected spike periods preview:")
print(spike_periods.head())

print("\nNumber of detected spike months:")
print(len(spike_periods))


# ============================================================
# 11. Save cleaned EDA-ready dataset copy
# ============================================================

eda_dataset_path = results_dir / "eda_ready_fx_dataset_2006_2024.csv"
df.to_csv(eda_dataset_path, index=False)


# ============================================================
# 12. Final summary
# ============================================================

print("\nEDA completed successfully.")

print("\nSaved tables:")
print(desc_stats_path)
print(corr_path)
print(spike_summary_path)
print(spike_periods_path)
print(eda_dataset_path)

print("\nSaved figures:")
print(corr_plot_path)
print(exchange_plot_path)
print(oil_plot_path)
print(reserves_plot_path)
print(mpr_plot_path)
print(fx_return_plot_path)
print(oil_return_plot_path)
print(volatility_plot_path)
print(spike_bar_path)
print(spike_plot_path)
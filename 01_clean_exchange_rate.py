# ============================================================
# 01_clean_exchange_rate.py
# Project: Multivariate 2D-CNN for ₦/USD Volatility Forecasting
# Task: Clean and merge CBN exchange-rate datasets from 2004–2024
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
BASE_DIR = Path(".")  # current folder

monthly_file = BASE_DIR / "datasets/Monthly_Average_Exchange_Rates_Data_in_Excel.xlsx"
daily_2022_2023_file = BASE_DIR / "datasets/ranges.xlsx"
daily_2024_file = BASE_DIR / "datasets/2024.xlsx"
daily_2021_file = BASE_DIR / "datasets/2021.xlsx"

output_dir = BASE_DIR / "data_cleaned"
figures_dir = BASE_DIR / "figures"

output_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)


# -----------------------------
# 3. Helper function
# -----------------------------
def clean_numeric(series):
    """
    Convert messy numeric columns to proper float values.
    Handles commas, spaces, and non-numeric characters.
    """
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace(["nan", "None", "", "-"], np.nan)
        .astype(float)
    )


# ============================================================
# 4. Clean monthly CBN exchange-rate data: 2004–2021
# ============================================================

monthly_df = pd.read_excel(monthly_file)

print("\nMonthly file columns:")
print(monthly_df.columns.tolist())

# Keep useful columns
# We are using IFEM Dollar as the official ₦/USD monthly rate
monthly_clean = monthly_df[["tyear", "tmonth", "ifemDollar"]].copy()

# Rename columns
monthly_clean = monthly_clean.rename(
    columns={
        "tyear": "year",
        "tmonth": "month",
        "ifemDollar": "usd_ngn_rate"
    }
)

# Clean numeric values
monthly_clean["usd_ngn_rate"] = clean_numeric(monthly_clean["usd_ngn_rate"])

# Create proper monthly date
monthly_clean["Date"] = pd.to_datetime(
    monthly_clean["year"].astype(int).astype(str) + "-" +
    monthly_clean["month"].astype(int).astype(str) + "-01",
    errors="coerce"
)

# Keep only required columns
monthly_clean = monthly_clean[["Date", "usd_ngn_rate"]]

# Sort
monthly_clean = monthly_clean.sort_values("Date").reset_index(drop=True)

print("\nMonthly cleaned preview:")
print(monthly_clean.head())
print(monthly_clean.tail())


# ============================================================
# 4.2. Clean daily CBN exchange-rate data: 2021
# ============================================================

daily_2021 = pd.read_excel(daily_2021_file)

print("\n2021 daily file columns:")
print(daily_2021.columns.tolist())

daily_2021_clean = daily_2021.copy()

# Convert date
daily_2021_clean["Date"] = pd.to_datetime(
    daily_2021_clean["Date"],
    errors="coerce"
)

# Use Central Rate as ₦/USD rate
daily_2021_clean["usd_ngn_rate"] = clean_numeric(
    daily_2021_clean["Central Rate"]
)

# Drop rows with missing date or rate
daily_2021_clean = daily_2021_clean.dropna(
    subset=["Date", "usd_ngn_rate"]
)

# Convert daily data to monthly average
daily_2021_monthly = (
    daily_2021_clean
    .set_index("Date")
    .resample("MS")["usd_ngn_rate"]
    .mean()
    .reset_index()
)

# We only need May 2021 to December 2021
# because Jan-Apr 2021 already exists in the monthly file
daily_2021_monthly = daily_2021_monthly[
    (daily_2021_monthly["Date"] >= "2021-05-01") &
    (daily_2021_monthly["Date"] <= "2021-12-01")
].copy()

print("\n2021 monthly converted preview:")
print(daily_2021_monthly.head())
print(daily_2021_monthly.tail())


# ============================================================
# 5. Clean daily CBN exchange-rate data: 2022–2023
# ============================================================

daily_2022_2023 = pd.read_excel(daily_2022_2023_file)

print("\n2022–2023 daily file columns:")
print(daily_2022_2023.columns.tolist())

# We expect columns like:
# Currency, Date, Buying Rate, Central Rate, Selling Rate
daily_2022_2023_clean = daily_2022_2023.copy()

# Convert date
daily_2022_2023_clean["Date"] = pd.to_datetime(
    daily_2022_2023_clean["Date"],
    errors="coerce"
)

# Use Central Rate as ₦/USD rate
daily_2022_2023_clean["usd_ngn_rate"] = clean_numeric(
    daily_2022_2023_clean["Central Rate"]
)

# Drop rows with missing date or rate
daily_2022_2023_clean = daily_2022_2023_clean.dropna(
    subset=["Date", "usd_ngn_rate"]
)

# Convert daily data to monthly average
daily_2022_2023_monthly = (
    daily_2022_2023_clean
    .set_index("Date")
    .resample("MS")["usd_ngn_rate"]
    .mean()
    .reset_index()
)

print("\n2022–2023 monthly converted preview:")
print(daily_2022_2023_monthly.head())
print(daily_2022_2023_monthly.tail())


# ============================================================
# 6. Clean daily CBN exchange-rate data: 2024
# ============================================================

daily_2024 = pd.read_excel(daily_2024_file)

print("\n2024 daily file columns:")
print(daily_2024.columns.tolist())

daily_2024_clean = daily_2024.copy()

# Convert date
daily_2024_clean["Date"] = pd.to_datetime(
    daily_2024_clean["Date"],
    errors="coerce"
)

# Use Central Rate as ₦/USD rate
daily_2024_clean["usd_ngn_rate"] = clean_numeric(
    daily_2024_clean["Central Rate"]
)

# Drop rows with missing date or rate
daily_2024_clean = daily_2024_clean.dropna(
    subset=["Date", "usd_ngn_rate"]
)

# Convert daily data to monthly average
daily_2024_monthly = (
    daily_2024_clean
    .set_index("Date")
    .resample("MS")["usd_ngn_rate"]
    .mean()
    .reset_index()
)

print("\n2024 monthly converted preview:")
print(daily_2024_monthly.head())
print(daily_2024_monthly.tail())


# ============================================================
# 7. Merge all exchange-rate data
# ============================================================

exchange_rate = pd.concat(
    [
        monthly_clean,
        daily_2021_monthly,
        daily_2022_2023_monthly,
        daily_2024_monthly
    ],
    ignore_index=True
)

# Remove duplicate months if any
exchange_rate["Date"] = pd.to_datetime(exchange_rate["Date"])
exchange_rate = exchange_rate.sort_values("Date")

exchange_rate = (
    exchange_rate
    .groupby("Date", as_index=False)["usd_ngn_rate"]
    .mean()
)

# Restrict to study period: 2004–2024
exchange_rate = exchange_rate[
    (exchange_rate["Date"] >= "2004-01-01") &
    (exchange_rate["Date"] <= "2024-12-01")
].copy()

# Create year-month column for easier viewing
exchange_rate["year_month"] = exchange_rate["Date"].dt.to_period("M").astype(str)

# Reorder columns
exchange_rate = exchange_rate[["Date", "year_month", "usd_ngn_rate"]]

print("\nFinal exchange-rate dataset preview:")
print(exchange_rate.head())
print(exchange_rate.tail())

print("\nFinal date range:")
print(exchange_rate["Date"].min(), "to", exchange_rate["Date"].max())

print("\nNumber of monthly observations:")
print(len(exchange_rate))


# ============================================================
# 8. Check for missing months
# ============================================================

expected_months = pd.date_range(
    start=exchange_rate["Date"].min(),
    end=exchange_rate["Date"].max(),
    freq="MS"
)

available_months = pd.DatetimeIndex(exchange_rate["Date"])

missing_months = expected_months.difference(available_months)

print("\nMissing months:")
if len(missing_months) == 0:
    print("No missing months. Good.")
else:
    print(missing_months)


# ============================================================
# 9. Check for missing values
# ============================================================

print("\nMissing values count:")
print(exchange_rate.isna().sum())


# Optional: forward-fill missing exchange rates if any exist
# But we will only do this if there are missing values inside existing rows.
exchange_rate["usd_ngn_rate"] = exchange_rate["usd_ngn_rate"].ffill()

# Check missing values AFTER filling
print("\nMissing values count after filling:")
print(exchange_rate.isna().sum())

print("\nRows still missing exchange rate:")
print(exchange_rate[exchange_rate["usd_ngn_rate"].isna()])
# ============================================================
# 10. Create basic exchange-rate plot
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(exchange_rate["Date"], exchange_rate["usd_ngn_rate"])
plt.title("Monthly ₦/USD Exchange Rate, 2004–2024")
plt.xlabel("Date")
plt.ylabel("₦/USD Rate")
plt.grid(True)
plt.tight_layout()

plot_path = figures_dir / "exchange_rate_trend_2004_2024.png"
plt.savefig(plot_path, dpi=300)
plt.show()


# ============================================================
# 11. Save cleaned exchange-rate dataset
# ============================================================

csv_output_path = output_dir / "clean_exchange_rate_2004_2024.csv"
excel_output_path = output_dir / "clean_exchange_rate_2004_2024.xlsx"

exchange_rate.to_csv(csv_output_path, index=False)
exchange_rate.to_excel(excel_output_path, index=False)

print("\nSaved cleaned exchange-rate dataset:")
print(csv_output_path)
print(excel_output_path)

print("\nSaved plot:")
print(plot_path)

print("\nExchange-rate cleaning completed successfully.")





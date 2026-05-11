# ============================================================
# 02_clean_oil_price.py
# Project: Multivariate 2D-CNN for ₦/USD Volatility Forecasting
# Task: Clean World Bank Pink Sheet Brent crude oil price data
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

oil_file = BASE_DIR / "datasets/cmo.xlsx"

output_dir = BASE_DIR / "data_cleaned"
figures_dir = BASE_DIR / "figures"

output_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)


# -----------------------------
# 3. Read the Excel file
# -----------------------------
# The World Bank Pink Sheet usually has the data in "Monthly Prices"
# Column codes like CRUDE_BRENT are around row 7.
oil_raw = pd.read_excel(
    oil_file,
    sheet_name="Monthly Prices",
    header=None
)

print("\nRaw oil file preview:")
print(oil_raw.head(10))


# -----------------------------
# 4. Extract column names
# -----------------------------
# In this file structure:
# Row 0-3: notes/title
# Row 4: commodity names
# Row 5: units
# Row 6: column codes
# Row 7 downward: actual monthly data

column_codes = oil_raw.iloc[6].tolist()

oil_data = oil_raw.iloc[7:].copy()
oil_data.columns = column_codes

print("\nOil data columns:")
print(oil_data.columns.tolist())


# -----------------------------
# 5. Select date and Brent crude oil column
# -----------------------------
# First column is usually the date column, with values like 1960M01
date_col = oil_data.columns[0]

# Brent crude oil column
brent_col = "CRUDE_BRENT"

oil_clean = oil_data[[date_col, brent_col]].copy()

oil_clean = oil_clean.rename(
    columns={
        date_col: "period",
        brent_col: "oil_price"
    }
)


# -----------------------------
# 6. Clean date column
# -----------------------------
# Convert values like 2004M01 to 2004-01-01

oil_clean["period"] = oil_clean["period"].astype(str).str.strip()

oil_clean["Date"] = pd.to_datetime(
    oil_clean["period"].str.replace("M", "-", regex=False) + "-01",
    errors="coerce"
)


# -----------------------------
# 7. Clean numeric oil price
# -----------------------------
oil_clean["oil_price"] = (
    oil_clean["oil_price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
    .replace(["nan", "None", "", "-"], np.nan)
    .astype(float)
)


# -----------------------------
# 8. Keep needed columns and filter study period
# -----------------------------
oil_clean = oil_clean[["Date", "oil_price"]]

oil_clean = oil_clean.dropna(subset=["Date", "oil_price"])

oil_clean = oil_clean[
    (oil_clean["Date"] >= "2004-01-01") &
    (oil_clean["Date"] <= "2024-12-01")
].copy()

oil_clean = oil_clean.sort_values("Date").reset_index(drop=True)

oil_clean["year_month"] = oil_clean["Date"].dt.to_period("M").astype(str)

oil_clean = oil_clean[["Date", "year_month", "oil_price"]]


# -----------------------------
# 9. Check result
# -----------------------------
print("\nClean oil price preview:")
print(oil_clean.head())
print(oil_clean.tail())

print("\nFinal date range:")
print(oil_clean["Date"].min(), "to", oil_clean["Date"].max())

print("\nNumber of monthly observations:")
print(len(oil_clean))

expected_months = pd.date_range(
    start="2004-01-01",
    end="2024-12-01",
    freq="MS"
)

available_months = pd.DatetimeIndex(oil_clean["Date"])

missing_months = expected_months.difference(available_months)

print("\nMissing months:")
if len(missing_months) == 0:
    print("No missing months. Good.")
else:
    print(missing_months)

print("\nMissing values count:")
print(oil_clean.isna().sum())


# -----------------------------
# 10. Plot oil price trend
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(oil_clean["Date"], oil_clean["oil_price"])
plt.title("Monthly Brent Crude Oil Price, 2004–2024")
plt.xlabel("Date")
plt.ylabel("Oil Price ($/bbl)")
plt.grid(True)
plt.tight_layout()

plot_path = figures_dir / "oil_price_trend_2004_2024.png"
plt.savefig(plot_path, dpi=300)
plt.show()


# -----------------------------
# 11. Save cleaned oil price dataset
# -----------------------------
csv_output_path = output_dir / "clean_oil_price_2004_2024.csv"
excel_output_path = output_dir / "clean_oil_price_2004_2024.xlsx"

oil_clean.to_csv(csv_output_path, index=False)
oil_clean.to_excel(excel_output_path, index=False)

print("\nSaved cleaned oil price dataset:")
print(csv_output_path)
print(excel_output_path)

print("\nSaved plot:")
print(plot_path)

print("\nOil price cleaning completed successfully.")
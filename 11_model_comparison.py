# ============================================================
# 11_model_comparison.py
# Project: Multivariate 2D-CNN for ₦/USD Volatility Forecasting
# Task: Compare ARIMA-GARCH, Bayesian VAR, and 2D-CNN model results
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

results_dir = BASE_DIR / "results"
figures_dir = BASE_DIR / "figures"

results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

arima_metrics_file = results_dir / "arima_garch_metrics.csv"
bvar_metrics_file = results_dir / "bayesian_var_metrics.csv"
cnn_metrics_file = results_dir / "cnn_2d_v3_metrics.csv"


# -----------------------------
# 3. Load model metrics
# -----------------------------
arima_metrics = pd.read_csv(arima_metrics_file)
bvar_metrics = pd.read_csv(bvar_metrics_file)
cnn_metrics = pd.read_csv(cnn_metrics_file)

print("\nARIMA-GARCH metrics:")
print(arima_metrics)

print("\nBayesian VAR metrics:")
print(bvar_metrics)

print("\n2D-CNN V3 metrics:")
print(cnn_metrics)


# -----------------------------
# 4. Combine metrics
# -----------------------------
comparison = pd.concat(
    [
        arima_metrics,
        bvar_metrics,
        cnn_metrics
    ],
    ignore_index=True
)

# Keep only common important metrics
main_metrics = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc"
]

comparison_main = comparison[main_metrics].copy()

# Round values for cleaner reporting
comparison_main_rounded = comparison_main.copy()

for col in ["accuracy", "precision", "recall", "f1_score", "auc"]:
    comparison_main_rounded[col] = comparison_main_rounded[col].round(4)

print("\nFinal model comparison:")
print(comparison_main_rounded)


# -----------------------------
# 5. Rank models
# -----------------------------
comparison_main_rounded["rank_by_accuracy"] = (
    comparison_main_rounded["accuracy"]
    .rank(ascending=False, method="dense")
    .astype(int)
)

comparison_main_rounded["rank_by_f1"] = (
    comparison_main_rounded["f1_score"]
    .rank(ascending=False, method="dense")
    .astype(int)
)

comparison_main_rounded["rank_by_auc"] = (
    comparison_main_rounded["auc"]
    .rank(ascending=False, method="dense")
    .astype(int)
)

print("\nModel ranking:")
print(comparison_main_rounded)


# -----------------------------
# 6. Identify best model by F1-score
# -----------------------------
best_model_row = comparison_main_rounded.sort_values(
    by="f1_score",
    ascending=False
).iloc[0]

best_model = best_model_row["model"]

print("\nBest model based on F1-score:")
print(best_model_row)


# -----------------------------
# 7. Save comparison table
# -----------------------------
comparison_output_path = results_dir / "final_model_comparison.csv"
comparison_main_rounded.to_csv(comparison_output_path, index=False)

print("\nSaved final model comparison table:")
print(comparison_output_path)


# ============================================================
# 8. Bar chart: Accuracy comparison
# ============================================================

plt.figure(figsize=(10, 6))
plt.bar(
    comparison_main_rounded["model"],
    comparison_main_rounded["accuracy"]
)

plt.title("Model Comparison by Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y")
plt.tight_layout()

accuracy_plot_path = figures_dir / "final_model_comparison_accuracy.png"
plt.savefig(accuracy_plot_path, dpi=300)
plt.show()


# ============================================================
# 9. Bar chart: F1-score and AUC comparison
# ============================================================

x = np.arange(len(comparison_main_rounded["model"]))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    comparison_main_rounded["f1_score"],
    width,
    label="F1-score"
)

plt.bar(
    x + width / 2,
    comparison_main_rounded["auc"],
    width,
    label="AUC"
)

plt.title("Model Comparison by F1-score and AUC")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(
    ticks=x,
    labels=comparison_main_rounded["model"],
    rotation=20,
    ha="right"
)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

f1_auc_plot_path = figures_dir / "final_model_comparison_f1_auc.png"
plt.savefig(f1_auc_plot_path, dpi=300)
plt.show()


# ============================================================
# 10. Grouped bar chart: All classification metrics
# ============================================================

metric_cols = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc"
]

x = np.arange(len(comparison_main_rounded["model"]))
width = 0.15

plt.figure(figsize=(12, 7))

for idx, metric in enumerate(metric_cols):
    plt.bar(
        x + (idx - 2) * width,
        comparison_main_rounded[metric],
        width,
        label=metric
    )

plt.title("Overall Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(
    ticks=x,
    labels=comparison_main_rounded["model"],
    rotation=20,
    ha="right"
)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

all_metrics_plot_path = figures_dir / "final_model_comparison_all_metrics.png"
plt.savefig(all_metrics_plot_path, dpi=300)
plt.show()


# ============================================================
# 11. Optional: Include CNN regression metrics
# ============================================================

cnn_regression_cols = []

for col in ["mae", "mse", "rmse"]:
    if col in cnn_metrics.columns:
        cnn_regression_cols.append(col)

if len(cnn_regression_cols) > 0:
    cnn_regression_metrics = cnn_metrics[["model"] + cnn_regression_cols].copy()

    for col in cnn_regression_cols:
        cnn_regression_metrics[col] = cnn_regression_metrics[col].round(4)

    cnn_regression_output_path = results_dir / "cnn_v3_regression_metrics.csv"
    cnn_regression_metrics.to_csv(cnn_regression_output_path, index=False)

    print("\nCNN V3 regression metrics:")
    print(cnn_regression_metrics)

    print("\nSaved CNN V3 regression metrics:")
    print(cnn_regression_output_path)


# ============================================================
# 12. Create written summary text file
# ============================================================

summary_text = f"""
Final Model Comparison Summary

The three models evaluated in this study are:
1. ARIMA-GARCH
2. Bayesian VAR
3. Multivariate 2D-CNN V3 Volatility Regression

The model with the best F1-score is: {best_model}

Model performance table:

{comparison_main_rounded.to_string(index=False)}

Interpretation:
ARIMA-GARCH and Bayesian VAR achieved stronger performance than the Multivariate 2D-CNN.
The ARIMA-GARCH model recorded the highest F1-score, making it the most effective model for detecting ₦/USD volatility spikes in this experiment.
The CNN model improved when reformulated as a volatility regression model, but it still underperformed the classical models, likely due to the small monthly dataset size.
"""

summary_path = results_dir / "final_model_comparison_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write(summary_text)

print("\nSaved written summary:")
print(summary_path)


# ============================================================
# 13. Final output
# ============================================================

print("\nSaved figures:")
print(accuracy_plot_path)
print(f1_auc_plot_path)
print(all_metrics_plot_path)

print("\nModel comparison completed successfully.")
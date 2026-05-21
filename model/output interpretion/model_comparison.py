# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# ── STEP 1: FIX THE PATH FIRST ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

def ensure_import_path():
    sys.path.insert(0, str(ROOT_DIR))
ensure_import_path() 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# ── STEP 2: NOW IMPORT YOUR PROJECT MODULES ──────────────────────────────────
from model.model.linear_regression import lr_results, lr_model
from model.model.Gradient_Boosting import gb_results, gb_model
from model.model.LightGBM import lgbm_results, lgbm_model
from model.model.XGBoost import xgb_results, xgb_model
from model.model.LSTM import lstm_results, lstm_model
from model.model.histogram_gradient_boosting import HGB_results, HGB_model
from model.model.ensemble_1 import ensemble_XGBoost_HGB_results, ensemble_XGBoost_HGB
from model.model.ensemble_2 import ensemble_GB_HGB_results, ensemble_GB_HGB
from tabulate import tabulate
from model.data.input_setup import y_test

# ── Improved Model Comparison Table ─────────────────────────────────────────

results_df = pd.DataFrame([

    {
        'Model': lr_results['Model'],
        'MAE': lr_results['MAE'],
        'RMSE': lr_results['RMSE'],
        'R²': lr_results['R2']
    },

    {
        'Model': lstm_results['Model'],
        'MAE': lstm_results['MAE'],
        'RMSE': lstm_results['RMSE'],
        'R²': lstm_results['R2']
    },

    {
        'Model': lgbm_results['Model'],
        'MAE': lgbm_results['MAE'],
        'RMSE': lgbm_results['RMSE'],
        'R²': lgbm_results['R2']
    },

    {
        'Model': gb_results['Model'],
        'MAE': gb_results['MAE'],
        'RMSE': gb_results['RMSE'],
        'R²': gb_results['R2']
    },

    {
        'Model': xgb_results['Model'],
        'MAE': xgb_results['MAE'],
        'RMSE': xgb_results['RMSE'],
        'R²': xgb_results['R2']
    },

    {
        'Model': HGB_results['Model'],
        'MAE': HGB_results['MAE'],
        'RMSE': HGB_results['RMSE'],
        'R²': HGB_results['R2']
    },

    {
        'Model': ensemble_XGBoost_HGB_results['Model'],
        'MAE': ensemble_XGBoost_HGB_results['MAE'],
        'RMSE': ensemble_XGBoost_HGB_results['RMSE'],
        'R²': ensemble_XGBoost_HGB_results['R2']
    },

    {
        'Model': ensemble_GB_HGB_results['Model'],
        'MAE': ensemble_GB_HGB_results['MAE'],
        'RMSE': ensemble_GB_HGB_results['RMSE'],
        'R²': ensemble_GB_HGB_results['R2']
    }

]).sort_values(
    'R²',
    ascending=False
).reset_index(drop=True)

# ── Add Ranking Column ──────────────────────────────────────────────────────

results_df.index = results_df.index + 1

results_df.index.name = 'Rank'

# ── Format Numeric Columns ──────────────────────────────────────────────────

formatted_df = results_df.copy()

formatted_df['MAE'] = formatted_df['MAE'].map('{:.4f}'.format)

formatted_df['RMSE'] = formatted_df['RMSE'].map('{:.4f}'.format)

formatted_df['R²'] = formatted_df['R²'].map('{:.4f}'.format)

# ── Print Table ─────────────────────────────────────────────────────────────

print("\n🏆 MODEL PERFORMANCE COMPARISON")
print("=" * 95)

print(tabulate(
    formatted_df,
    headers='keys',
    tablefmt='fancy_grid'
))

print("=" * 95)
# ── Export Results to CSV ───────────────────────────────────────────────────

# 1. Define the output file path inside your project structure
csv_file_path = ROOT_DIR / 'model' / 'model_performance_comparison.csv'

# 2. Save the dataframe to CSV (index=True includes your 'Rank' column!)
results_df.to_csv(csv_file_path, index=True)

print(f"💾 Results successfully exported to CSV at:\n   {csv_file_path}")
# ── Best Model ──────────────────────────────────────────────────────────────

best_model_name = results_df.iloc[0]['Model']

best_r2 = results_df.iloc[0]['R²']

print(f"\n🥇 Best Model: {best_model_name}")

print(f"📈 Best R² Score: {best_r2:.4f} ({best_r2*100:.2f}% accuracy)")

# ── Visualize: Actual vs Predicted (Automatically Uses Best Model) ──────────

# ── Dictionary of All Model Results ──────────────────────────────────────────

all_results = {
    lr_results['Model']: lr_results,
    lstm_results['Model']: lstm_results,
    lgbm_results['Model']: lgbm_results,
    gb_results['Model']: gb_results,
    xgb_results['Model']: xgb_results,
    HGB_results['Model']: HGB_results,
    ensemble_XGBoost_HGB_results['Model']: ensemble_XGBoost_HGB_results,
    ensemble_GB_HGB_results['Model']: ensemble_GB_HGB_results
}

all_models = {
    lr_results['Model']: lr_model,
    lgbm_results['Model']: lgbm_model,
    gb_results['Model']: gb_model,
    xgb_results['Model']: xgb_model,
    HGB_results['Model']: HGB_model,
    ensemble_XGBoost_HGB_results['Model']: ensemble_XGBoost_HGB,
    ensemble_GB_HGB_results['Model']: ensemble_GB_HGB
}

# ── Get Best Model Automatically ─────────────────────────────────────────────
best_model_name = results_df.iloc[0]['Model']
best_model_results = all_results[best_model_name]
y_pred_best = best_model_results['Predictions']

# Flatten predictions if needed
y_pred_best = np.array(y_pred_best).flatten()

# ── Create Figure ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# ── Scatter Plot: Actual vs Predicted ───────────────────────────────────────
sample_size = min(2000, len(y_test))
idx = np.random.choice(len(y_test), sample_size, replace=False)
axes[0].scatter(
    y_test.values[idx],
    y_pred_best[idx],
    alpha=0.3,
    color='steelblue',
    s=10
)
axes[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2,
    label='Perfect Prediction'
)
axes[0].set_title(
    f'Actual vs Predicted Demand\n({best_model_name})',
    fontsize=13,
    fontweight='bold'
)
axes[0].set_xlabel('Actual Demand')

axes[0].set_ylabel('Predicted Demand')

axes[0].legend()

# ── Error Distribution ───────────────────────────────────────────────────────

errors = y_test.values - y_pred_best

axes[1].hist(
    errors,
    bins=60,
    color='coral',
    edgecolor='white',
    alpha=0.8
)

axes[1].axvline(
    0,
    color='black',
    linestyle='--',
    linewidth=2,
    label='Zero Error'
)

axes[1].set_title(
    'Prediction Error Distribution\n(Centered near 0 = good!)',
    fontsize=13,
    fontweight='bold'
)

axes[1].set_xlabel('Prediction Error (Actual - Predicted)')

axes[1].set_ylabel('Frequency')

axes[1].legend()

# ── Final Layout ─────────────────────────────────────────────────────────────
plt.tight_layout()
output_path_0 = ROOT_DIR / 'chart_prediction.png'
plt.savefig(
    output_path_0,
    dpi=150,
    bbox_inches='tight'
)
plt.show()

# ── Error Statistics ─────────────────────────────────────────────────────────

print(f"\n📊 Error Stats for Best Model: {best_model_name}")
print(f"   Mean Error:   {errors.mean():.2f} units")
print(f"   Std Dev:      {errors.std():.2f} units")
print(f"   Within ±20:   {(np.abs(errors) <= 20).mean()*100:.1f}% of predictions")

# ── Compare Actual vs Predicted for ALL Models ──────────────────────────────
# ── Create Subplots ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 2, figsize=(22, 10))

axes = axes.flatten()

# ── Sample Data for Cleaner Visualization ──────────────────────────────────

sample_size = min(1500, len(y_test))

idx = np.random.choice(
    len(y_test),
    sample_size,
    replace=False
)

# ── Plot Each Model ─────────────────────────────────────────────────────────

for ax, (model_name, results) in zip(axes, all_results.items()):

    y_pred = np.array(results['Predictions']).flatten()

    ax.scatter(
        y_test.values[idx],
        y_pred[idx],
        alpha=0.3,
        s=10
    )

    # Perfect prediction line
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--',
        linewidth=1
    )

    # Get R²
    r2 = results['R2']

    ax.set_title(
        f"{model_name}\nR² = {r2:.4f}",
        fontsize=11,
        fontweight='bold'
    )

    ax.set_xlabel("Actual")

    ax.set_ylabel("Predicted")

# ── Overall Layout ──────────────────────────────────────────────────────────

plt.suptitle(
    "Actual vs Predicted Comparison Across All Models",
    fontsize=18,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path_1 = ROOT_DIR / 'actual_predicted_all_models.png'
plt.savefig(
    output_path_1,
    dpi=150,
    bbox_inches='tight'
)
plt.show()

# ── Line Chart: Actual vs Predicted (Index 150 → 200) ──────────────────────
# ── Select Smaller Range for Clear Comparison ──────────────────────────────
start_idx = 150
end_idx = 200
x = np.arange(start_idx, end_idx)
actual_values = y_test.values[start_idx:end_idx]

# ── Create Wider Figure ────────────────────────────────────────────────────
plt.figure(figsize=(24, 12))   # Increased width

# ── Plot Actual Values ─────────────────────────────────────────────────────

plt.plot(
    x,
    actual_values,
    linewidth=4,
    label='Actual Values'
)

# ── Plot Predictions for Each Model ────────────────────────────────────────

for model_name, results in all_results.items():
    y_pred = np.array(results['Predictions']).flatten()
    y_pred_range = y_pred[start_idx:end_idx]
    r2 = results['R2']
    plt.plot(
        x,
        y_pred_range,
        linewidth=2,
        alpha=0.9,
        label=f"{model_name} (R²={r2:.3f})"
    )

# ── Styling ────────────────────────────────────────────────────────────────

plt.title(
    'Actual vs Predicted Demand Comparison\n(Index 150 → 200)',
    fontsize=18,
    fontweight='bold'
)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Demand', fontsize=12)
plt.grid(True, alpha=0.3)

# ── Put Legend INSIDE Figure ──────────────────────────────────────────────

plt.legend(
    loc='upper right',     # Inside plot
    fontsize=10,
    frameon=True
)
plt.tight_layout()

# ── Save Figure ────────────────────────────────────────────────────────────
output_path_2 = ROOT_DIR / 'model_comparison_150_200.png'
plt.savefig(
    output_path_2,
    dpi=150
)
plt.show()
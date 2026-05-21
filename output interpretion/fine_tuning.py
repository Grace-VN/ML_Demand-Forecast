# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# ── STEP 1: FIX THE PATH FIRST ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

def ensure_import_path():
    sys.path.insert(0, str(ROOT_DIR))
ensure_import_path() 

import numpy as np

# ── STEP 2: NOW IMPORT YOUR PROJECT MODULES ──────────────────────────────────
from model.data.input_setup import X_train, X_test, y_train, y_test
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from model.model.ensemble_2 import ensemble_GB_HGB_results    
import pandas as pd
import matplotlib.pyplot as plt


# ── Hyperparameter Tuning for Hybrid Ensemble ──────────────────────────────

# ── Base Models ─────────────────────────────────────────────────────────────

gb_base = GradientBoostingRegressor(
    loss='squared_error',
    random_state=42
)

HGB_base = HistGradientBoostingRegressor(
    random_state=42
)

# ── Hybrid Ensemble ─────────────────────────────────────────────────────────

hybrid_ensemble = VotingRegressor([

    ('gb', gb_base),

    ('HGB', HGB_base)

])

# ── Parameter Space ─────────────────────────────────────────────────────────

param_grid = {

    # GB parameters
    'gb__n_estimators': [100, 200, 300],

    'gb__max_depth': [4, 6, 8],

    'gb__learning_rate': [0.1, 0.2, 0.5],

    # HGB parameters
    'HGB__max_iter': [100, 200, 300],

    'HGB__max_depth': [4, 6, 8],

    'HGB__learning_rate': [0.1, 0.2, 0.5],

    # Ensemble weights
    'weights': [
        [1, 1],
        [2, 1],
        [1, 2],
        [1, 3],
        [3, 1],
        [1, 3],
        [3, 4],
        [4, 3],
        [1, 4]
    ]
}

# ── Random Search ───────────────────────────────────────────────────────────

search = RandomizedSearchCV(

    estimator=hybrid_ensemble,

    param_distributions=param_grid,

    n_iter=30,

    scoring='neg_mean_absolute_error',

    cv=8,

    verbose=1,

    random_state=42,

    n_jobs=-1
)

# ── Train Search ────────────────────────────────────────────────────────────

search.fit(X_train, y_train)

# ── Best Model ──────────────────────────────────────────────────────────────

best_hybrid_model = search.best_estimator_

print("🏆 Best Parameters:")
print(search.best_params_)

print(f"\nBest CV MAE: {search.best_score_:.4f}")

# ── Predictions ─────────────────────────────────────────────────────────────

y_pred = best_hybrid_model.predict(X_test)

# ── Evaluation ──────────────────────────────────────────────────────────────

mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

# ── Store Results ───────────────────────────────────────────────────────────

tuned_hybrid_results = {

    'Model': 'Tuned Hybrid Ensemble',

    'MAE': mae,

    'RMSE': rmse,

    'R2': r2,

    'Predictions': y_pred
}

# ── Final Results ───────────────────────────────────────────────────────────

print("\n🤖 Tuned Hybrid Ensemble")
print(f"   MAE  : {mae:.2f}")
print(f"   RMSE : {rmse:.2f}")
print(f"   R²   : {r2:.4f}")
# ── Original v.s. Fine-tuned version Comparison ────────────────────────────────────────────────

# ── Helper function ────────────────────────────────────────────────
def percent_change(new, old):
    return ((new - old) / abs(old)) * 100
    
base = ensemble_GB_HGB_results
tuned = tuned_hybrid_results

comparison = {

    "Metric": ["MAE", "RMSE", "R²"],

    "Ensemble 2 (Base)": [
        base["MAE"],
        base["RMSE"],
        base["R2"]
    ],
    "Tuned Hybrid": [
        tuned["MAE"],
        tuned["RMSE"],
        tuned["R2"]
    ],
    "Percentage of Improvement (%)": [
        percent_change(base["MAE"], tuned["MAE"]),
        percent_change(base["RMSE"], tuned["RMSE"]),
        percent_change(tuned["R2"], base["R2"])
    ]
}
df_compare = pd.DataFrame(comparison)

# ── Formatting ────────────────────────────────────────────────
df_compare["Ensemble 2 (Base)"] = df_compare["Ensemble 2 (Base)"].map(lambda x: f"{x:.4f}")
df_compare["Tuned Hybrid"] = df_compare["Tuned Hybrid"].map(lambda x: f"{x:.4f}")
df_compare["Percentage of Improvement (%)"] = df_compare["Percentage of Improvement (%)"].map(lambda x: f"{x:.2f}%")

# ── Display ────────────────────────────────────────────────────
print("\n📊 Ensemble Comparison: Base vs Tuned Model")
print("=" * 70)
print(df_compare.to_string(index=False))
print("=" * 70)
# ── Original v.s. Fine-tuned version Plotting ────────────────────────────────────────────────

y_pred_base = np.array(ensemble_GB_HGB_results['Predictions']).flatten()
y_pred_tuned = np.array(tuned_hybrid_results['Predictions']).flatten()
y_true = np.array(y_test).flatten()

# ── Select range 150 to 200 ─────────────────────────────────────

start, end = 150, 200

idx = np.arange(start, end)

# Safety check (avoid out-of-range errors)
idx = idx[idx < len(y_true)]

# ── Slice data ──────────────────────────────────────────────────

y_true_slice = y_true[idx]
base_slice = y_pred_base[idx]
tuned_slice = y_pred_tuned[idx]

# ── Plot ───────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(idx, y_true_slice, label="Actual", linewidth=2, color='black')
plt.plot(idx, base_slice, label="Original Ensemble", linestyle='--', color='steelblue')
plt.plot(idx, tuned_slice, label="Tuned Ensemble", linestyle='--', color='darkorange')

# ── Styling ────────────────────────────────────────────────────

plt.title("Actual vs Predicted (Tuned vs Original Models - Index 150 to 200)")
plt.xlabel("Index")
plt.ylabel("Demand")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
output_path = ROOT_DIR / 'actual_predicted_tuned_original.png'
plt.savefig(output_path)
plt.show()
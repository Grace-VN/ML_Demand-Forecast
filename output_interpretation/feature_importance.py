# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# ── Permutation Importance for Best Model ───────────────────────────────────
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# ── STEP 1: FIX THE PATH FIRST ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

def ensure_import_path():
    sys.path.insert(0, str(ROOT_DIR))
ensure_import_path() 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── STEP 2: NOW IMPORT YOUR PROJECT MODULES ──────────────────────────────────
from model.linear_regression import lr_results, lr_model
from model.Gradient_Boosting import gb_results, gb_model
from model.LightGBM import lgbm_results, lgbm_model
from model.XGBoost import xgb_results, xgb_model
from model.LSTM import lstm_results, lstm_model
from model.histogram_gradient_boosting import HGB_results, HGB_model
from model.ensemble_1 import ensemble_XGBoost_HGB_results, ensemble_XGBoost_HGB
from model.ensemble_2 import ensemble_GB_HGB_results, ensemble_GB_HGB
from input_processing.input_setup import y_test, X_test, feature_columns
from model.LSTM import X_test_lstm
from output_interpretation import results_df


# ── Store ALL Trained Models ────────────────────────────────────────────────
# IMPORTANT:
# These variables must be your ACTUAL trained model objects

all_models = {

    lr_results['Model']: lr_model,

    lgbm_results['Model']: lgbm_model,

    lstm_results['Model']: lstm_model,

    gb_results['Model']: gb_model,

    xgb_results['Model']: xgb_model,

    HGB_results['Model']: HGB_model,

    ensemble_XGBoost_HGB_results['Model']: ensemble_XGBoost_HGB,

    ensemble_GB_HGB_results['Model']: ensemble_GB_HGB
}

# ── Get Best Model Name from Comparison Table ───────────────────────────────

best_model_name = results_df.iloc[0]['Model']

print(f"🏆 Best Model: {best_model_name}")

# ── Get Actual Model Object ─────────────────────────────────────────────────

best_model = all_models[best_model_name]

# ── Choose Correct Test Data ────────────────────────────────────────────────
# LSTM uses 3D data, sklearn models use 2D

if 'LSTM' in best_model_name:

    X_perm = X_test_lstm

else:

    X_perm = X_test

# ── Calculate Permutation Importance ────────────────────────────────────────

perm = permutation_importance(
    estimator=best_model,
    X=X_perm,
    y=y_test,
    n_repeats=5,
    random_state=42,
    scoring='r2'
)

# ── Create Importance DataFrame ─────────────────────────────────────────────

perm_df = pd.DataFrame({

    'Feature': feature_columns,

    'Importance': perm.importances_mean

}).sort_values(
    'Importance',
    ascending=False
)

# ── Plot Top Features ───────────────────────────────────────────────────────

top_n = 15

top_features = perm_df.head(top_n)

plt.figure(figsize=(12, 7))

bars = plt.barh(
    top_features['Feature'][::-1],
    top_features['Importance'][::-1]
)

plt.title(
    f'Permutation Importance\n({best_model_name})',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Importance Score')

# ── Add Labels ──────────────────────────────────────────────────────────────

for bar, val in zip(
    bars,
    top_features['Importance'][::-1]
):

    plt.text(
        bar.get_width() + 0.001,
        bar.get_y() + bar.get_height()/2,
        f'{val:.3f}',
        va='center',
        fontsize=9
    )

# ── Final Layout ────────────────────────────────────────────────────────────

plt.tight_layout()
output_path_2 = ROOT_DIR / 'chart_permutation_importance.png'
plt.savefig(
    output_path_2,
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# ── Print Top Features ──────────────────────────────────────────────────────

print(f"\n🔝 Top 10 Most Important Features ({best_model_name}):")

for rank, (_, row) in enumerate(
    perm_df.head(10).iterrows(),
    start=1
):

    print(
        f"   {rank}. "
        f"{row['Feature']:25s} → "
        f"{row['Importance']:.4f}"
    )
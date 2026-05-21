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
from model.data.input_setup import X_train, X_test, y_train, y_test, evaluate_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor
from model.model.Gradient_Boosting import gb_model
from model.model.histogram_gradient_boosting import HGB_model

# ── Model 7: Ensemble Model 2 ───────────────────────────────────────────────
ensemble_GB_HGB = VotingRegressor([
    ('gb', gb_model),
    ('HGB', HGB_model)
])
# Train ensemble
ensemble_GB_HGB.fit(X_train, y_train)
# Predictions
y_pred = ensemble_GB_HGB.predict(X_test)

# ── Evaluation ───────────────────────────────────────────────────────────────
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ── Store Results ────────────────────────────────────────────────────────────
ensemble_GB_HGB_results = {
    'Model': 'Ensemble 2 (GB + HGB)',
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2,
    'Predictions': y_pred
}

# ── Print Results ────────────────────────────────────────────────────────────

print(f"\n🤖 Ensemble Model 2: Gradient Boosting + HGB")
print(f"   MAE  : {mae:.2f}")
print(f"   RMSE : {rmse:.2f}")
print(f"   R²   : {r2:.4f}")
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
from input_processing.input_setup import X_train, X_test, y_train, y_test
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor
from model.XGBoost import xgb_model
from model.histogram_gradient_boosting import HGB_model

# ── Model 7: Ensemble Model 1 ───────────────────────────────────────────────────────────

ensemble_XGBoost_HGB = VotingRegressor([
    ('xgb', xgb_model),
    ('HGB', HGB_model)
])
# Train ensemble
ensemble_XGBoost_HGB.fit(X_train, y_train)
# Predictions
y_pred = ensemble_XGBoost_HGB.predict(X_test)
# ── Evaluation ───────────────────────────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ensemble_XGBoost_HGB_results = {
    'Model': 'Ensemble 1 (XGBoost + HGB)',
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2,
    'Predictions': y_pred
}
print(f"\n🤖 Ensemble model 1: XGBoost and Histogram-based Gradient Boosting")
print(f"   MAE  (Avg Error in units): {mae:.2f}")
print(f"   RMSE (Root Mean Sq Error): {rmse:.2f}")
print(f"   R²   (Accuracy Score):     {r2:.4f}  ({r2*100:.2f}% accuracy)")
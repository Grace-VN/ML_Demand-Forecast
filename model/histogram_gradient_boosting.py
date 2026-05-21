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
from sklearn.ensemble import HistGradientBoostingRegressor

# ── Model 6: HistGradientBoosting Regressor ───────────────────────────────────

if len(X_train.shape) == 3:
    X_train_HGB = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test_HGB = X_test.reshape(X_test.shape[0], X_test.shape[1])
else:
    X_train_HGB = X_train
    X_test_HGB = X_test
# ── Initialize Model ─────────────────────────────────────────────────────────
HGB_model = HistGradientBoostingRegressor(
    loss='squared_error',
    learning_rate=0.5,
    max_iter=300,
    max_depth=6,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=42
)
# ── Train Model ──────────────────────────────────────────────────────────────
HGB_model.fit(
    X_train_HGB,
    y_train
)
# ── Predictions ──────────────────────────────────────────────────────────────
y_pred = HGB_model.predict(X_test_HGB)

# ── Evaluation ───────────────────────────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
HGB_results = {
    'Model': 'Histogram-based Gradient Boosting (HGB)',
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2,
    'Predictions': y_pred
}
# ── Results ──────────────────────────────────────────────────────────────────
print(f"\n🤖 Histogram-based Gradient Boosting model")
print(f"   MAE  (Avg Error in units): {mae:.2f}")
print(f"   RMSE (Root Mean Sq Error): {rmse:.2f}")
print(f"   R²   (Accuracy Score):     {r2:.4f}  ({r2*100:.2f}% accuracy)")
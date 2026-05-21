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
from sklearn.linear_model import LinearRegression
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

# ── Model 5: XGBoost Regressor ─────────────────────────────────────────────────

# XGBoost requires 2D input
if len(X_train.shape) == 3:
    X_train_xgb = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test_xgb = X_test.reshape(X_test.shape[0], X_test.shape[1])
else:
    X_train_xgb = X_train
    X_test_xgb = X_test
# ── Initialize Model ─────────────────────────────────────────────────────────
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.5,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
# ── Train Model ──────────────────────────────────────────────────────────────

xgb_model.fit(
    X_train_xgb,
    y_train
)
# ── Predictions ──────────────────────────────────────────────────────────────
y_pred = xgb_model.predict(X_test_xgb)

# ── Evaluation ───────────────────────────────────────────────────────────────

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n🤖 XGBoost model")
print(f"   MAE  (Avg Error in units): {mae:.2f}")
print(f"   RMSE (Root Mean Sq Error): {rmse:.2f}")
print(f"   R²   (Accuracy Score):     {r2:.4f}  ({r2*100:.2f}% accuracy)")

# Store results
xgb_results = {
    'Model': 'XGBoost',
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2,
    'Predictions': y_pred
}
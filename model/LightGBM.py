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
from input_processing.input_setup import X_train, X_test, y_train, y_test, evaluate_model
from lightgbm import LGBMRegressor

# ── Model 3: LightGBM Regressor ────────────────────────────────────────────────
# LightGBM requires 2D input
if len(X_train.shape) == 3:
    X_train_lgbm = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test_lgbm = X_test.reshape(X_test.shape[0], X_test.shape[1])
else:
    X_train_lgbm = X_train
    X_test_lgbm = X_test

# ── Initialize Model ─────────────────────────────────────────────────────────

lgbm_model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.5,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ── Evaluate Model Using evaluate_model() ────────────────────────────────────

lgbm_results = evaluate_model(
    model=lgbm_model,
    X_train=X_train_lgbm,
    X_test=X_test_lgbm,
    y_train=y_train,
    y_test=y_test,
    model_name="LightGBM Regressor"
)
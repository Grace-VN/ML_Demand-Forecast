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
from sklearn.ensemble import GradientBoostingRegressor

# ── Model 4: Gradient Boosting ────────────────────────────────────────────────
X_train_gb = X_train
X_test_gb = X_test

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.5,
    max_depth=6,
    random_state=42
)

gb_results = evaluate_model(
    gb_model,
    X_train_gb,
    X_test_gb,
    y_train,
    y_test,
    'Gradient Boosting'
)


# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# ── STEP 1: FIX THE PATH FIRST ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

def ensure_import_path():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

ensure_import_path() 


# ── STEP 2: NOW IMPORT YOUR PROJECT MODULES ──────────────────────────────────
from sklearn.linear_model import LinearRegression
from input_processing.input_setup import X_train, X_test, y_train, y_test, evaluate_model


# ── STEP 3: RUN THE MODEL ────────────────────────────────────────────────────
# Draws a straight line through the data to make predictions
# Good as a baseline (starting point)

lr_model = LinearRegression()
lr_results = evaluate_model(lr_model, X_train, X_test, y_train, y_test, 
                            'Linear Regression')
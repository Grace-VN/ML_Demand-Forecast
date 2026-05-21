# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).parent.parent


def ensure_import_path():
    sys.path.insert(0, str(ROOT_DIR))
# If ensure_import_path() modifies sys.path, run it first:
ensure_import_path() 

# Now import df globally so it's available everywhere in this file
from model.data.data_loading import df 
from model.data.categorical_feature_encoding import df_encoded

# ── Define Features (X) and Target (y) ───────────────────────────────────────
# X = all the input columns the model will learn from
# y = the target column we want to predict (Demand)

feature_columns = [
    'Store ID', 'Product ID', 'Category', 'Region',
    'Inventory Level', 'Units Sold', 'Units Ordered',
    'Price', 'Discount', 'Weather Condition', 'Promotion',
    'Competitor Pricing', 'Seasonality', 'Epidemic',
    'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter',
    'WeekOfYear', 'Price_Diff', 'Effective_Price', 'Inventory_Ratio'
]

X = df_encoded[feature_columns]   # Input features
y = df_encoded['Demand']          # Target: what we want to predict

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape:   {y.shape}")
print(f"\n📌 We're using {len(feature_columns)} features to predict Demand")

# ── Split into Train and Test Sets ─────────────────────────────────────────────
# 80% of data for training, 20% for testing
# random_state=42 ensures we get the same split every time we run this

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20,      # 20% goes to testing
    random_state=42      # For reproducibility
)

print("📊 Data Split:")
print(f"   🏋️  Training samples: {len(X_train):,}  ({len(X_train)/len(X)*100:.0f}%)")
print(f"   🧪  Testing samples:  {len(X_test):,}   ({len(X_test)/len(X)*100:.0f}%)")

# ── Helper function to evaluate any model ────────────────────────────────────
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Trains a model and returns its performance metrics.
    
    Metrics explained:
    - MAE  (Mean Absolute Error)  → Average error in units (lower is better)
    - RMSE (Root Mean Sq. Error)  → Penalizes big mistakes more (lower is better)
    - R²   (R-squared Score)      → How much variance is explained (1.0 = perfect!)
    """
    model.fit(X_train, y_train)          # Train the model
    y_pred = model.predict(X_test)       # Make predictions on test data
    
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    
    print(f"\n🤖 {model_name}")
    print(f"   MAE  (Avg Error in units): {mae:.2f}")
    print(f"   RMSE (Root Mean Sq Error): {rmse:.2f}")
    print(f"   R²   (Accuracy Score):     {r2:.4f}  ({r2*100:.2f}% accuracy)")
    
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Predictions': y_pred}

print("✅ Evaluation function defined!")
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).parent.parent

def ensure_import_path():
    sys.path.insert(0, str(ROOT_DIR))
ensure_import_path() 

# Import original dataframe
from model.data.data_loading import df 

# ── STEP 1: Feature Engineering (CRUCIAL FIX) ─────────────────────────────────
print("🛠️ Engineering new features...")
df_encoded = df.copy()  # Make our working copy here

# A. Extract Date Features (Assuming your date column is named 'Date')
if 'Date' in df_encoded.columns:
    df_encoded['Date'] = pd.to_datetime(df_encoded['Date'])
    df_encoded['Year'] = df_encoded['Date'].dt.year
    df_encoded['Month'] = df_encoded['Date'].dt.month
    df_encoded['Day'] = df_encoded['Date'].dt.day
    df_encoded['DayOfWeek'] = df_encoded['Date'].dt.dayofweek
    df_encoded['Quarter'] = df_encoded['Date'].dt.quarter
    df_encoded['WeekOfYear'] = df_encoded['Date'].dt.isocalendar().week
else:
    print("⚠️ Warning: 'Date' column not found. Date features might be missing!")

# B. Calculate Pricing and Inventory Ratios
df_encoded['Effective_Price'] = df_encoded['Price'] - df_encoded['Discount']
df_encoded['Price_Diff'] = df_encoded['Price'] - df_encoded['Competitor Pricing']
# Adding +1 prevents a crash if Inventory Level is ever 0
df_encoded['Inventory_Ratio'] = df_encoded['Units Sold'] / (df_encoded['Inventory Level'] + 1)


# ── STEP 2: Label Encode all categorical (text) columns ───────────────────────
le = LabelEncoder()

# This will now safely catch any new text columns too
categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
print("\nAutomatically detected categorical columns:")
print(categorical_columns)

for col in categorical_columns:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str)) # astype(str) keeps it safe
    print(f"  ✅ Encoded: {col}")

print("\n🔢 After encoding — first 3 rows:")
print(df_encoded[categorical_columns].head())
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


def ensure_import_path():
    sys.path.insert(0, str(ROOT_DIR))
# If ensure_import_path() modifies sys.path, run it first:
ensure_import_path() 

# Now import df globally so it's available everywhere in this file
from input_processing.data_loading import df
from input_processing.categorical_feature_encoding import df_encoded

# You don't even need the load_data() function anymore!
print(df.head())
# ── Extract Date Features ──────────────────────────────────────────────────────
df['Year']        = df['Date'].dt.year          # e.g., 2022
df['Month']       = df['Date'].dt.month         # e.g., 1 = January
df['Day']         = df['Date'].dt.day           # e.g., 15
df['DayOfWeek']   = df['Date'].dt.dayofweek     # 0=Monday, 6=Sunday
df['Quarter']     = df['Date'].dt.quarter       # 1, 2, 3, or 4
df['WeekOfYear']  = df['Date'].dt.isocalendar().week.astype(int)  # Week number

# ── Create New Business Features ──────────────────────────────────────────────
# Price competitiveness: are we cheaper or expensive vs competitor?
df['Price_Diff']       = df['Price'] - df['Competitor Pricing']

# Effective price after discount
df['Effective_Price']  = df['Price'] * (1 - df['Discount'] / 100)

# Inventory per unit sold ratio
df['Inventory_Ratio']  = df['Inventory Level'] / (df['Units Sold'] + 1)  # +1 avoids division by 0

print("✅ New features created:")
new_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 
            'WeekOfYear', 'Price_Diff', 'Effective_Price', 'Inventory_Ratio']
print(df[new_cols].head(3))
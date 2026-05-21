# -*- coding: utf-8 -*-
import sys
import io
import pandas as pd

# Configure UTF-8 output for Windows compatibility
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Load the dataset ──────────────────────────────────────────────────────────

df = pd.read_csv('D:\Job\Portfolio\Machine Learning\Demand Forecast\demand_forecasting.csv')

# Convert 'Date' column from text to actual date format
df['Date'] = pd.to_datetime(df['Date'])

print(f"✅ Dataset loaded!")
print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"📅 Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print()
print("👀 First 5 rows:")
print(df.head())
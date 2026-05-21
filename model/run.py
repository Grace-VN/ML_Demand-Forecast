# -*- coding: utf-8 -*-
from tabulate import tabulate

# 1. Create some dummy model data
test_data = [
    ["1", "XGBoost", "0.9245"],
    ["2", "Linear Regression", "0.7120"],
    ["3", "LSTM", "0.8830"]
]

# 2. Define headers
col_headers = ["Rank", "Model Name", "R² Score"]

print("\n Testing Tabulate Installation:")
print("=" * 40)

# 3. Print using the same 'fancy_grid' format you have in your main script
print(tabulate(test_data, headers=col_headers, tablefmt="fancy_grid"))

print("=" * 40)
# ── Importing all the tools we need ──────────────────────────────────────────
# Install type stubs for the library in the terminal, not in Python code:
#   python -m pip install opencv-stubs

# ── Core Libraries ─────────────────────────────────────────────
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ── Scikit-learn Utilities ─────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Machine Learning Models ────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    HistGradientBoostingRegressor
)

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ── Deep Learning ──────────────────────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ── Utilities ──────────────────────────────────────────────────
from tabulate import tabulate

# --- Make charts look nice ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

from pathlib import Path
import pandas as pd

# 1. Get the directory where input_setup.py itself lives (\model)
MODEL_DIR = Path(__file__).resolve().parent

# 2. Point directly to the file sitting right next to this script
csv_path = MODEL_DIR / "demand_forecasting.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"❌ Cannot find dataset at: {csv_path}")

# 3. Read the file
df = pd.read_csv(csv_path)
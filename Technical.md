# Part 1 — Data Pipeline & Preprocessing

This section covers everything from raw data ingestion to a fully encoded, feature-rich dataset ready for model training.

---

## Folder Structure

```
input_processing/
├── data_loading.py
├── data_processing.py
├── data_summary.py
├── EDA.py
├── data_visualization.py
├── feature_engineering.py
├── categorical_feature_encoding.py
└── input_setup.py
```

---

## Pipeline Overview

```
data_loading.py
      ↓
data_processing.py       ← clean missing values & duplicates
      ↓
data_summary.py / EDA.py ← explore the dataset
      ↓
data_visualization.py    ← generate charts
      ↓
feature_engineering.py   ← create new features
      ↓
categorical_feature_encoding.py  ← encode text columns
      ↓
input_setup.py           ← define X, y and train/test split
```

---

## File Descriptions

### `data_loading.py`
Loads `demand_forecasting.csv` into a pandas DataFrame and parses the `Date` column.

- Prints dataset shape and date range on load
- **Note:** The CSV path is currently hardcoded for Windows (`D:\Job\...`). Update this path for your own environment.

**Dataset shape:** rows × 14 columns

---

### `data_processing.py`
Checks data quality before any modeling.

- Detects and reports missing values (count + percentage)
- Detects and removes duplicate rows
- Prints a clean confirmation if no issues are found

---

### `data_summary.py` / `EDA.py`
Both files produce the same exploratory output:

- Column names and data types (`df.info()`)
- Statistical summary for numeric columns (`df.describe()`)
- Unique values for categorical columns: `Store ID`, `Category`, `Region`, `Weather Condition`, `Seasonality`

> **Note:** These two files are nearly identical. They can be consolidated into one in a future refactor.

---

### `data_visualization.py`
Generates and saves 3 charts to the project root:

| Output File | Description |
|---|---|
| `chart_demand_distribution.png` | Histogram of Demand + boxplot by Category |
| `chart_region_promo.png` | Average Demand by Region + Promotion impact |
| `chart_heatmap.png` | Correlation heatmap for all numeric columns |

Uses `matplotlib` (Agg backend for headless/script execution) and `seaborn`.

---

### `feature_engineering.py`
Creates new columns from existing data to improve model signal.

**Date features** (extracted from `Date`):

| Feature | Description |
|---|---|
| `Year` | Calendar year |
| `Month` | Month number (1–12) |
| `Day` | Day of month |
| `DayOfWeek` | 0 = Monday, 6 = Sunday |
| `Quarter` | Quarter (1–4) |
| `WeekOfYear` | ISO week number |

**Business ratio features:**

| Feature | Formula |
|---|---|
| `Price_Diff` | `Price − Competitor Pricing` |
| `Effective_Price` | `Price × (1 − Discount / 100)` |
| `Inventory_Ratio` | `Inventory Level / (Units Sold + 1)` |

> The `+1` in `Inventory_Ratio` prevents division-by-zero errors.

---

### `categorical_feature_encoding.py`
Encodes all text/categorical columns into numeric values using `sklearn.preprocessing.LabelEncoder`.

- Automatically detects columns of type `object` or `category`
- Applies `LabelEncoder` to each, using `.astype(str)` for safety
- Also re-engineers date and ratio features on the encoded copy (`df_encoded`)

**Encoded columns include:** `Store ID`, `Product ID`, `Category`, `Region`, `Weather Condition`, `Seasonality`

---

### `input_setup.py`
Defines the final feature set and splits the data for training.

**23 features used:**

```
Store ID, Product ID, Category, Region,
Inventory Level, Units Sold, Units Ordered,
Price, Discount, Weather Condition, Promotion,
Competitor Pricing, Seasonality, Epidemic,
Year, Month, Day, DayOfWeek, Quarter,
WeekOfYear, Price_Diff, Effective_Price, Inventory_Ratio
```

**Target variable:** `Demand`

**Train/test split:** 80% training / 20% testing (`random_state=42`)

Also defines the reusable `evaluate_model()` helper function used by all model scripts, which returns MAE, RMSE, and R².

---

## Dataset Columns

| Column | Type | Description |
|---|---|---|
| `Date` | datetime | Transaction date |
| `Store ID` | categorical | Store identifier |
| `Product ID` | categorical | Product identifier |
| `Category` | categorical | Product category |
| `Region` | categorical | Geographic region |
| `Weather Condition` | categorical | Weather on that day |
| `Seasonality` | categorical | Season label |
| `Inventory Level` | numeric | Stock on hand |
| `Units Sold` | numeric | Units sold |
| `Units Ordered` | numeric | Units ordered |
| `Price` | numeric | Product price |
| `Discount` | numeric | Discount applied |
| `Competitor Pricing` | numeric | Competitor's price |
| `Promotion` | binary | Whether promoted |
| `Epidemic` | binary | Epidemic indicator |
| `Demand` | numeric | **Target variable** |

---

## Dependencies

```
pandas
scikit-learn
matplotlib
seaborn
```

---

## Known Issues / Suggestions

- The CSV path in `data_loading.py` is hardcoded for Windows. Use a relative path or environment variable for portability.
- `data_summary.py` and `EDA.py` are duplicates — consider merging them.
- Feature engineering happens in both `feature_engineering.py` (on raw `df`) and `categorical_feature_encoding.py` (on `df_encoded`). The version in `categorical_feature_encoding.py` is the one actually used downstream.
# Part 2 — Model Training

This section covers all machine learning models trained to predict retail demand, from a simple linear baseline to ensemble methods.

---

## Folder Structure

```
model/
├── linear_regression.py
├── LSTM.py
├── LightGBM.py
├── Gradient_Boosting.py
├── XGBoost.py
├── histogram_gradient_boosting.py
├── ensemble_1.py
└── ensemble_2.py
```

---

## Models Overview

| # | Model | File | Type |
|---|---|---|---|
| 1 | Linear Regression | `linear_regression.py` | Baseline |
| 2 | LSTM | `LSTM.py` | Deep Learning |
| 3 | LightGBM | `LightGBM.py` | Gradient Boosting |
| 4 | Gradient Boosting | `Gradient_Boosting.py` | Gradient Boosting |
| 5 | XGBoost | `XGBoost.py` | Gradient Boosting |
| 6 | Histogram Gradient Boosting | `histogram_gradient_boosting.py` | Gradient Boosting |
| 7 | Ensemble 1: XGBoost + HGB | `ensemble_1.py` | Voting Ensemble |
| 8 | Ensemble 2: GB + HGB | `ensemble_2.py` | Voting Ensemble |

All models are evaluated on three metrics:

| Metric | Description | Goal |
|---|---|---|
| **MAE** | Mean Absolute Error — average error in demand units | Lower is better |
| **RMSE** | Root Mean Squared Error — penalizes large errors more | Lower is better |
| **R²** | R-squared — proportion of variance explained | Higher is better (max 1.0) |

---

## Model Details

### Model 1 — Linear Regression (`linear_regression.py`)
A straight-line baseline model with no hyperparameters.

- Used to establish a minimum performance floor
- Uses the shared `evaluate_model()` helper from `input_setup.py`

```python
LinearRegression()
```

---

### Model 2 — LSTM (`LSTM.py`)
A two-layer Long Short-Term Memory neural network built with TensorFlow/Keras.

- Input data reshaped to 3D `(samples, timesteps, 1)` for LSTM compatibility
- Data cast to `float32` to avoid TensorFlow type errors

**Architecture:**

```
Input → LSTM(64, return_sequences=True) → LSTM(64) → Dense(1)
```

**Training config:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | MSE |
| Epochs | 20 |
| Batch size | 32 |
| Validation | 20% test set |

---

### Model 3 — LightGBM (`LightGBM.py`)
Microsoft's gradient boosting framework, optimized for speed on large datasets.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 |
| `learning_rate` | 0.5 |
| `max_depth` | 6 |
| `num_leaves` | 31 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

---

### Model 4 — Gradient Boosting (`Gradient_Boosting.py`)
Scikit-learn's classic gradient boosting regressor.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 |
| `learning_rate` | 0.5 |
| `max_depth` | 6 |

---

### Model 5 — XGBoost (`XGBoost.py`)
Extreme Gradient Boosting, a high-performance tree ensemble.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 |
| `learning_rate` | 0.5 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `objective` | `reg:squarederror` |

---

### Model 6 — Histogram Gradient Boosting (`histogram_gradient_boosting.py`)
Scikit-learn's `HistGradientBoostingRegressor` — faster than standard GB for large datasets by bucketing features into histograms.

| Hyperparameter | Value |
|---|---|
| `loss` | `squared_error` |
| `learning_rate` | 0.5 |
| `max_iter` | 300 |
| `max_depth` | 6 |
| `min_samples_leaf` | 20 |
| `l2_regularization` | 0.1 |

---

### Model 7 — Ensemble 1: XGBoost + HGB (`ensemble_1.py`)
A `VotingRegressor` that averages predictions from XGBoost and Histogram Gradient Boosting.

```python
VotingRegressor([
    ('xgb', xgb_model),
    ('HGB', HGB_model)
])
```

Both component models are imported from their respective files (already trained).

---

### Model 8 — Ensemble 2: GB + HGB (`ensemble_2.py`)
A `VotingRegressor` that averages predictions from Gradient Boosting and Histogram Gradient Boosting.

```python
VotingRegressor([
    ('gb', gb_model),
    ('HGB', HGB_model)
])
```

> This ensemble is later used as the base model for fine-tuning in Part 3.

---

## Shared Design Patterns

All model scripts follow the same 2-step structure:

```python
# Step 1 — Fix import path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Step 2 — Import project modules
from input_processing.input_setup import X_train, X_test, y_train, y_test
```

Models that require 2D input (LightGBM, XGBoost, HGB) include a reshape guard:

```python
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
```

Each model stores its results in a dictionary:

```python
results = {
    'Model': 'Model Name',
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2,
    'Predictions': y_pred
}
```

---

## Dependencies

```
scikit-learn
lightgbm
xgboost
tensorflow
numpy
```
# Part 3 — Output Interpretation

This section covers model comparison, feature importance analysis, and hyperparameter fine-tuning of the best-performing ensemble.

---

## Folder Structure

```
output_interpretation/
├── model_comparison.py
├── feature_importance.py
└── fine_tuning.py
```

---

## Pipeline Overview

```
model_comparison.py      ← rank all 8 models, export results, generate prediction charts
        ↓
feature_importance.py    ← run permutation importance on the best model
        ↓
fine_tuning.py           ← tune Ensemble 2 (GB + HGB) with RandomizedSearchCV
```

---

## File Descriptions

### `model_comparison.py`
Collects results from all 8 trained models and produces a ranked comparison table.

**What it does:**

- Builds a `results_df` DataFrame with MAE, RMSE, and R² for all models
- Ranks models by R² (descending)
- Prints a formatted table using `tabulate` (fancy_grid style)
- Exports results to `model/model_performance_comparison.csv`
- Automatically identifies the best model by R²
- Generates 4 output charts (see below)

**Output charts:**

| File | Description |
|---|---|
| `chart_prediction.png` | Scatter plot (Actual vs Predicted) + error distribution for best model |
| `actual_predicted_all_models.png` | 4×2 grid of scatter plots for all 8 models |
| `model_comparison_150_200.png` | Line chart — Actual vs all model predictions for index range 150–200 |

**Error statistics printed for the best model:**
- Mean error (in demand units)
- Standard deviation of errors
- % of predictions within ±20 units

---

### `feature_importance.py`
Uses permutation importance to identify which features matter most to the best model.

**How it works:**

1. Imports all trained model objects and the ranked `results_df` from `model_comparison.py`
2. Automatically selects the best model (rank 1 by R²)
3. Handles LSTM separately — uses its 3D `X_test_lstm` array; all other models use standard 2D `X_test`
4. Runs `sklearn.inspection.permutation_importance` with 5 repeats (`scoring='r2'`)
5. Plots the top 15 most important features as a horizontal bar chart

**Output chart:**

| File | Description |
|---|---|
| `chart_permutation_importance.png` | Top 15 features ranked by permutation importance score |

**Console output:** Top 10 features with their importance scores, ranked and labeled.

---

### `fine_tuning.py`
Tunes Ensemble 2 (Gradient Boosting + Histogram Gradient Boosting) using `RandomizedSearchCV`.

**Search space:**

| Component | Hyperparameters Tuned |
|---|---|
| Gradient Boosting | `n_estimators` [100, 200, 300], `max_depth` [4, 6, 8], `learning_rate` [0.1, 0.2, 0.5] |
| HGB | `max_iter` [100, 200, 300], `max_depth` [4, 6, 8], `learning_rate` [0.1, 0.2, 0.5] |
| Ensemble | `weights` — 9 combinations from [1,1] to [4,3] |

**Search configuration:**

| Parameter | Value |
|---|---|
| `n_iter` | 30 random combinations |
| `cv` | 8-fold cross-validation |
| `scoring` | `neg_mean_absolute_error` |
| `n_jobs` | -1 (all CPU cores) |

**Comparison output:**

After tuning, prints a side-by-side table comparing the base Ensemble 2 vs. the tuned model:

| Metric | Ensemble 2 (Base) | Tuned Hybrid | Improvement (%) |
|---|---|---|---|
| MAE | ... | ... | ... |
| RMSE | ... | ... | ... |
| R² | ... | ... | ... |

**Output chart:**

| File | Description |
|---|---|
| `actual_predicted_tuned_original.png` | Line chart — Actual vs Base vs Tuned predictions (index 150–200) |

---

## Output Files Summary

| File | Generated By | Description |
|---|---|---|
| `model_performance_comparison.csv` | `model_comparison.py` | All model metrics, ranked by R² |
| `chart_prediction.png` | `model_comparison.py` | Best model scatter + error distribution |
| `actual_predicted_all_models.png` | `model_comparison.py` | All 8 models scatter grid |
| `model_comparison_150_200.png` | `model_comparison.py` | Line chart comparison (index 150–200) |
| `chart_permutation_importance.png` | `feature_importance.py` | Top 15 feature importance bar chart |
| `actual_predicted_tuned_original.png` | `fine_tuning.py` | Tuned vs base ensemble line chart |

---

## Dependencies

```
scikit-learn
numpy
pandas
matplotlib
tabulate
```

---

## Notes

- `model_comparison.py` imports `results_df` from itself (`from model_comparison import results_df`) — this is a circular import and may cause issues. Consider extracting `results_df` into a separate `results_store.py` module.
- The fine-tuning targets Ensemble 2 (GB + HGB) regardless of which model ranked best. If a different model wins, consider making the tuning target dynamic.
- The line charts use a fixed index window (150–200) for a quick visual sanity check. This range can be adjusted to inspect different segments of the test set.

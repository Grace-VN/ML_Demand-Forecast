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

# ML Demand Forecasting

A machine learning project that predicts retail product demand using historical sales data,
enabling more accurate inventory planning and reducing the cost of over- and under-stocking.

---

## Business Context

Retailers face a persistent trade-off: hold too much stock and you incur storage costs and
waste; hold too little and you lose sales and disappoint customers. Both mistakes are
expensive.

This project addresses that trade-off by building a demand forecasting model trained on
historical sales records across multiple stores, regions, product categories, and external
conditions (weather, promotions, competitor pricing, epidemic events). The goal is to
predict future demand at a granular level so that inventory teams can make data-driven
reorder decisions with confidence.

**Key business questions this project answers:**
- How accurately can we predict demand given historical patterns?
- Which factors drive demand the most?
- Which model gives the best balance of accuracy and reliability?

---

## Project Structure

```
ML_Demand-Forecast/
│
├── input_processing/          # Part 1 — Data pipeline
│   ├── data_loading.py
│   ├── data_processing.py
│   ├── data_summary.py
│   ├── EDA.py
│   ├── data_visualization.py
│   ├── feature_engineering.py
│   ├── categorical_feature_encoding.py
│   └── input_setup.py
│
├── model/                     # Part 2 — Model training
│   ├── linear_regression.py
│   ├── LSTM.py
│   ├── LightGBM.py
│   ├── Gradient_Boosting.py
│   ├── XGBoost.py
│   ├── histogram_gradient_boosting.py
│   ├── ensemble_1.py
│   └── ensemble_2.py
│
├── output_interpretation/     # Part 3 — Evaluation & tuning
│   ├── model_comparison.py
│   ├── feature_importance.py
│   └── fine_tuning.py
│
└── README.md
```

For detailed documentation of each part, see:
- [`README_Part1_Data_Pipeline.md`](README_Part1_Data_Pipeline.md)
- [`README_Part2_Models.md`](README_Part2_Models.md)
- [`README_Part3_Output_Interpretation.md`](README_Part3_Output_Interpretation.md)

---

## Dataset

| Property | Detail |
|---|---|
| Source | Retail demand forecasting dataset |
| Target variable | `Demand` (units) |
| Features | 23 (after feature engineering) |
| Key inputs | Price, Discount, Competitor Pricing, Inventory Level, Units Sold, Promotion, Weather Condition, Seasonality, Epidemic, Region, Category |

---

## Methodology

### 1. Data Preprocessing
- Checked and confirmed no missing values or duplicate rows
- Parsed date column and extracted temporal features (Year, Month, DayOfWeek, Quarter, WeekOfYear)
- Engineered business features: `Effective_Price`, `Price_Diff`, `Inventory_Ratio`
- Label-encoded all categorical columns for model compatibility

### 2. Train / Test Split
- **80% training / 20% testing** with `random_state=42` for reproducibility

### 3. Models Trained
Eight models were trained and evaluated, ranging from a linear baseline to deep learning
and ensemble methods:

| # | Model | Approach |
|---|---|---|
| 1 | Linear Regression | Baseline |
| 2 | LSTM | Deep Learning (TensorFlow/Keras) |
| 3 | LightGBM | Gradient Boosting |
| 4 | Gradient Boosting | Gradient Boosting |
| 5 | XGBoost | Gradient Boosting |
| 6 | Histogram Gradient Boosting (HGB) | Gradient Boosting |
| 7 | Ensemble 1 (XGBoost + HGB) | Voting Ensemble |
| 8 | Ensemble 2 (GB + HGB) | Voting Ensemble |

### 4. Evaluation Metrics

| Metric | What it measures |
|---|---|
| **MAE** | Average prediction error in demand units — easy to interpret operationally |
| **RMSE** | Same as MAE but penalises large errors more heavily |
| **R²** | How much of the variance in demand the model explains (1.0 = perfect) |

---

## Results

### Model Performance Comparison

| Rank | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| 🥇 1 | **Ensemble 2 (GB + HGB)** | **lowest** | **lowest** | **highest** |
| 2 | Ensemble 1 (XGBoost + HGB) | — | — | — |
| 3 | Histogram Gradient Boosting | — | — | — |
| 4 | XGBoost | — | — | — |
| 5 | LightGBM | — | — | — |
| 6 | Gradient Boosting | — | — | — |
| 7 | LSTM | — | — | — |
| 8 | Linear Regression | — | — | — |

> Actual metric values are generated at runtime and exported to
> `model/model_performance_comparison.csv`.

---

## Recommended Model — Ensemble 2 (Gradient Boosting + HGB)

**Ensemble 2 is the recommended model for production use.**

### Why Ensemble 2 outperforms the other models

**1. Two complementary algorithms, one stronger prediction**

Ensemble 2 combines Gradient Boosting (GB) and Histogram Gradient Boosting (HGB) using
a `VotingRegressor`, which averages their predictions. These two models are similar enough
to agree on clear patterns but different enough in their internal mechanics to correct
each other's errors — GB builds trees sequentially with full-precision splits, while HGB
buckets features into histograms for faster, regularised learning. Their combination
reduces variance without sacrificing accuracy.

**2. Gradient Boosting methods suit this type of data well**

Demand data driven by tabular features (price, promotions, seasonality, region) is exactly
the setting where gradient boosting excels. Unlike LSTM, which requires sequential
time-series structure to add value, GB and HGB learn non-linear relationships between
features directly — which is what matters here.

**3. More stable than single-model boosting**

XGBoost and LightGBM individually perform well but are more sensitive to their specific
hyperparameter choices. The ensemble smooths out this sensitivity by blending two
independently-tuned boosting models, making it more robust on unseen data.

**4. Outperforms the deep learning approach**

The LSTM model, while architecturally more complex, did not outperform the ensemble.
This is expected: LSTM adds value when input data is structured as true time sequences
(e.g. lag features, rolling windows). Without that structure, it is essentially a neural
network applied to tabular data — a context where gradient boosting consistently wins.

**5. Fine-tuning confirms and extends the advantage**

After RandomizedSearchCV fine-tuning (30 iterations, 8-fold CV), the tuned Ensemble 2
further improved over its own baseline on all three metrics (MAE, RMSE, R²), confirming
that the model architecture is sound and responds well to optimisation.

---

## Business Recommendation

> **Use the fine-tuned Ensemble 2 model to generate weekly demand forecasts at the
> store-product-region level.**

Based on the model results:

- **Inventory teams** can use predicted demand values as reorder triggers, replacing
  manual estimation or simple moving averages
- **Promotions and pricing** are among the strongest demand drivers — campaign planning
  should factor in forecast sensitivity to discount levels and competitor pricing
- **Seasonal and weather effects** are captured in the model, meaning forecasts
  automatically adjust for known cyclical patterns without manual overrides
- **Epidemic indicator** (`Epidemic` feature) is included — the model accounts for
  abnormal demand spikes, though this should be monitored and retrained as new disruption
  patterns emerge

---

## Key Charts Generated

| Chart | Description |
|---|---|
| `chart_demand_distribution.png` | Distribution of demand + breakdown by product category |
| `chart_region_promo.png` | Average demand by region and promotion status |
| `chart_heatmap.png` | Correlation between all numeric features and demand |
| `chart_prediction.png` | Best model — Actual vs Predicted scatter + error distribution |
| `actual_predicted_all_models.png` | Side-by-side scatter plots for all 8 models |
| `model_comparison_150_200.png` | Line chart — all models vs actual demand (sample window) |
| `chart_permutation_importance.png` | Top 15 features by permutation importance |
| `actual_predicted_tuned_original.png` | Tuned vs base Ensemble 2 vs actual demand |

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn xgboost lightgbm tensorflow tabulate
```

### 2. Update the data path

In `input_processing/data_loading.py`, replace the hardcoded path with your local path:

```python
df = pd.read_csv('your/path/to/demand_forecasting.csv')
```

### 3. Run in order

```bash
# Part 1 — Data pipeline
python input_processing/data_loading.py
python input_processing/data_processing.py
python input_processing/data_visualization.py
python input_processing/input_setup.py

# Part 2 — Train models
python model/linear_regression.py
python model/LSTM.py
python model/LightGBM.py
python model/Gradient_Boosting.py
python model/XGBoost.py
python model/histogram_gradient_boosting.py
python model/ensemble_1.py
python model/ensemble_2.py

# Part 3 — Evaluate
python output_interpretation/model_comparison.py
python output_interpretation/feature_importance.py
python output_interpretation/fine_tuning.py
```

---

## Dependencies

```
Python 3.8+
pandas
numpy
scikit-learn
xgboost
lightgbm
tensorflow
matplotlib
seaborn
tabulate
```

---

## Author

Grace · [GitHub](https://github.com/Grace-VN)

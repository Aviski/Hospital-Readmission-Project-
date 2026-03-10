# Hospital Readmission Risk Predictor

A production-quality, end-to-end machine learning pipeline for predicting **30-day hospital readmission risk** at discharge, built on public synthetic clinical data.

---

## Project Overview

Hospital readmissions within 30 days of discharge are costly, often preventable, and strongly associated with worse patient outcomes. A well-calibrated risk score at discharge allows care-management teams to prioritise follow-up resources — phone calls, home visits, medication reconciliation — for the patients most likely to return.

This project demonstrates a complete ML pipeline in a healthcare context:

| Stage | Description |
|---|---|
| **EDA** | Understand distributions, missingness, and class imbalance |
| **Cleaning** | Imputation, duplicate removal, validation, encoding |
| **Feature Engineering** | Comorbidity burden, age groups, utilisation features |
| **Modelling** | Logistic Regression, Random Forest, Gradient Boosting (XGBoost/LightGBM) |
| **Calibration** | CalibratedClassifierCV (isotonic / sigmoid) for reliable probabilities |
| **Interpretability** | SHAP global + local explanations, subgroup error analysis |
| **Reporting** | Full written report + figures |

---

## Repository Structure

```
hospital-readmission-risk/
├── config/
│   └── config.yaml                  # Central config: paths, seeds, hyperparameters
├── data/
│   ├── raw/                         # Original CSVs — never modified manually
│   └── processed/                   # Cleaned and feature-engineered datasets
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_cleaning_check.ipynb      # Cleaning and feature verification
│   ├── 03_modeling_baseline.ipynb   # Baseline model comparison
│   ├── 04_modeling_tuned.ipynb      # Tuned models, calibration, final results
│   └── 05_interpretation_error_analysis.ipynb  # SHAP + subgroup analysis
├── src/
│   ├── utils.py                     # Shared helpers (logger, config, model I/O, seed)
│   ├── data_preparation.py          # Loading, cleaning, and validation
│   ├── feature_engineering.py       # Healthcare feature engineering
│   └── modeling.py                  # Training, evaluation, calibration
├── models/                          # Serialised model files (.joblib)
├── reports/
│   └── figures/                     # ROC, PR, calibration, SHAP plots
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and set up the environment

```bash
git clone https://github.com/<your-username>/hospital-readmission-risk.git
cd hospital-readmission-risk
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your dataset

Download a hospital readmission dataset (e.g. from Kaggle) and place the CSV at:

```
data/raw/readmission.csv
```

Update `config/config.yaml` to reflect the actual column names — especially `data.target_column` and `features.age_col`.

### 3. Run exploratory data analysis

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Run the full pipeline (once `src/modeling.py` is complete)

```python
from src.utils import load_config, set_seed
from src.data_preparation import load_raw_data, clean_data
from src.feature_engineering import create_features

config = load_config()
set_seed(config['random_seed'])

df_raw = load_raw_data(config['paths']['raw_data'])
df_clean = clean_data(df_raw, config)
df_features = create_features(df_clean, config)
```

---

## Pipeline Stages

### Stage 1 — EDA (`notebooks/01_eda.ipynb`)
- Inspect shape, dtypes, and summary statistics
- Visualise missingness, class balance, feature distributions
- Flag multicollinearity (|r| > 0.85)

### Stage 2 — Data Preparation (`src/data_preparation.py`)
- `load_raw_data(path)` — CSV ingestion with validation
- `clean_data(df, config)` — duplicate removal, median / "Unknown" imputation, optional one-hot encoding

### Stage 3 — Feature Engineering (`src/feature_engineering.py`)
- `create_features(df, config)` — comorbidity count, age groups, prior-admission flag, discharge disposition grouping

### Stage 4 — Modelling (`src/modeling.py`)
- Stratified 60/20/20 train/val/test split
- Baseline: Logistic Regression, Random Forest, XGBoost / LightGBM
- 5-fold StratifiedKFold CV; primary metric: PR-AUC
- Hyperparameter tuning with `RandomizedSearchCV`
- Probability calibration with `CalibratedClassifierCV`
- Persists final model to `models/best_model.joblib`

### Stage 5 — Interpretability (`notebooks/05_interpretation_error_analysis.ipynb`)
- SHAP beeswarm (global importance)
- SHAP waterfall plots for individual predictions (TP / FP / FN)
- Subgroup performance by age, sex, comorbidity burden, and discharge disposition

---

## Configuration

All runtime parameters live in `config/config.yaml`. Key sections:

| Key | Description |
|---|---|
| `paths.*` | File paths for raw data, processed data, and model output |
| `random_seed` | Global seed for reproducibility |
| `data.target_column` | Name of the binary readmission label column |
| `data.drop_columns` | Columns to exclude (IDs, leakage, free-text) |
| `features.*` | Column names and bin definitions for feature engineering |
| `model.*` | Hyperparameter defaults for each model type |

---

## Dataset

> **Add details here once a dataset is selected.**

- **Name:** _e.g. Diabetes 130-US Hospitals Dataset_
- **Source:** _e.g. Kaggle / UCI_
- **Rows / Columns:** _e.g. 101,766 × 50_
- **Positive class rate:** _e.g. 11.2%_

---

## Requirements

See `requirements.txt`. Core dependencies:

```
pandas
numpy
scikit-learn
xgboost
lightgbm
imbalanced-learn
shap
matplotlib
seaborn
pyyaml
joblib
jupyter
```

---

## License

MIT — see `LICENSE`.

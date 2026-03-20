# Hospital Readmission Predictor

An end-to-end machine learning pipeline for predicting hospital readmission risk at discharge,
built on the checked-in `hospital_readmissions.csv` dataset (25,000 diabetic patient records).

---

## Overview

Hospital readmissions are costly and often associated with worse patient outcomes. A calibrated
risk score at discharge could help care-management teams prioritise follow-up resources — phone
calls, home visits, medication reconciliation — for the patients most likely to return.

This project demonstrates a complete ML pipeline in a healthcare context: data cleaning and
validation, feature engineering, multi-model comparison, hyperparameter tuning, post-hoc
probability calibration, recall-constrained threshold selection, SHAP-based interpretation, and
error group profiling. The pipeline is fully reproducible from a single CLI command.

| Stage | Description |
|---|---|
| EDA | Distributions, class balance, utilisation patterns, diagnosis breakdown |
| Cleaning | Target mapping, duplicate removal, schema validation, median imputation |
| Feature Engineering | Age ordinal, utilisation flags, specialty known flag, interaction terms |
| Modelling | LR, RF, HGBC baselines; RandomizedSearchCV tuning; post-hoc calibration; recall-constrained threshold |
| Interpretability | SHAP global + local explanations on the full calibrated model |
| Error Analysis | TP/TN/FP/FN subgroup profiling |

---

## Results Summary

Best model: **Calibrated Gradient Boosting (sigmoid)** at decision threshold **0.35**.

Threshold selected on the validation set as the highest-precision point where recall ≥ 0.80 —
a project design choice reflecting the asymmetric cost of missing a high-risk patient in a
discharge-prioritisation context. This is not a validated clinical deployment policy.

| Metric | Validation | Test |
|---|---|---|
| ROC-AUC | 0.658 | 0.653 |
| Recall | 0.877 | 0.879 |
| Precision | 0.512 | 0.512 |
| F1 | 0.646 | 0.647 |
| Threshold | 0.35 (selected on val) | — |

Validation and test metrics are similar, which is consistent with stable generalisation.
However, validation metrics at the chosen threshold are somewhat optimistic because threshold
selection used the validation set; the test set is the cleaner measure.

---

## Key Findings

- **`total_prior_utilization`** (sum of all prior outpatient, inpatient, and emergency visits) is
  the strongest predictor by SHAP importance — prior healthcare engagement is the clearest
  readmission signal in this dataset.

- **The interaction term `n_inpatient × time_in_hospital`** is the second-ranked feature.
  Patients with both a history of prior inpatient admissions and a long current stay have the
  highest risk signal.

- **Diabetes medication (`diabetes_med_yes`) and age ordinal** are also in the top four SHAP
  features.

- **Primary failure mode — low-history patients:** Patients with near-zero prior utilisation
  look low-risk to the model despite genuine readmission. These false negatives are the hardest
  to catch with the available features.

- **False positive rate is intentionally high:** At threshold 0.35, approximately 39% of the
  validation set is flagged as high-risk. This is consistent with the recall constraint — the
  model errs toward false alarms rather than missed patients.

---

## Design Decisions

**Recall-constrained threshold**
Threshold swept from 0.05–0.95 in steps of 0.05. The highest-precision threshold where
recall ≥ 0.80 is selected. There is no readmission cost data in this dataset; the 0.80 recall
floor is a project-level assumption about clinical priority, not a validated policy.

**Post-hoc probability calibration**
Gradient Boosting probability outputs are not guaranteed to be well-calibrated. Sigmoid and
isotonic calibration were compared on a held-out 30% of the validation set; sigmoid was
marginally better (Brier 0.2279 vs 0.2287). The uncalibrated Brier was 0.2303, so the
improvement is small (~0.0007). Calibration is retained for probability correctness, not
AUC gain.

**sklearn HistGradientBoostingClassifier**
Chosen to avoid external dependencies (no XGBoost or LightGBM). HGBC handles the dataset
scale comfortably and is part of the standard sklearn distribution.

**OHE with schema alignment**
`pd.get_dummies` with `drop_first=True`. A post-encoding reindex step aligns data to the
training schema: missing dummies are filled as 0, and unexpected categorical values are
detected and logged with a WARNING; encoding then proceeds via schema alignment, which maps
unseen categories to the reference level. This approach is appropriate for reproducible batch
reruns on this dataset, but it is not a production-grade categorical handling system for
arbitrary new inputs.

**`_PrefitCalibratedModel`**
A custom calibration wrapper replacing `CalibratedClassifierCV(cv="prefit")`, which was
removed in sklearn 1.6+. Non-standard but self-contained and confined to `src/modeling.py`.

---

## Limitations

1. **Threshold optimism on validation** — Threshold selection used the validation set. Reported
   validation metrics at threshold 0.35 are somewhat optimistic. Test set metrics are the
   cleaner measure.

2. **Single dataset** — All 25,000 records come from one source. Generalisability to other
   institutions, patient populations, or time periods is unverified.

3. **No provenance documentation** — The dataset is a checked-in CSV. Detailed source
   provenance is not documented in this repo.

4. **Schema alignment is not production-grade** — Unseen categorical values at inference time
   are mapped to the reference level (with a WARNING logged to the console). A fitted sklearn
   `OneHotEncoder` would be more robust for production inference on arbitrary inputs.

5. **No temporal validation** — Records are not ordered chronologically. If the dataset has
   temporal structure, temporal leakage cannot be ruled out.

6. **Low-history false negatives** — The model systematically misses patients with little or no
   prior utilisation history. This is a feature-availability limitation, not a bug.

---

## Reproducible Pipeline Run

The canonical way to regenerate all outputs from scratch:

```bash
# Full run — features, models, metrics, figures, SHAP, error analysis
.venv\Scripts\python.exe -m src.rerun_report_pipeline

# Fast run — skips SHAP and error-analysis figures
.venv\Scripts\python.exe -m src.rerun_report_pipeline --skip-interpretation
```

This regenerates:
- `data/processed/readmission_features.csv` and `readmission_features_raw.csv`
- `data/processed/features_metadata.json`
- `data/processed/metrics_summary.csv` and `metrics_tuned_summary.csv`
- `models/best_baseline_model.pkl` and `models/best_tuned_model.pkl`
- All figures under `reports/figures/`

**Note:** `reports/final_report.md` is a static narrative document and is not rewritten by the
CLI. If you skip interpretation (`--skip-interpretation`), SHAP and error-analysis figures from
the previous full run remain in place.

The notebooks in `notebooks/` remain useful for exploration and step-by-step narrative, but the
CLI is the primary reproducibility path.

Warnings during execution (e.g., unseen categories or model convergence) are surfaced explicitly in the console.

Validate preprocessing only (faster check):

```bash
.venv\Scripts\python.exe -m src.cleaning_validation
```

---

## Repository Structure

```
hospital-readmission-risk/
├── config/
│   └── config.yaml                  # Central config: paths, seeds, hyperparameters
├── data/
│   ├── raw/                         # Original CSV — never modified manually
│   │   └── hospital_readmissions.csv
│   └── processed/                   # Pipeline outputs (generated — safe to delete and rebuild)
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_modeling_baseline.ipynb   # Baseline model comparison
│   ├── 03_model_tuning.ipynb        # Tuning, calibration, threshold selection
│   └── 04_model_interpretation_and_error_analysis.ipynb
├── src/
│   ├── utils.py                     # Shared helpers (logger, config, model I/O, seed)
│   ├── data_preparation.py          # Loading, cleaning, validation, encoding
│   ├── feature_engineering.py       # Healthcare feature engineering
│   ├── modeling.py                  # Training, evaluation, calibration, threshold sweep
│   ├── interpretation.py            # SHAP explanations and error analysis
│   ├── pipeline.py                  # Orchestrates load → clean → engineer → encode
│   ├── rerun_report_pipeline.py     # CLI entry point for full pipeline rerun
│   └── cleaning_validation.py       # Preprocessing validation script
├── models/                          # Serialised model files (.pkl) — generated
├── reports/
│   ├── figures/                     # All plots — generated
│   └── final_report.md              # Static narrative report
├── requirements.txt
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| **File** | `data/raw/hospital_readmissions.csv` |
| **Rows** | 25,000 |
| **Columns** | 17 |
| **Target** | `readmitted` — `yes` → 1, `no` → 0 |
| **Positive rate** | 47.0% (11,754 / 25,000) — near-balanced |
| **Source** | Diabetic patient records; detailed provenance is not documented in this repo |

### Feature Schema

| Column | Type | Notes |
|---|---|---|
| `age` | Categorical | Bracketed ranges: `[40-50)`, ..., `[90-100)` |
| `time_in_hospital` | Integer | Days admitted (1–14) |
| `n_lab_procedures` | Integer | Lab procedures during stay |
| `n_procedures` | Integer | Non-lab procedures during stay |
| `n_medications` | Integer | Distinct medications prescribed |
| `n_outpatient` | Integer | Prior outpatient visits |
| `n_inpatient` | Integer | Prior inpatient admissions |
| `n_emergency` | Integer | Prior emergency visits |
| `medical_specialty` | Categorical | 49.5% of values are `"Missing"` (valid category, not NaN) |
| `diag_1` / `diag_2` / `diag_3` | Categorical | Diagnosis codes (8 categories each) |
| `glucose_test` | Categorical | `no` / `normal` / `high` |
| `A1Ctest` | Categorical | `no` / `normal` / `high` |
| `change` | Categorical | `yes` / `no` — medication change made |
| `diabetes_med` | Categorical | `yes` / `no` — diabetes medication prescribed |

---

## Quickstart

### 1. Set up the environment

```bash
cd hospital-readmission-risk
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify the dataset is in place

The raw CSV is checked into the repo:

```
data/raw/hospital_readmissions.csv
```

### 3. Regenerate all outputs

```bash
.venv\Scripts\python.exe -m src.rerun_report_pipeline
```

This is the canonical rerun path. It rebuilds all processed data, models, metrics, and figures
from the raw CSV. Expect a runtime of several minutes (SHAP computation is the slowest step).

To skip SHAP and error analysis for a faster run:

```bash
.venv\Scripts\python.exe -m src.rerun_report_pipeline --skip-interpretation
```

### 4. (Optional) Step through notebooks interactively

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling_baseline.ipynb
jupyter notebook notebooks/03_model_tuning.ipynb
jupyter notebook notebooks/04_model_interpretation_and_error_analysis.ipynb
```

The notebooks are useful for exploration and step-by-step narrative. They must be run in order.
**If you have already run the CLI pipeline**, the notebooks will use the generated artifacts
and do not need to be re-executed unless you want the interactive cell-by-cell experience.

---

## Configuration

All runtime parameters live in `config/config.yaml`.

| Key | Description |
|---|---|
| `paths.*` | File paths for raw data, processed data, model output |
| `random_seed` | Global seed for reproducibility (`42`) |
| `cache_version` | Bump manually when preprocessing logic changes (current: `2`) |
| `data.target_column` | `"readmitted"` |
| `data.target_positive_value` | `"yes"` (mapped to 1) |
| `data.target_negative_value` | `"no"` (mapped to 0) |
| `data.required_columns` | Schema declaration — cleaning raises on missing columns |
| `features.age_brackets` | Bracket string → ordinal integer mapping |
| `features.prior_utilization_cols` | Columns for binary utilisation flags |
| `model.cv_folds` | CV folds for baseline cross-validation (keep in sync with `tuning.cv_folds`) |
| `model.threshold.recall_target` | Recall floor for threshold selection (`0.80`) |

No hardcoded hyperparameters exist outside `config.yaml`.

---

## Requirements

See `requirements.txt`. Core dependencies:

```
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
shap >= 0.44
matplotlib >= 3.7
seaborn >= 0.13
pyyaml >= 6.0
joblib >= 1.3
jupyter >= 1.0
```

---

## License / Repository Notes

No `LICENSE` file is currently included in this checked-in repo state.

Generated outputs (`data/processed/`, `models/`, `reports/figures/`) are excluded from version
control via `.gitignore`. The raw CSV (`data/raw/hospital_readmissions.csv`) is checked in and
is the source of truth for all pipeline runs.

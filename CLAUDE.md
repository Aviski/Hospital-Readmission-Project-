# CLAUDE.md — Hospital Readmission Predictor

This file is for AI assistants working in this repository. Read it before touching any code.

---

## What This Project Is

An end-to-end ML pipeline predicting hospital readmission risk for diabetic patients. The goal is a well-calibrated, interpretable classifier that a care-management team could use at discharge to prioritise follow-up resources. Clinical context matters: false negatives (missed high-risk patients) are more costly than false positives, which is why the threshold is optimised for recall ≥ 0.80.

---

## Repository Layout

```
src/            Core Python library — reusable, importable, no side effects
notebooks/      Sequential Jupyter workflow — calls src/, adds narrative and plots
config/         Single config.yaml — all parameters live here
data/raw/       Original CSV — never modified
data/processed/ Generated artifacts (CSVs, metadata JSON) — safe to delete and rebuild
models/         Serialised model artifacts (.pkl)
reports/        Generated plots — safe to delete and rebuild
```

---

## The src/ vs notebooks/ Contract

`src/` modules contain logic. Notebooks orchestrate and present. The rule is:
- Business logic belongs in `src/`
- Notebooks import from `src/`, call functions, display results
- Never duplicate pipeline logic between a notebook and `src/`

The authoritative pipeline function is `src/pipeline.py::build_feature_artifacts()`. It is the single path for `load_raw_data → clean_data → create_features → encode_features`. Both `notebooks/02_modeling_baseline.ipynb` and `src/cleaning_validation.py` call it. Do not add a third inline copy anywhere.

---

## Artifact Chain (Notebooks Run in Order)

```
01_eda.ipynb
  └─ Read-only. No artifacts written.

02_modeling_baseline.ipynb
  ├─ Calls build_feature_artifacts() once
  ├─ Writes data/processed/readmission_features.csv       (encoded, model-ready)
  ├─ Writes data/processed/readmission_features_raw.csv   (pre-OHE, analysis-only)
  ├─ Writes data/processed/features_metadata.json         (cache sidecar)
  └─ Writes models/best_baseline_model.pkl                (bare model object)

03_model_tuning.ipynb
  ├─ Reads readmission_features.csv
  └─ Writes models/best_tuned_model.pkl                   (dict — see below)

04_model_interpretation_and_error_analysis.ipynb
  ├─ Reads readmission_features.csv     (for model inference)
  ├─ Reads readmission_features_raw.csv (for diagnosis/error analysis)
  └─ Reads models/best_tuned_model.pkl
```

### best_tuned_model.pkl format — DO NOT CHANGE

Notebook 03 saves a dict, not a bare model. Notebook 04 expects this exact shape:
```python
{
    "model":            <fitted _PrefitCalibratedModel>,
    "threshold":        <float, optimal decision threshold>,
    "model_name":       <str>,
    "val_metrics":      <dict[str, float]>,
    "test_metrics":     <dict[str, float]>,
    "feature_columns":  <list[str]>,
}
```
Never flatten this to a bare model. Never rename the keys.

### best_baseline_model.pkl format — DO NOT CHANGE

Notebook 02 saves the bare fitted model (not a dict). This is intentional — baseline models do not carry a threshold.

---

## Row ID Alignment — Critical

Both processed CSVs share a `row_id` index set immediately after `clean_data()`. This guarantees row order is identical between them regardless of how many duplicates are dropped during cleaning.

When notebook 04 creates the aligned pre-OHE frame, it does:
```python
X_val_raw = df_analysis.loc[X_val.index]
```
This only works because `load_features()` restores `row_id` as the DataFrame index via `index_col="row_id"`. If you ever touch the CSV write or load logic, preserve this.

---

## Feature Artifact Cache

Notebook 02 skips rebuilding the CSVs if the cache is still valid. The cache is validated against a sidecar JSON (`features_metadata.json`) containing:
- SHA-256 hash of feature-relevant config sections (`features.*`, `data.categorical_columns`, etc.)
- Raw CSV file size and mtime
- `cache_version` (top-level key in config.yaml)
- Persisted encoded `feature_columns` for schema-stable scoring

**If you change preprocessing logic** (not just config), manually bump `cache_version` in `config.yaml`. The cache will not auto-invalidate on code changes — only on config or data changes.

---

## Config Design Rules

- All parameters live in `config/config.yaml`. No hardcoded hyperparameters in src/.
- `model.cv_folds` and `model.tuning.cv_folds` are separate keys used by different functions. They must stay in sync — a comment in config.yaml says so. If you change one, change both.
- `model.threshold.recall_target` is the active recall constraint used by `threshold_sweep()`. There is no other recall_target key in the file.
- `cache_version` is a top-level key. Bump it manually when preprocessing logic changes.
- `model.exclude_columns` and `model.drop_redundant_cols` are intentionally empty. They are supported by `load_features()` and exist for future use. Do not remove them.

---

## Known Design Decisions (Do Not "Fix" These)

**Two-pass OHE in `encode_features()`** — First pass encodes explicitly configured `categorical_columns`. Second pass catches any remaining object columns. Both use `drop_first=True`. This is intentional: it ensures that columns created by feature engineering that happen to be categorical are also encoded, without requiring them to be listed in config.

**`_PrefitCalibratedModel` is non-standard sklearn** — It does not inherit from `BaseEstimator`. It cannot be cloned, pickled through sklearn's joblib pipeline, or used with `check_estimator`. This is accepted. It exists because `CalibratedClassifierCV(cv="prefit")` was removed in sklearn 1.6+. Do not replace it with the sklearn class.

**SHAP on the calibrated model** — SHAP values reflect influence on the calibrated probability output, not the base model's raw score. This is correct: the calibrated probability is what the threshold is applied to at inference time.

**Calibration adds minimal Brier improvement (~0.0007)** — This is known. The calibration step is retained for correctness of probability estimates, not for AUC improvement. Do not remove it.

**HGBC has no `feature_importances_`** — `plot_feature_importance()` silently skips GradientBoosting and logs "No feature importance available". This is expected behaviour from sklearn's HistGradientBoostingClassifier. It is not a bug.

**`medical_specialty = "Missing"` is a valid category** — It is not NaN. It is not imputed. It is one-hot encoded like any other category value. Do not treat it as missing data.

**`analysis` frame excludes the target column** — `readmission_features_raw.csv` does not contain the `readmitted` column. The target is only in the encoded CSV. This is intentional.

---

## Common Commands

```bash
# Validate the full preprocessing pipeline
python -m src.cleaning_validation

# Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling_baseline.ipynb
jupyter notebook notebooks/03_model_tuning.ipynb
jupyter notebook notebooks/04_model_interpretation_and_error_analysis.ipynb

# Force a feature artifact rebuild (delete sidecar)
rm data/processed/features_metadata.json
```

---

## What Not to Touch Without a Good Reason

- `src/utils.py` — stable, no known issues
- `src/pipeline.py` — clean, do not add file I/O to it
- `src/data_preparation.py` — the two-pass OHE is intentional
- The `FeatureArtifacts` dataclass field names
- The `best_tuned_model.pkl` dict key names
- `config/config.yaml` key names (downstream code reads them by exact string)

---

## What Is Safe to Regenerate

Everything under `data/processed/`, `models/`, and `reports/` is generated output. Delete any of it and re-run the relevant notebook to rebuild. The source of truth is always the raw CSV plus config.yaml.

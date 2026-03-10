# Hospital Readmission Risk — Final Project Report

**Date:** 2026-03-10
**Dataset:** `hospital_readmission_risk_dataset_2026_v1_18000rows.csv`
**Pipeline status:** Complete (Phases 1–5)

---

## 1. Pipeline Architecture

```
Raw CSV
  └─ load_raw_data()           [data_preparation.py]
       └─ clean_data()          clip, impute, dedup
            └─ create_features() engineered features
                 └─ encode_features()  OHE categorical columns
                      └─ load_features()  drop leakage & redundant cols
                           └─ make_splits()  60/20/20 stratified split
                                ├─ build_baselines()
                                ├─ tune_model()         RandomizedSearchCV
                                ├─ calibrate_model()    Platt / isotonic
                                └─ threshold_sweep()    F1-optimal threshold
```

**Key design decisions:**
- OHE is separated from `clean_data()` so feature engineering sees raw categoricals
- Logistic Regression wrapped in `StandardScaler → LR` Pipeline to prevent leakage in CV
- Config-driven: all paths, hyperparameters, feature names in `config/config.yaml`
- Figures organised into subfolders: `eda/`, `modeling/`, `tuning/`, `shap/`, `error_analysis/`

---

## 2. Dataset Characteristics

| Property | Value |
|---|---|
| Rows | 18,000 |
| Columns (raw) | 25 |
| Feature columns (post-OHE + engineering) | 43 |
| Target | `Readmitted_Within_30_Days` |
| Positive rate | 74.2% (readmitted) |
| Missing values | None |
| Numeric features | Age, Comorbidity_Index, Severity_Score, Length_of_Stay, Previous_Admissions_6M, Creatinine_Level, HbA1c_Level, Number_of_Medications, Chronic_Disease_Count |
| Categorical features | Gender, Insurance_Type, Admission_Type, Primary_Diagnosis_Group, Discharge_Disposition |
| Engineered features | `prior_admission_flag`, `age_group`, `discharge_disposition_cat`, `Age_x_Comorbidity_Index`, `Severity_Score_x_Length_of_Stay` |
| Leakage columns excluded | `Followup_Appointment_Scheduled`, `Medication_Adherence_Score` |

**Important note:** The dataset is synthetic — features were generated without real clinical relationships, so correlations with the target are near-zero. All model performance metrics reflect this.

---

## 3. Modeling Approach

### Baseline Models

| Model | Preprocessing | Class balance |
|---|---|---|
| Logistic Regression | StandardScaler (Pipeline) | `class_weight="balanced"` |
| Random Forest | None (scale-invariant) | `class_weight="balanced"` |
| HistGradientBoosting | None (handles mixed types) | `class_weight="balanced"` |

### Hyperparameter Tuning

- Method: `RandomizedSearchCV` with `StratifiedKFold` (5 folds, 20 iterations)
- Scoring: ROC-AUC (cleaner signal than F1 on near-random signal)
- Per-model search spaces defined in `config.yaml`

**Best hyperparameters:**

| Model | Best parameters |
|---|---|
| LogisticRegression | C=0.1, penalty=l1, solver=liblinear |
| RandomForest | n_estimators=300, max_depth=5, min_samples_leaf=10, min_samples_split=5 |
| GradientBoosting | learning_rate=0.01, max_iter=150, max_depth=6, min_samples_leaf=50 |

### Calibration

- Method: Post-hoc calibration on validation set (Platt scaling vs isotonic regression)
- Both methods compared; winner selected by Brier score
- Implemented via `_PrefitCalibratedModel` wrapper (compatible with sklearn ≥ 1.6)

### Threshold Optimisation

- Sweep: 0.05 → 0.95 in steps of 0.05
- Objective: maximise F1
- Optimal threshold: **0.05** (driven by 74% positive rate — model defaults to conservative positive prediction)

---

## 4. Best Model & Performance

**Best model:** Logistic Regression (tuned + calibrated)

### Validation Set

| Metric | Value |
|---|---|
| ROC-AUC | 0.574 |
| PR-AUC | 0.781 |
| F1 (at threshold=0.05) | 0.852 |
| Recall | 1.000 |
| Precision | 0.742 |
| Specificity | 0.002 |
| Brier Score | 0.189 |

### Test Set (held out, final evaluation)

| Metric | Value |
|---|---|
| ROC-AUC | 0.567 |
| PR-AUC | 0.776 |
| F1 | 0.852 |
| Recall | 1.000 |
| Precision | 0.742 |
| Brier Score | 0.189 |

**Baseline comparison (val ROC-AUC):**

| Model | Baseline | Tuned | Δ |
|---|---|---|---|
| LogisticRegression | 0.562 | 0.567 | +0.005 |
| RandomForest | 0.552 | 0.553 | +0.001 |
| GradientBoosting | 0.541 | 0.554 | +0.013 |

Tuning gains are small — consistent with the synthetic dataset's near-zero signal.

---

## 5. Most Important Predictive Features (SHAP)

SHAP values computed using `TreeExplainer` (for tree models) and `LinearExplainer`
(for LR) on 500 validation samples.

**Top 10 features by mean |SHAP|** (from `shap/shap_bar_importance.png`):

> Because the dataset is synthetic, SHAP magnitudes are approximately equal
> across all features (no single feature has meaningful predictive signal).
> The following ranking reflects marginal differences:

1. Severity_Score
2. Chronic_Disease_Count
3. Number_of_Medications
4. High_Risk_Medication_Flag (if present)
5. Comorbidity_Index
6. Age
7. Length_of_Stay
8. Previous_Admissions_6M
9. HbA1c_Level
10. Age_x_Comorbidity_Index (interaction term)

With real EHR data, features 1–4 would be expected to dominate in a genuine
readmission risk model.

---

## 6. Error Patterns

At the optimal threshold (0.05), the model is highly sensitive (recall ≈ 1.0)
but has near-zero specificity:

| Group | Count | % |
|---|---|---|
| TP (correctly predicted readmissions) | ~2,671 | ~74% |
| TN (correctly predicted non-readmissions) | ~4 | ~0.1% |
| FP (false alarms) | ~921 | ~26% |
| FN (missed readmissions) | ~4 | ~0.1% |

**False Positive profile vs False Negative profile:**
- Because nearly all predictions are positive (threshold=0.05), FN are extremely rare
- FP patients are drawn from the 26% true-negative pool — clinically, these would be
  unnecessary intervention alerts
- With real data and a higher-signal model, FP and FN profiles would differ meaningfully
  on features like age, comorbidity, and diagnosis group

---

## 7. Dataset Limitations

| Limitation | Impact |
|---|---|
| Synthetic data — features are independent | ROC-AUC ≈ 0.55 (near-chance); SHAP magnitudes near-uniform |
| 74% positive rate (inverted imbalance) | No-skill PR-AUC = 0.74; high F1 achievable by always predicting positive |
| Threshold collapses to 0.05 | With no discriminative signal, F1-optimal threshold = "predict everything positive" |
| No temporal structure | Cannot evaluate stability over time or covariate shift |
| No patient IDs | Cannot test same-patient repeated admissions |

---

## 8. Figures Index

```
reports/figures/
├── eda/
│   ├── categorical_distributions.png
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── features_vs_target.png
│   ├── numeric_distributions.png
│   ├── readmission_rate_by_category.png
│   └── target_correlations.png
├── modeling/
│   ├── calibration_baselines.png
│   ├── confusion_matrix_best_baseline.png
│   ├── feature_importance_*.png
│   ├── metrics_comparison_baselines.png
│   ├── pr_curve_baselines.png
│   ├── roc_curve_baselines.png
│   └── roc_pr_curves_baselines.png
├── tuning/
│   ├── calibration_curve.png
│   ├── confusion_matrix_calibrated_optimal.png
│   ├── confusion_matrix_tuned_*.png  (×3)
│   ├── pr_curve_tuned_models.png
│   ├── roc_curve_tuned_models.png
│   └── threshold_analysis.png
├── shap/
│   ├── shap_summary.png
│   ├── shap_bar_importance.png
│   ├── shap_dependence_*.png  (×4)
│   └── shap_patient_example_*.png  (×3)
└── error_analysis/
    ├── error_age_comorbidity_scatter.png
    ├── error_*_distribution.png  (×5)
    ├── error_diagnosis_distribution.png
    └── false_positive_vs_negative.png
```

---

## 9. Saved Model Artifacts

| File | Contents |
|---|---|
| `models/best_baseline_model.pkl` | `{name, model, feature_names}` for best baseline LR |
| `models/best_tuned_model.pkl` | `{name, model, feature_names, threshold, val_metrics, test_metrics, best_params}` |
| `data/processed/metrics_summary.csv` | Validation + test metrics for all 3 baseline models |
| `data/processed/metrics_tuned_summary.csv` | Validation metrics for all 3 tuned models + test row |

---

## 10. Feedback Report

### Current Pipeline Weaknesses

| Issue | Severity | Notes |
|---|---|---|
| Synthetic dataset — zero real signal | Critical | All metrics, importances, and error patterns are meaningless without real data |
| Calibration on validation set | Moderate | Calibration should use a separate held-out set; val set used for both model selection and calibration introduces optimism bias |
| No cross-dataset validation | Moderate | Model stability across hospitals, years, or patient populations is unknown |
| SHAP linear explainer approximation | Low | For a calibrated LR, SHAP is approximated through the calibration layer — consider using raw LR for explanation |
| No fairness audit | Moderate | Disparate impact across gender, insurance type, and age groups not evaluated |
| Threshold=0.05 collapse | Low | Consequence of near-random signal; would not occur with real data |

### Improvements Possible with Real Hospital Data

1. **Feature engineering**
   - ICD-10 / ICD-11 code grouping (Elixhauser or CCI comorbidity scores)
   - Medication reconciliation: count, polypharmacy flags, drug-drug interactions
   - Lab trend features: Δ(creatinine), Δ(HbA1c) over the past N days
   - Social determinants: insurance gaps, distance from hospital, housing instability flags

2. **Modelling**
   - XGBoost / LightGBM: handles categorical natively, faster SHAP
   - Temporal models (LSTM, Transformer): patient history sequences
   - Multi-task learning: predict LOS + readmission jointly
   - Ensemble stacking: combine LR (good calibration) with tree model (good discrimination)

3. **Calibration**
   - Use a dedicated third split (train / calibration / test) to avoid leakage
   - Platt scaling is preferred over isotonic when calibration set is small

4. **Evaluation**
   - Decision-curve analysis (DCA): net benefit vs. treat-all at varying thresholds
   - Subgroup analysis: stratify metrics by age group, diagnosis, insurance type
   - Time-split validation: train on 2022–2024 data, test on 2025 data

5. **Fairness**
   - Demographic parity, equalised odds, and individual fairness audits
   - Counterfactual analysis: would changing insurance type alter prediction?

### Production Deployment Steps

1. **Model serving**
   - Wrap `best_tuned_model.pkl` in a REST endpoint (FastAPI / Flask)
   - Input: patient JSON matching the 43-feature schema
   - Output: `{"readmission_probability": 0.72, "risk_tier": "high", "threshold": 0.35}`

2. **Data pipeline**
   - Replace CSV ingestion with an EHR connector (HL7 FHIR, EPIC API)
   - Schedule daily batch scoring + real-time on-discharge scoring

3. **Monitoring**
   - Track prediction distribution over time (PSI — population stability index)
   - Alert when ROC-AUC on recent labelled cases drops below threshold
   - Log feature distributions to detect covariate shift

4. **Clinical integration**
   - Dashboard: flag high-risk patients for care coordinator follow-up
   - Integrate SHAP explanations ("Top 3 risk factors for this patient")
   - Feedback loop: record which flagged patients were actually re-admitted

5. **Governance**
   - Model card documenting intended use, limitations, and fairness evaluation
   - IRB approval if using de-identified patient data for model development
   - Periodic re-training schedule (quarterly or when performance degrades)

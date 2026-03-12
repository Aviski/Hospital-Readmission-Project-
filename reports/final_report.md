# Hospital Readmission Prediction — Project Report

**Dataset:** `hospital_readmissions.csv` | **Date:** 2026-03-11
**Model:** Calibrated Gradient Boosting (sigmoid) | **Threshold:** 0.35 (recall-constrained)

---

## 1. Dataset

| Attribute | Value |
|---|---|
| Source file | `data/raw/hospital_readmissions.csv` |
| Rows | 25,000 |
| Raw columns | 17 |
| Target column | `readmitted` (`yes` → 1, `no` → 0) |
| Positive rate | 47.0% (11,754 / 25,000) — near-balanced |

### Schema

| Column | Type | Description |
|---|---|---|
| `age` | categorical | Age bracket string: `[40-50)` … `[90-100)` |
| `time_in_hospital` | numeric | Days admitted (1–14) |
| `n_lab_procedures` | numeric | Number of lab tests ordered |
| `n_procedures` | numeric | Number of non-lab procedures |
| `n_medications` | numeric | Number of medications administered |
| `n_outpatient` | numeric | Prior outpatient visits |
| `n_inpatient` | numeric | Prior inpatient admissions |
| `n_emergency` | numeric | Prior emergency visits |
| `medical_specialty` | categorical | Admitting specialty (49.5% are literal "Missing") |
| `diag_1` / `diag_2` / `diag_3` | categorical | Primary and secondary diagnosis codes |
| `glucose_test` | categorical | Glucose serum test result: no / normal / high |
| `A1Ctest` | categorical | HbA1c test result: no / normal / high |
| `change` | categorical | Medication change during stay: yes / no |
| `diabetes_med` | categorical | Diabetes medication prescribed: yes / no |
| `readmitted` | binary target | Readmitted within 30 days |

No missing values in the raw dataset. No rows required imputation.

---

## 2. Pipeline

```
load_raw_data → clean_data → create_features → encode_features
```

### Steps

| Step | Function | Key actions |
|---|---|---|
| Load | `load_raw_data` | Read CSV; raise on missing file |
| Clean | `clean_data` | Map target (yes→1/no→0); drop duplicates; validate schema; median/mode imputation |
| Feature engineering | `create_features` | Add 6 derived features (see below); drop raw `age` |
| Encode | `encode_features` | OHE with `drop_first=True` on all categorical columns |

### Engineered Features

| Feature | Description |
|---|---|
| `age_ordinal` | Ordered integer from age bracket (1=40–50 … 6=90–100); replaces raw `age` |
| `any_n_inpatient` | Binary: any prior inpatient admission |
| `any_n_emergency` | Binary: any prior emergency visit |
| `total_prior_utilization` | Sum of outpatient + inpatient + emergency prior visits |
| `specialty_known` | Binary: `medical_specialty` ≠ "Missing" |
| `n_inpatient_x_time_in_hospital` | Interaction: prior inpatient count × length of stay |
| `n_medications_x_time_in_hospital` | Interaction: medication count × length of stay |

**Final feature matrix:** 25,000 rows × 47 features (after OHE with `drop_first=True`).

---

## 3. Baseline Model Performance (Validation Set)

Cross-validation on the training set (5-fold, ROC-AUC scoring), then full evaluation on the held-out validation set (5,000 rows).

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.6579** | **0.6351** | 0.5954 | 0.5776 | 0.5864 | 0.2305 |
| Random Forest | 0.6563 | 0.6359 | 0.5978 | 0.5746 | 0.5860 | 0.2303 |
| Logistic Regression | 0.6504 | 0.6276 | 0.6123 | 0.5287 | 0.5675 | 0.2320 |

All metrics at default threshold (0.5). Gradient Boosting selected as best baseline by ROC-AUC.

---

## 4. Tuned Model Performance

### Hyperparameter Tuning

`RandomizedSearchCV` (20 iterations, 5-fold CV, `roc_auc` scoring) on the training set for all three model families. Gradient Boosting remained the best performer on the validation set after tuning.

### Calibration

Post-hoc probability calibration compared sigmoid vs. isotonic methods on a held-out 30% of the validation set:

| Method | Brier score (held-out) |
|---|---|
| **Sigmoid** | **0.2279** |
| Isotonic | 0.2287 |

**Best calibration method: sigmoid.**

Uncalibrated Brier: 0.2303 → Calibrated: 0.2296 (marginal improvement; probabilities were already reasonably calibrated).

### Threshold Selection

Recall-constrained threshold sweep (range 0.05–0.95, step 0.05): find the highest-precision threshold where recall ≥ 0.80.

**Optimal threshold: 0.35**

### Final Metrics

| Split | ROC-AUC | PR-AUC | Precision | Recall | F1 | Specificity | Brier |
|---|---|---|---|---|---|---|---|
| Validation | 0.6578 | 0.6354 | 0.5119 | 0.8766 | 0.6464 | 0.2582 | 0.2296 |
| **Test** | **0.6534** | **0.6233** | **0.5115** | **0.8792** | **0.6467** | **0.2548** | **0.2309** |

Val and test metrics are consistent — no overfitting to the validation threshold.
Recall constraint of ≥ 0.80 is satisfied on both splits (val: 0.877, test: 0.879).

---

## 5. SHAP Feature Importance

SHAP values computed via `shap.PermutationExplainer` on 500 validation samples drawn from the full calibrated model (treats the calibration wrapper as a black box).

### Top Features (SHAP bar importance)

| Rank | Feature | Notes |
|---|---|---|
| 1 | `total_prior_utilization` | Engineered: sum of all prior visits — strongest single predictor |
| 2 | `n_inpatient_x_time_in_hospital` | Interaction: high prior inpatient × long stay = highest risk |
| 3 | `diabetes_med_yes` | Having diabetes medication is associated with increased readmission risk |
| 4 | `age_ordinal` | Older patients have higher readmission risk |

SHAP dependence plots confirm real feature ranges (e.g., `time_in_hospital` spans 1–14 days as expected from the dataset schema).

---

## 6. Error Analysis

Error groups on the validation set (5,000 rows) at threshold = 0.35:

| Group | N | % of val set |
|---|---|---|
| True Positive (TP) | 2,061 | 41.2% |
| False Negative (FN) | 290 | 5.8% |
| True Negative (TN) | 684 | 13.7% |
| False Positive (FP) | 1,965 | 39.3% |

The low threshold (0.35) deliberately accepts a high FP rate to maximise recall — consistent with the clinical priority of not missing genuine readmissions.

### FP vs FN Profile

| Feature | FP (mean) | FN (mean) | Interpretation |
|---|---|---|---|
| `age_ordinal` | 3.55 | lower | FP patients are older on average |
| `time_in_hospital` | 4.59 | lower | FP patients have longer stays |
| `n_medications` | 16.60 | lower | FP patients have more medications |
| `total_prior_utilization` | high | very low | FN patients have near-zero prior utilisation — model misses low-history readmitters |

**FN profile:** Patients the model misses tend to have little or no prior utilisation history, making them look low-risk despite actual readmission. This is the primary failure mode.

---

## 7. Limitations

1. **Single dataset origin** — all 25,000 records come from a single source; generalisability to other institutions is unverified.
2. **Calibration held-out evaluation only** — calibration was compared on a 30% held-out sub-split of the validation set, not via cross-validation. The sigmoid advantage over isotonic is small and may not generalise.
3. **SHAP approximation** — `PermutationExplainer` is used because the calibration wrapper prevents TreeExplainer from accessing internal tree structure. SHAP values reflect the full model including the calibrator, which is correct but slightly slower and more approximate.
4. **No temporal validation** — records are not ordered chronologically; temporal leakage cannot be ruled out if the dataset has temporal structure.
5. **Low-history FN gap** — patients with no prior utilisation are systematically harder to identify (see Error Analysis §6).

---

## 8. Figures Index

### EDA (`reports/figures/eda/`)

| File | Description |
|---|---|
| `class_distribution.png` | Bar chart of readmitted vs. not readmitted counts |
| `age_distribution.png` | Age bracket frequencies |
| `numeric_distributions.png` | Histograms of all numeric features |
| `prior_utilization_distributions.png` | Outpatient / inpatient / emergency visit distributions |
| `categorical_distributions.png` | Bar charts for all categorical features |
| `medical_specialty_distribution.png` | Top specialties (including "Missing" category) |
| `correlation_matrix.png` | Pearson correlation heatmap (numeric features) |
| `target_correlations.png` | Feature correlations with the readmitted target |
| `features_vs_target.png` | Box plots of numeric features split by target |
| `readmission_rate_by_category.png` | Readmission rate per level of each categorical feature |

### Baseline Modeling (`reports/figures/modeling/`)

| File | Description |
|---|---|
| `roc_curve_baselines.png` | ROC curves for all three baseline models |
| `pr_curve_baselines.png` | Precision-Recall curves for all three baseline models |
| `calibration_baselines.png` | Calibration curves (reliability diagrams) |
| `confusion_matrix_best_baseline.png` | Confusion matrix for best baseline (GradientBoosting, threshold=0.5) |
| `feature_importance_logisticregression.png` | Top-20 absolute coefficients for Logistic Regression |
| `feature_importance_randomforest.png` | Top-20 Gini importances for Random Forest |
| `metrics_comparison_baselines.png` | Bar chart comparing key metrics across all baselines |

### Tuning & Calibration (`reports/figures/tuning/`)

| File | Description |
|---|---|
| `roc_curve_tuned_models.png` | ROC curves for all tuned models |
| `pr_curve_tuned_models.png` | Precision-Recall curves for all tuned models |
| `confusion_matrix_tuned_logisticregression.png` | Confusion matrix for tuned Logistic Regression |
| `confusion_matrix_tuned_randomforest.png` | Confusion matrix for tuned Random Forest |
| `confusion_matrix_tuned_gradientboosting.png` | Confusion matrix for tuned Gradient Boosting |
| `calibration_curve.png` | Sigmoid vs. isotonic calibration comparison |
| `threshold_analysis.png` | Precision / Recall / F1 / Specificity vs. threshold sweep |
| `confusion_matrix_calibrated_optimal.png` | Confusion matrix at optimal threshold (0.35) |

### SHAP Interpretation (`reports/figures/shap/`)

| File | Description |
|---|---|
| `shap_bar_importance.png` | Mean absolute SHAP values (global feature importance) |
| `shap_summary.png` | Beeswarm plot: SHAP values coloured by feature value |
| `shap_dependence_total_prior_utilization.png` | SHAP dependence for total prior utilisation |
| `shap_dependence_n_inpatient_x_time_in_hospital.png` | SHAP dependence for inpatient × stay interaction |
| `shap_dependence_diabetes_med_yes.png` | SHAP dependence for diabetes medication flag |
| `shap_dependence_age_ordinal.png` | SHAP dependence for age ordinal |
| `shap_patient_example_1.png` | Waterfall plot: individual patient explanation (example 1) |
| `shap_patient_example_2.png` | Waterfall plot: individual patient explanation (example 2) |
| `shap_patient_example_3.png` | Waterfall plot: individual patient explanation (example 3) |

### Error Analysis (`reports/figures/error_analysis/`)

| File | Description |
|---|---|
| `error_age_ordinal_distribution.png` | Age distribution across TP/TN/FP/FN groups |
| `error_time_in_hospital_distribution.png` | Length-of-stay distribution across error groups |
| `error_n_medications_distribution.png` | Medication count distribution across error groups |
| `error_n_inpatient_distribution.png` | Prior inpatient count distribution across error groups |
| `error_total_prior_utilization_distribution.png` | Total prior utilisation distribution across error groups |
| `false_positive_vs_negative.png` | FP vs. FN feature comparison (mean values) |
| `error_age_utilization_scatter.png` | Scatter plot of age vs. prior utilisation coloured by error group |

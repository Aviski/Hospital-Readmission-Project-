"""
interpretation.py — SHAP-based model explanation and error analysis.

Public API — SHAP interpretation
---------------------------------
    compute_shap_values(model, X, config)        → shap.Explanation
    plot_shap_summary(shap_vals, X, config)      → None (saves shap_summary.png)
    plot_shap_bar(shap_vals, config)             → None (saves shap_bar_importance.png)
    plot_shap_dependence(shap_vals, X, config)   → None (saves shap_dependence_*.png)
    plot_shap_waterfall(shap_vals, idx, config)  → None (saves shap_patient_example.png)

Public API — Error analysis
----------------------------
    compute_error_groups(model, X, y, threshold) → dict
    summarise_errors(groups, df_original, config) → pd.DataFrame
    plot_error_distributions(groups, df_orig, config) → None
    plot_error_comparison(groups, df_orig, config)   → None
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _figs(config: dict, subfolder: str) -> Path:
    """Resolve (and create) a figures sub-directory."""
    raw  = config.get("paths", {}).get("figures_dir", "reports/figures/")
    base = config.get("_base_dir")
    p    = (Path(base) / raw if base else Path(raw)) / subfolder
    p.mkdir(parents=True, exist_ok=True)
    return p


def _shap_dir(config: dict) -> Path:
    return _figs(config, "shap")


def _err_dir(config: dict) -> Path:
    return _figs(config, "error_analysis")


def _inner_model(model: Any) -> Any:
    """Unwrap a Pipeline to the core estimator for SHAP."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    # Also unwrap _PrefitCalibratedModel from modeling.py
    if hasattr(model, "base_model"):
        inner = model.base_model
        if isinstance(inner, Pipeline):
            return inner.steps[-1][1]
        return inner
    return model


def _transform_X(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Apply Pipeline preprocessing steps (e.g. StandardScaler) to X.

    Returns the transformed matrix that the final estimator actually sees.
    If ``model`` is not a Pipeline (or has no preprocessing), returns ``X``
    unchanged.
    """
    from sklearn.pipeline import Pipeline
    base = model
    if hasattr(model, "base_model"):
        base = model.base_model
    if isinstance(base, Pipeline) and len(base.steps) > 1:
        preprocessor = Pipeline(base.steps[:-1])
        X_t = preprocessor.transform(X)
        return pd.DataFrame(X_t, columns=X.columns, index=X.index)
    return X


# ---------------------------------------------------------------------------
# 1. SHAP value computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    config: dict,
    max_samples: int = 500,
) -> tuple[Any, pd.DataFrame]:
    """Compute SHAP values for the best-fit model.

    Uses ``TreeExplainer`` for tree-based models and ``LinearExplainer``
    for linear models; falls back to ``KernelExplainer`` for other types.

    Parameters
    ----------
    model       : fitted estimator or ``_PrefitCalibratedModel``.
    X           : feature DataFrame (pre-OHE, as passed to the model).
    config      : parsed config dict.
    max_samples : for ``KernelExplainer``, limit background samples for speed.

    Returns
    -------
    (shap_values, X_sample)
        - ``shap_values``: SHAP ``Explanation`` object (positive-class values).
        - ``X_sample``  : the subset of X used (same row order as shap_values).
    """
    import shap

    seed        = config.get("random_seed", 42)
    inner       = _inner_model(model)
    X_t         = _transform_X(model, X)

    n           = min(max_samples, len(X))
    rng         = np.random.default_rng(seed)
    idx         = rng.choice(len(X), size=n, replace=False)
    X_sample    = X.iloc[idx].copy()
    X_t_sample  = X_t.iloc[idx].copy()

    logger.info("Computing SHAP values for %d samples (model: %s)...",
                n, type(inner).__name__)

    try:
        if hasattr(inner, "feature_importances_"):          # tree model
            explainer   = shap.TreeExplainer(inner)
            raw         = explainer.shap_values(X_t_sample)
            # For binary classifiers raw may be a list [neg, pos]
            if isinstance(raw, list) and len(raw) == 2:
                vals = raw[1]
            else:
                vals = raw
            shap_values = shap.Explanation(
                values          = vals,
                base_values     = (explainer.expected_value[1]
                                   if isinstance(explainer.expected_value, list)
                                   else explainer.expected_value),
                data            = X_t_sample.values,
                feature_names   = list(X_t_sample.columns),
            )

        elif hasattr(inner, "coef_"):                       # linear model
            explainer   = shap.LinearExplainer(inner, X_t_sample)
            shap_values = explainer(X_t_sample)

        else:                                               # fallback
            logger.info("Using KernelExplainer (may be slow).")
            predict_fn  = lambda x: model.predict_proba(
                pd.DataFrame(x, columns=X_t_sample.columns))[:, 1]
            bg          = shap.sample(X_t_sample, min(100, n))
            explainer   = shap.KernelExplainer(predict_fn, bg)
            vals        = explainer.shap_values(X_t_sample)
            shap_values = shap.Explanation(
                values          = vals,
                base_values     = explainer.expected_value,
                data            = X_t_sample.values,
                feature_names   = list(X_t_sample.columns),
            )

    except Exception as exc:
        logger.warning("SHAP TreeExplainer failed (%s); falling back to KernelExplainer.", exc)
        predict_fn  = lambda x: model.predict_proba(
            pd.DataFrame(x, columns=X_t_sample.columns))[:, 1]
        bg          = shap.sample(X_t_sample, min(100, n))
        explainer   = shap.KernelExplainer(predict_fn, bg, seed=seed)
        vals        = explainer.shap_values(X_t_sample)
        shap_values = shap.Explanation(
            values          = vals,
            base_values     = explainer.expected_value,
            data            = X_t_sample.values,
            feature_names   = list(X_t_sample.columns),
        )

    logger.info("SHAP values computed. Shape: %s", shap_values.values.shape)
    return shap_values, X_sample


# ---------------------------------------------------------------------------
# 2. SHAP plots
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_values: Any,
    X_sample: pd.DataFrame,
    config: dict,
    max_features: int = 20,
) -> None:
    """Beeswarm summary plot — global feature impact coloured by feature value."""
    import shap

    out = _shap_dir(config) / "shap_summary.png"
    fig, ax = plt.subplots(figsize=(9, max(6, min(max_features, 20) * 0.4)))
    shap.plots.beeswarm(shap_values, max_display=max_features, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary → %s", out)


def plot_shap_bar(
    shap_values: Any,
    config: dict,
    max_features: int = 20,
) -> None:
    """Bar chart of mean absolute SHAP values (global feature importance)."""
    import shap

    out = _shap_dir(config) / "shap_bar_importance.png"
    shap.plots.bar(shap_values, max_display=max_features, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP bar importance → %s", out)


def plot_shap_dependence(
    shap_values: Any,
    X_sample: pd.DataFrame,
    config: dict,
    top_n: int = 4,
) -> None:
    """Scatter dependence plots for the top-N most important features."""
    import shap

    # Rank features by mean |SHAP|
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    feat_names = list(X_sample.columns)
    ranked = sorted(range(len(feat_names)), key=lambda i: mean_abs[i], reverse=True)
    top_features = [feat_names[i] for i in ranked[:top_n]]

    for feat in top_features:
        if feat not in feat_names:
            continue
        safe_name = feat.lower().replace(" ", "_").replace("/", "_")
        out = _shap_dir(config) / f"shap_dependence_{safe_name}.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.plots.scatter(shap_values[:, feat], color=shap_values, ax=ax, show=False)
        ax.set_title(f"SHAP Dependence — {feat}")
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved SHAP dependence (%s) → %s", feat, out)


def plot_shap_waterfall(
    shap_values: Any,
    config: dict,
    n_examples: int = 3,
) -> None:
    """Waterfall plots for individual patient predictions.

    Saves ``shap_patient_fp_N.png`` (false-positive examples) and
    ``shap_patient_fn_N.png`` (false-negative examples) if error groups
    are available, otherwise just picks the first ``n_examples`` rows.
    """
    import shap

    out_dir = _shap_dir(config)
    indices = list(range(min(n_examples, len(shap_values))))

    for k, idx in enumerate(indices):
        out = out_dir / f"shap_patient_example_{k + 1}.png"
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP waterfall (example %d) → %s", k + 1, out)


# ---------------------------------------------------------------------------
# 3. Error analysis
# ---------------------------------------------------------------------------

def compute_error_groups(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, pd.Index]:
    """Classify predictions into TP / TN / FP / FN.

    Returns a dict with keys ``"TP"``, ``"TN"``, ``"FP"``, ``"FN"``
    mapping to the index labels of rows belonging to each group.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    y_arr  = y.to_numpy()

    tp_mask = (y_pred == 1) & (y_arr == 1)
    tn_mask = (y_pred == 0) & (y_arr == 0)
    fp_mask = (y_pred == 1) & (y_arr == 0)
    fn_mask = (y_pred == 0) & (y_arr == 1)

    groups = {
        "TP": X.index[tp_mask],
        "TN": X.index[tn_mask],
        "FP": X.index[fp_mask],
        "FN": X.index[fn_mask],
    }

    for name, idx in groups.items():
        logger.info("Error group %-4s: %d rows (%.1f%%)", name,
                    len(idx), 100 * len(idx) / len(y))
    return groups


def summarise_errors(
    groups: dict[str, pd.Index],
    df_original: pd.DataFrame,
    config: dict,
    numeric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Mean statistics per error group for selected numeric features.

    Parameters
    ----------
    groups       : output of :func:`compute_error_groups`.
    df_original  : raw / pre-OHE DataFrame aligned with the model's index.
    config       : parsed config dict.
    numeric_cols : columns to summarise (auto-detected if None).

    Returns
    -------
    pd.DataFrame with rows = error groups, columns = mean numeric stats.
    """
    AUTO_COLS = [
        "Age", "Comorbidity_Index", "Chronic_Disease_Count",
        "Severity_Score", "Length_of_Stay", "Previous_Admissions_6M",
        "HbA1c_Level", "Number_of_Medications",
    ]
    if numeric_cols is None:
        numeric_cols = [c for c in AUTO_COLS if c in df_original.columns]

    rows = []
    for name, idx in groups.items():
        sub = df_original.loc[idx]
        row: dict[str, Any] = {"group": name, "n": len(sub)}
        for col in numeric_cols:
            row[f"mean_{col}"] = sub[col].mean() if col in sub.columns else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("group")
    logger.info("Error summary:\n%s", summary.to_string())
    return summary


def plot_error_distributions(
    groups: dict[str, pd.Index],
    df_original: pd.DataFrame,
    config: dict,
) -> None:
    """Violin plots comparing key numeric features across FP / FN / TP / TN."""
    AUTO_COLS = [
        "Age", "Comorbidity_Index", "Severity_Score",
        "Length_of_Stay", "Previous_Admissions_6M",
    ]
    plot_cols = [c for c in AUTO_COLS if c in df_original.columns]
    if not plot_cols:
        logger.warning("No numeric columns available for error distribution plot.")
        return

    # Build long-form DataFrame for plotting
    records = []
    for name, idx in groups.items():
        sub = df_original.loc[idx, plot_cols]
        for _, row_data in sub.iterrows():
            for col in plot_cols:
                records.append({"group": name, "feature": col, "value": row_data[col]})
    long_df = pd.DataFrame(records)

    out_dir = _err_dir(config)
    palette = {"TP": "steelblue", "TN": "lightsteelblue",
               "FP": "tomato", "FN": "salmon"}

    for col in plot_cols:
        col_data = long_df[long_df["feature"] == col]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.violinplot(data=col_data, x="group", y="value",
                       palette=palette, order=["TP", "TN", "FP", "FN"],
                       inner="box", ax=ax)
        ax.set(title=f"{col} — distribution by prediction group",
               xlabel="Prediction group", ylabel=col)
        safe = col.lower().replace(" ", "_")
        out = out_dir / f"error_{safe}_distribution.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved error distribution (%s) → %s", col, out)


def plot_error_age_comorbidity(
    groups: dict[str, pd.Index],
    df_original: pd.DataFrame,
    config: dict,
) -> None:
    """Scatter plot of Age vs Comorbidity_Index coloured by error group."""
    needed = {"Age", "Comorbidity_Index"}
    if not needed.issubset(df_original.columns):
        logger.warning("Age or Comorbidity_Index not in df — skipping scatter plot.")
        return

    records = []
    palette = {"TP": "steelblue", "TN": "lightsteelblue",
               "FP": "tomato",    "FN": "darkorange"}

    for name, idx in groups.items():
        sub = df_original.loc[idx, ["Age", "Comorbidity_Index"]]
        sub = sub.assign(group=name)
        records.append(sub)

    combined = pd.concat(records)

    fig, ax = plt.subplots(figsize=(9, 6))
    for grp, color in palette.items():
        sub = combined[combined["group"] == grp]
        ax.scatter(sub["Age"], sub["Comorbidity_Index"],
                   c=color, label=grp, alpha=0.4, s=15)
    ax.set(xlabel="Age", ylabel="Comorbidity Index",
           title="Age vs Comorbidity Index — coloured by prediction group")
    ax.legend(title="Group", fontsize=9)

    out = _err_dir(config) / "error_age_comorbidity_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved age-comorbidity scatter → %s", out)


def plot_false_positive_vs_negative(
    groups: dict[str, pd.Index],
    df_original: pd.DataFrame,
    config: dict,
) -> None:
    """Side-by-side bar chart: FP vs FN mean statistics for key numeric features."""
    AUTO_COLS = [
        "Age", "Comorbidity_Index", "Severity_Score",
        "Length_of_Stay", "Previous_Admissions_6M", "Number_of_Medications",
    ]
    plot_cols = [c for c in AUTO_COLS if c in df_original.columns]
    if not plot_cols:
        return

    fp_idx = groups.get("FP", pd.Index([]))
    fn_idx = groups.get("FN", pd.Index([]))

    fp_means = df_original.loc[fp_idx, plot_cols].mean()
    fn_means = df_original.loc[fn_idx, plot_cols].mean()

    comparison = pd.DataFrame({"FP (False Positives)": fp_means,
                                "FN (False Negatives)": fn_means})

    # Normalise by overall mean for comparability across scales
    overall_mean = df_original[plot_cols].mean().replace(0, np.nan)
    comparison_norm = comparison.div(overall_mean, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    comparison.plot.barh(ax=axes[0], color=["tomato", "darkorange"])
    axes[0].set_title("FP vs FN — raw means")
    axes[0].set_xlabel("Mean value")

    comparison_norm.plot.barh(ax=axes[1], color=["tomato", "darkorange"])
    axes[1].set_title("FP vs FN — normalised by overall mean")
    axes[1].axvline(1.0, color="black", linestyle="--", lw=1, label="Overall mean")
    axes[1].set_xlabel("Ratio to overall mean")
    axes[1].legend(fontsize=8)

    out = _err_dir(config) / "false_positive_vs_negative.png"
    fig.suptitle("False Positive vs False Negative Profiles", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved FP vs FN comparison → %s", out)


def plot_error_diagnosis_distribution(
    groups: dict[str, pd.Index],
    df_original: pd.DataFrame,
    config: dict,
    diag_col: str = "Primary_Diagnosis_Group",
) -> None:
    """Stacked bar: diagnosis group distribution for FP and FN."""
    if diag_col not in df_original.columns:
        logger.warning("'%s' not in df — skipping diagnosis plot.", diag_col)
        return

    fp_idx = groups.get("FP", pd.Index([]))
    fn_idx = groups.get("FN", pd.Index([]))

    fp_counts = df_original.loc[fp_idx, diag_col].value_counts(normalize=True)
    fn_counts = df_original.loc[fn_idx, diag_col].value_counts(normalize=True)

    combined = pd.DataFrame({"FP": fp_counts, "FN": fn_counts}).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    combined.plot.bar(ax=ax, color=["tomato", "darkorange"], width=0.7)
    ax.set(title=f"Diagnosis distribution — FP vs FN ({diag_col})",
           ylabel="Proportion", xlabel="")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Error type")

    out = _err_dir(config) / "error_diagnosis_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diagnosis distribution → %s", out)

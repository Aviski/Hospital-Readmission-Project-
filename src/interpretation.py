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


def _predict_proba_pos(model: Any, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Wrapper: call model.predict_proba on a numpy array, return P(positive)."""
    return model.predict_proba(pd.DataFrame(X, columns=feature_names))[:, 1]


# ---------------------------------------------------------------------------
# 1. SHAP value computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    config: dict,
    max_samples: int = 500,
) -> tuple[Any, pd.DataFrame]:
    """Compute SHAP values for the full calibrated model using unscaled features.

    Uses model-agnostic ``shap.Explainer`` which calls ``model.predict_proba``
    directly — explaining the complete pipeline including any calibration layer.
    Feature values in SHAP plots reflect the original (unscaled) input space
    since the model itself handles scaling internally.

    Parameters
    ----------
    model       : fitted estimator or ``_PrefitCalibratedModel``.
    X           : feature DataFrame in original (unscaled) space.
    config      : parsed config dict.
    max_samples : number of samples used for both background and explanation.

    Returns
    -------
    (shap_values, X_sample)
        - ``shap_values``: SHAP ``Explanation`` object (positive-class values).
        - ``X_sample``  : the subset of X used (same row order as shap_values).
    """
    import shap

    seed         = config.get("random_seed", 42)
    feature_names = list(X.columns)

    n        = min(max_samples, len(X))
    rng      = np.random.default_rng(seed)
    idx      = rng.choice(len(X), size=n, replace=False)
    X_sample = X.iloc[idx].copy()

    logger.info("Computing SHAP values for %d samples via model-agnostic Explainer...", n)

    background = shap.sample(X_sample, min(100, n), random_state=seed)
    predict_fn = lambda x: _predict_proba_pos(model, x, feature_names)

    try:
        explainer   = shap.Explainer(predict_fn, background, feature_names=feature_names)
        shap_values = explainer(X_sample, silent=True)

    except (ValueError, TypeError, ImportError) as exc:
        logger.warning(
            "shap.Explainer failed (%s: %s). Falling back to PermutationExplainer.",
            type(exc).__name__, exc,
        )
        explainer   = shap.PermutationExplainer(predict_fn, background,
                                                feature_names=feature_names)
        shap_values = explainer(X_sample, silent=True)

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
    target_col = config.get("data", {}).get("target_column", "readmitted")
    default_cols = [
        c for c in df_original.select_dtypes(include="number").columns
        if c != target_col
    ]
    if numeric_cols is None:
        numeric_cols = default_cols

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
    target_col = config.get("data", {}).get("target_column", "readmitted")
    plot_cols = [
        c for c in df_original.select_dtypes(include="number").columns
        if c != target_col
    ]
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
                       hue="group", palette=palette, order=["TP", "TN", "FP", "FN"],
                       dodge=False, legend=False, inner="box", ax=ax)
        ax.set(title=f"{col} — distribution by prediction group",
               xlabel="Prediction group", ylabel=col)
        safe = col.lower().replace(" ", "_")
        out = out_dir / f"error_{safe}_distribution.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved error distribution (%s) → %s", col, out)


def plot_error_utilization_scatter(
    groups: dict[str, pd.Index],
    df_original: pd.DataFrame,
    config: dict,
) -> None:
    """Scatter plot of age_ordinal vs total_prior_utilization coloured by error group."""
    needed = {"age_ordinal", "total_prior_utilization"}
    if not needed.issubset(df_original.columns):
        logger.warning(
            "age_ordinal or total_prior_utilization not in df — skipping scatter plot."
        )
        return

    records = []
    palette = {"TP": "steelblue", "TN": "lightsteelblue",
               "FP": "tomato",    "FN": "darkorange"}

    for name, idx in groups.items():
        sub = df_original.loc[idx, ["age_ordinal", "total_prior_utilization"]]
        sub = sub.assign(group=name)
        records.append(sub)

    combined = pd.concat(records)

    fig, ax = plt.subplots(figsize=(9, 6))
    for grp, color in palette.items():
        sub = combined[combined["group"] == grp]
        ax.scatter(sub["age_ordinal"], sub["total_prior_utilization"],
                   c=color, label=grp, alpha=0.4, s=15)
    ax.set(xlabel="Age (ordinal)", ylabel="Total Prior Utilization",
           title="Age vs Prior Utilization — coloured by prediction group")
    ax.legend(title="Group", fontsize=9)

    out = _err_dir(config) / "error_age_utilization_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved age-utilization scatter → %s", out)


def plot_false_positive_vs_negative(
    groups: dict[str, pd.Index],
    df_original: pd.DataFrame,
    config: dict,
) -> None:
    """Side-by-side bar chart: FP vs FN mean statistics for key numeric features."""
    target_col = config.get("data", {}).get("target_column", "readmitted")
    plot_cols = [
        c for c in df_original.select_dtypes(include="number").columns
        if c != target_col
    ]
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
    diag_col: str = "diag_1",
) -> None:
    """Stacked bar: primary diagnosis distribution for FP and FN."""
    if diag_col not in df_original.columns:
        raise ValueError(
            f"diag_col '{diag_col}' not found in the provided DataFrame. "
            "Pass the pre-OHE analysis frame (readmission_features_raw.csv aligned to X_val.index), "
            "not the encoded feature matrix."
        )

    fp_idx = groups.get("FP", pd.Index([]))
    fn_idx = groups.get("FN", pd.Index([]))

    fp_counts = df_original.loc[fp_idx, diag_col].value_counts(normalize=True)
    fn_counts = df_original.loc[fn_idx, diag_col].value_counts(normalize=True)

    combined = pd.DataFrame({"FP": fp_counts, "FN": fn_counts}).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    combined.plot.bar(ax=ax, color=["tomato", "darkorange"], width=0.7)
    ax.set(title=f"Primary diagnosis distribution — FP vs FN ({diag_col})",
           ylabel="Proportion", xlabel="")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Error type")

    out = _err_dir(config) / "error_diagnosis_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diagnosis distribution → %s", out)

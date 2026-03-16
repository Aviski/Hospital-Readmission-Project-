"""
modeling.py — Training, evaluation, tuning, calibration, and model persistence.

Public API — Baseline
----------------------
    load_features(config, base_dir)          → X, y, feature_names
    make_splits(X, y, config)                → X_train, X_val, X_test, …
    build_baselines(config)                  → {name: estimator}
    cross_validate_model(model, X, y, cfg)   → dict of CV metric arrays
    evaluate(model, X, y, threshold)         → dict of scalar metrics
    plot_roc_curves(models, X_val, y_val, cfg, title, filename)
    plot_pr_curves(models, X_val, y_val, cfg, title, filename)
    plot_confusion_matrix(model, X, y, name, cfg, threshold, filename)
    plot_calibration_curves(models, X_val, y_val, cfg, filename)
    plot_feature_importance(model, feat_names, name, cfg, top_n)
    compare_models(eval_results)             → pd.DataFrame
    select_best_model(summary, metric)       → name str

Public API — Tuning & calibration (Phase 4)
--------------------------------------------
    build_param_distributions(model_name, config)   → dict
    tune_model(model, param_dist, X, y, config)     → {estimator, best_params, cv_score}
    calibrate_model(model, X_calib, y_calib, config) → best CalibratedClassifierCV
    threshold_sweep(model, X, y, config)            → (sweep_df, optimal_threshold)
    plot_threshold_analysis(sweep_df, opt_thresh, cfg, filename)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _figures_dir(config: dict, subfolder: str = "") -> Path:
    """Return (and create) a figures output directory.

    Resolves against ``config['_base_dir']`` when present (set by notebook
    callers to anchor relative paths to the project root).  An optional
    ``subfolder`` (e.g. ``"modeling"``, ``"tuning"``) is appended after the
    base figures directory, allowing plots to be organised into sub-directories.
    The caller can also set ``config['_figures_subfolder']`` as a default for
    the entire session; an explicit ``subfolder`` argument takes precedence.

    Final path: ``{figures_dir}/{subfolder}/``
    """
    raw      = config.get("paths", {}).get("figures_dir", "reports/figures/")
    base     = config.get("_base_dir")
    sub      = subfolder or config.get("_figures_subfolder", "")
    p        = Path(base) / raw if base else Path(raw)
    if sub:
        p = p / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def _inner_estimator(model: Any) -> Any:
    """Return the core estimator from a Pipeline, or the model itself."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_features(
    config: dict,
    base_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the processed feature dataset and separate X from y.

    Steps applied after loading:
    1. Column-name sanitisation (spaces → underscores).
    2. Drop leakage columns (``model.exclude_columns``).
    3. Drop redundant / collinear columns (``model.drop_redundant_cols``).

    Parameters
    ----------
    config   : parsed config dict.
    base_dir : optional project root to resolve relative paths (useful from
               notebooks in a sub-directory).

    Returns
    -------
    X, y, feature_names
    """
    raw_path = config["paths"]["features_data"]
    path = Path(base_dir) / raw_path if base_dir else Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Features file not found: {path.resolve()}\n"
            "Run: clean_data → create_features → encode_features → save CSV"
        )

    try:
        df = pd.read_csv(path, index_col="row_id")
    except ValueError:
        logger.warning(
            "Features CSV does not contain row_id; loading with default integer index: %s",
            path,
        )
        df = pd.read_csv(path)
    logger.info("Loaded feature dataset: %d rows × %d columns", *df.shape)

    # Sanitise column names
    original = df.columns.tolist()
    df.columns = df.columns.str.replace(" ", "_", regex=False)
    renamed = [(o, n) for o, n in zip(original, df.columns) if o != n]
    if renamed:
        logger.info("Sanitised %d column name(s): %s", len(renamed), renamed)

    target    = config["data"]["target_column"]
    model_cfg = config.get("model", {})

    # Leakage exclusions
    exclude = [c.replace(" ", "_") for c in model_cfg.get("exclude_columns", [])]
    to_drop = [c for c in exclude if c in df.columns]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        logger.info("Excluded leakage column(s): %s", to_drop)

    # Redundant / collinear column removal
    redundant = [c.replace(" ", "_") for c in model_cfg.get("drop_redundant_cols", [])]
    to_drop_r = [c for c in redundant if c in df.columns]
    if to_drop_r:
        df.drop(columns=to_drop_r, inplace=True)
        logger.info("Dropped redundant column(s): %s", to_drop_r)

    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found. Available: {df.columns.tolist()}")

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].copy()
    y = df[target].copy()

    neg, pos = (y == 0).sum(), (y == 1).sum()
    logger.info(
        "Feature matrix: %d rows × %d features | pos=%d (%.1f%%) neg=%d (%.1f%%)",
        X.shape[0], X.shape[1], pos, 100 * pos / len(y), neg, 100 * neg / len(y),
    )
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# 2. Train / validation / test split
# ---------------------------------------------------------------------------

def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
    """Stratified 60 / 20 / 20 train / validation / test split."""
    data_cfg = config.get("data", {})
    seed     = config.get("random_seed", 42)
    test_sz  = data_cfg.get("test_size", 0.20)
    val_sz   = data_cfg.get("val_size",  0.20)
    strat    = y if data_cfg.get("stratify", True) else None

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_sz, stratify=strat, random_state=seed
    )
    val_frac = val_sz / (1.0 - test_sz)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac,
        stratify=y_tv if strat is not None else None,
        random_state=seed,
    )

    logger.info(
        "Split: train=%d | val=%d | test=%d",
        len(y_train), len(y_val), len(y_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# 3. Build baseline estimators
# ---------------------------------------------------------------------------

def build_baselines(config: dict) -> dict[str, Any]:
    """Instantiate the three baseline estimators from config hyperparameters.

    Logistic Regression is wrapped in a ``StandardScaler → LR`` Pipeline so
    that the scaler is applied correctly during cross-validation without
    data leakage.  Tree-based models (RF, HGBC) are scale-invariant and are
    returned unwrapped.

    Gradient Boosting uses sklearn's ``HistGradientBoostingClassifier``.
    XGBoost and LightGBM are NOT dependencies of this project.

    Returns
    -------
    dict mapping model name → unfitted sklearn-compatible estimator.
    """
    seed   = config.get("random_seed", 42)
    m_cfg  = config.get("model", {})
    lr_cfg = m_cfg.get("logistic_regression", {})
    rf_cfg = m_cfg.get("random_forest", {})
    gb_cfg = m_cfg.get("gradient_boosting", {})

    # Logistic Regression — wrapped in Pipeline with StandardScaler
    logit_pipeline = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            penalty=lr_cfg.get("penalty", "l2"),
            max_iter=lr_cfg.get("max_iter", 1000),
            class_weight=lr_cfg.get("class_weight", "balanced"),
            solver=lr_cfg.get("solver", "lbfgs"),
            random_state=seed,
        )),
    ])

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 200),
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 10),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        n_jobs=rf_cfg.get("n_jobs", -1),
        random_state=seed,
    )

    # Gradient Boosting — sklearn HistGradientBoostingClassifier
    gb = HistGradientBoostingClassifier(
        max_iter=gb_cfg.get("max_iter", 200),
        learning_rate=gb_cfg.get("learning_rate", 0.05),
        max_depth=gb_cfg.get("max_depth", 5),
        min_samples_leaf=gb_cfg.get("min_samples_leaf", 20),
        class_weight=gb_cfg.get("class_weight", "balanced"),
        random_state=seed,
    )

    models = {
        "LogisticRegression": logit_pipeline,
        "RandomForest":       rf,
        "GradientBoosting":   gb,
    }
    logger.info("Baseline estimators: %s", list(models.keys()))
    return models


# ---------------------------------------------------------------------------
# 4. Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> dict[str, np.ndarray]:
    """Stratified k-fold cross-validation on the training set.

    Returns dict mapping ``test_<metric>`` → array of per-fold scores.
    """
    cv_folds = config.get("model", {}).get("cv_folds", 5)
    seed     = config.get("random_seed", 42)
    cv       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    logger.info(
        "Cross-validating %s (%d folds)...",
        _inner_estimator(model).__class__.__name__,
        cv_folds,
    )
    results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "average_precision": "average_precision", "f1": "f1"},
        return_train_score=False,
        n_jobs=-1,
    )
    for k, v in results.items():
        if k.startswith("test_"):
            logger.info("  CV %s: %.4f ± %.4f", k[5:], v.mean(), v.std())
    return results


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics at a given probability threshold.

    Returns
    -------
    dict: roc_auc, pr_auc, accuracy, precision, recall, f1, brier.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc":     roc_auc_score(y, y_prob),
        "pr_auc":      average_precision_score(y, y_prob),
        "accuracy":    accuracy_score(y, y_pred),
        "precision":   precision_score(y, y_pred, zero_division=0),
        "recall":      recall_score(y, y_pred, zero_division=0),
        "f1":          f1_score(y, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "brier":       brier_score_loss(y, y_prob),
    }


# ---------------------------------------------------------------------------
# 6. Plotting helpers
# ---------------------------------------------------------------------------

def plot_roc_curves(
    fitted_models: dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    title: str = "ROC Curves",
    filename: str = "roc_curve_baselines.png",
) -> None:
    """ROC curves for all models on the validation set."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, model), color in zip(
        fitted_models.items(), sns.color_palette("tab10", len(fitted_models))
    ):
        prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, prob)
        ax.plot(fpr, tpr, label=f"{name}  AUC={roc_auc_score(y_val, prob):.3f}",
                color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set(xlabel="FPR", ylabel="TPR", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=9)
    out = _figures_dir(config) / filename
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved ROC curve → %s", out)


def plot_pr_curves(
    fitted_models: dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    title: str = "Precision-Recall Curves",
    filename: str = "pr_curve_baselines.png",
) -> None:
    """Precision-Recall curves for all models."""
    prevalence = float(y_val.mean())
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, model), color in zip(
        fitted_models.items(), sns.color_palette("tab10", len(fitted_models))
    ):
        prob = model.predict_proba(X_val)[:, 1]
        prec, rec, _ = precision_recall_curve(y_val, prob)
        ax.plot(rec, prec,
                label=f"{name}  PR-AUC={average_precision_score(y_val, prob):.3f}",
                color=color, lw=2)
    ax.axhline(prevalence, color="k", linestyle="--", lw=1,
               label=f"No-skill ({prevalence:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right", fontsize=9)
    out = _figures_dir(config) / filename
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved PR curve → %s", out)


def plot_confusion_matrix(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    config: dict,
    threshold: float = 0.5,
    filename: str = "confusion_matrix_best_baseline.png",
) -> None:
    """Confusion matrix at a given probability threshold."""
    y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    cm     = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        cm, display_labels=["Not Readmitted (0)", "Readmitted (1)"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold={threshold:.2f})")
    out = _figures_dir(config) / filename
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved confusion matrix → %s", out)


def plot_calibration_curves(
    fitted_models: dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    n_bins: int = 10,
    filename: str = "calibration_baselines.png",
) -> None:
    """Reliability curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, model), color in zip(
        fitted_models.items(), sns.color_palette("tab10", len(fitted_models))
    ):
        CalibrationDisplay.from_estimator(
            model, X_val, y_val, n_bins=n_bins, ax=ax, name=name, color=color
        )
    ax.set_title("Calibration Curves")
    ax.legend(loc="upper left", fontsize=9)
    out = _figures_dir(config) / filename
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved calibration curves → %s", out)


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str,
    config: dict,
    top_n: int = 20,
) -> None:
    """Feature importances: Gini (RF/XGB), |coef| (LR Pipeline), or skip."""
    inner = _inner_estimator(model)
    if hasattr(inner, "feature_importances_"):
        importances = inner.feature_importances_
        xlabel = "Gini Importance"
    elif hasattr(inner, "coef_"):
        importances = np.abs(inner.coef_[0])
        xlabel = "|Coefficient|"
    else:
        logger.info("No feature importance available for %s — skipping.", model_name)
        return

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    ax.barh([feature_names[i] for i in idx], importances[idx],
            color=sns.color_palette("muted")[0])
    ax.set(title=f"{xlabel} — {model_name}", xlabel=xlabel)
    safe = model_name.lower().replace(" ", "_")
    out  = _figures_dir(config) / f"feature_importance_{safe}.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved feature importance → %s", out)


# ---------------------------------------------------------------------------
# 7. Model comparison and selection
# ---------------------------------------------------------------------------

def compare_models(eval_results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Metrics summary DataFrame sorted by ROC-AUC descending."""
    df = pd.DataFrame(eval_results).T
    df.index.name = "model"
    sort_col = "roc_auc" if "roc_auc" in df.columns else df.columns[0]
    return df.sort_values(sort_col, ascending=False).round(4)


def select_best_model(summary: pd.DataFrame, metric: str = "roc_auc") -> str:
    """Return the model name with the highest value for ``metric``."""
    if metric not in summary.columns:
        metric = summary.columns[0]
    best = summary[metric].idxmax()
    logger.info("Best model by %s: %s (%.4f)", metric, best, summary.loc[best, metric])
    return best


# ---------------------------------------------------------------------------
# 8. Hyperparameter tuning
# ---------------------------------------------------------------------------

def build_param_distributions(model_name: str, config: dict) -> dict:
    """Return the parameter search space for a given model name.

    Parameter names for ``LogisticRegression`` are prefixed with
    ``classifier__`` to match the ``StandardScaler → LR`` Pipeline
    returned by :func:`build_baselines`.

    Parameters
    ----------
    model_name : one of ``"LogisticRegression"``, ``"RandomForest"``,
                 ``"GradientBoosting"``.
    config     : parsed config dict.

    Returns
    -------
    dict suitable for ``RandomizedSearchCV(param_distributions=...)``.
    """
    spaces = (
        config.get("model", {})
              .get("tuning", {})
              .get("param_spaces", {})
    )

    if model_name in spaces:
        return dict(spaces[model_name])

    # Sensible fallback defaults
    defaults: dict[str, dict] = {
        "LogisticRegression": {
            "classifier__C":       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver":  ["liblinear", "saga"],
        },
        "RandomForest": {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [5, 10, 20],
        },
        "GradientBoosting": {
            "learning_rate":    [0.01, 0.05, 0.1, 0.2],
            "max_iter":         [100, 150, 200],
            "max_depth":        [3, 4, 5, 6],
            "min_samples_leaf": [10, 20, 30],
        },
    }
    return defaults.get(model_name, {})


def tune_model(
    model: Any,
    param_dist: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> dict:
    """Randomised hyperparameter search using StratifiedKFold CV.

    Parameters
    ----------
    model      : unfitted estimator (or Pipeline).
    param_dist : parameter distributions; use :func:`build_param_distributions`.
    X_train, y_train : training data.
    config     : parsed config dict.

    Returns
    -------
    dict with keys:
        ``estimator``   – best fitted estimator
        ``best_params`` – winning hyperparameter dict
        ``cv_score``    – best cross-validated score
    """
    tuning_cfg = config.get("model", {}).get("tuning", {})
    seed       = config.get("random_seed", 42)
    n_iter     = tuning_cfg.get("n_iter", 20)
    cv_folds   = tuning_cfg.get("cv_folds", 5)
    scoring    = tuning_cfg.get("scoring", "roc_auc")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    logger.info(
        "RandomizedSearchCV: model=%s  n_iter=%d  cv=%d  scoring=%s",
        type(model).__name__, n_iter, cv_folds, scoring,
    )
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        refit=True,
        n_jobs=-1,
        random_state=seed,
        verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info(
        "Best %s: %.4f | params: %s",
        scoring, search.best_score_, search.best_params_,
    )
    return {
        "estimator":   search.best_estimator_,
        "best_params": search.best_params_,
        "cv_score":    search.best_score_,
    }


# ---------------------------------------------------------------------------
# 9. Probability calibration
# ---------------------------------------------------------------------------

class _PrefitCalibratedModel:
    """Lightweight wrapper that applies post-hoc calibration to a fitted model.

    Equivalent to the legacy ``CalibratedClassifierCV(cv="prefit")`` behaviour
    that was removed in scikit-learn 1.6+.  The base model's weights are frozen;
    only the calibration layer is fitted on the calibration set.

    Parameters
    ----------
    base_model  : already-fitted estimator with ``predict_proba``.
    calibrator  : fitted calibration object.
                  - sigmoid: ``LogisticRegression`` fitted on raw probs.
                  - isotonic: ``IsotonicRegression`` fitted on raw probs.
    method      : ``"sigmoid"`` or ``"isotonic"`` (for labelling / logging).
    """

    def __init__(self, base_model: Any, calibrator: Any, method: str) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.method     = method

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1]
        if self.method == "sigmoid":
            cal_pos = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            cal_pos = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal_pos, cal_pos])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def _fit_prefit_calibrator(model: Any, X_calib: pd.DataFrame,
                            y_calib: pd.Series, method: str) -> "_PrefitCalibratedModel":
    """Fit a post-hoc calibrator on the raw probability outputs of ``model``."""
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.isotonic import IsotonicRegression as _IR

    raw = model.predict_proba(X_calib)[:, 1]
    if method == "sigmoid":
        cal = _LR(C=1.0, max_iter=200)
        cal.fit(raw.reshape(-1, 1), y_calib)
    elif method == "isotonic":
        cal = _IR(out_of_bounds="clip")
        cal.fit(raw, y_calib)
    else:
        raise ValueError(f"Unknown calibration method: '{method}'")
    return _PrefitCalibratedModel(model, cal, method)


def calibrate_model(
    model: Any,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    config: dict,
    save_plot: bool = True,
) -> "_PrefitCalibratedModel":
    """Compare sigmoid and isotonic calibration; return the better method.

    Both calibrators are fitted on a 70% sub-split of the calibration set.
    Brier scores are evaluated on the held-out 30% to avoid self-referential
    comparison (fitting and evaluating on the same data).

    Parameters
    ----------
    model            : already-fitted estimator or Pipeline.
    X_calib, y_calib : calibration data (typically the validation set).
    config           : parsed config dict.
    save_plot        : if True, saves a calibration comparison figure.

    Returns
    -------
    _PrefitCalibratedModel with the better calibration method applied.
    """
    cal_cfg = config.get("model", {}).get("calibration", {})
    methods = cal_cfg.get("methods", ["sigmoid", "isotonic"])
    seed    = config.get("random_seed", 42)

    # 70/30 sub-split: fit calibrator on X_calib_fit, evaluate on X_calib_eval
    X_calib_fit, X_calib_eval, y_calib_fit, y_calib_eval = train_test_split(
        X_calib, y_calib, test_size=0.30, random_state=seed, stratify=y_calib
    )

    results: dict[str, tuple["_PrefitCalibratedModel", float]] = {}
    for method in methods:
        cal_model = _fit_prefit_calibrator(model, X_calib_fit, y_calib_fit, method)
        brier = brier_score_loss(y_calib_eval, cal_model.predict_proba(X_calib_eval)[:, 1])
        results[method] = (cal_model, brier)
        logger.info("Calibration (%s): Brier=%.4f (held-out eval)", method, brier)

    best_method = min(results, key=lambda m: results[m][1])
    logger.info("Best calibration method: %s", best_method)

    # Refit best calibrator on full calibration set
    best_cal = _fit_prefit_calibrator(model, X_calib, y_calib, best_method)

    if save_plot:
        fig, ax = plt.subplots(figsize=(7, 6))
        palette = sns.color_palette("tab10", len(results) + 1)

        # Uncalibrated reliability curve
        raw_prob = model.predict_proba(X_calib_eval)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_calib_eval, raw_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="s", color=palette[0], label="Uncalibrated")

        for i, (method, (cal_model, brier)) in enumerate(results.items(), 1):
            cal_prob = cal_model.predict_proba(X_calib_eval)[:, 1]
            frac_pos_c, mean_pred_c = calibration_curve(y_calib_eval, cal_prob, n_bins=10)
            ax.plot(mean_pred_c, frac_pos_c, marker="s", color=palette[i],
                    label=f"{method.capitalize()} (Brier={brier:.4f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
               title="Calibration Comparison (held-out eval)", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left", fontsize=9)
        out = _figures_dir(config) / "calibration_curve.png"
        fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        logger.info("Saved calibration comparison → %s", out)

    return best_cal


# ---------------------------------------------------------------------------
# 10. Threshold analysis
# ---------------------------------------------------------------------------

def threshold_sweep(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> tuple[pd.DataFrame, float]:
    """Evaluate precision, recall, and F1 across a range of thresholds.

    Parameters
    ----------
    model  : fitted estimator with ``predict_proba``.
    X, y   : evaluation split (typically validation set).
    config : parsed config dict.

    Returns
    -------
    sweep_df : pd.DataFrame with columns
               [threshold, precision, recall, f1, specificity, accuracy].
    optimal  : float — threshold that maximises the objective in
               ``config['model']['threshold']['optimize_by']``.
    """
    thr_cfg  = config.get("model", {}).get("threshold", {})
    lo       = thr_cfg.get("sweep_min",  0.05)
    hi       = thr_cfg.get("sweep_max",  0.95)
    step     = thr_cfg.get("sweep_step", 0.05)
    optimize = thr_cfg.get("optimize_by", "recall")
    recall_target = thr_cfg.get("recall_target", 0.80)

    n_steps    = round((hi - lo) / step) + 1
    thresholds = np.linspace(lo, hi, n_steps)   # linspace avoids float accumulation error
    y_prob     = model.predict_proba(X)[:, 1]

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold":   round(float(t), 3),
            "precision":   precision_score(y, y_pred, zero_division=0),
            "recall":      recall_score(y, y_pred, zero_division=0),
            "f1":          f1_score(y, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "accuracy":    accuracy_score(y, y_pred),
        })

    sweep_df = pd.DataFrame(rows)

    if optimize == "recall":
        # Find the highest-precision threshold where recall >= recall_target
        candidates = sweep_df[sweep_df["recall"] >= recall_target]
        if candidates.empty:
            logger.warning(
                "No threshold achieves recall >= %.2f. Falling back to max-F1.", recall_target
            )
            optimal = float(sweep_df.loc[sweep_df["f1"].idxmax(), "threshold"])
        else:
            optimal = float(candidates.loc[candidates["precision"].idxmax(), "threshold"])
    elif optimize in sweep_df.columns:
        optimal = float(sweep_df.loc[sweep_df[optimize].idxmax(), "threshold"])
    else:
        logger.warning("optimize_by='%s' not in sweep columns — defaulting to f1.", optimize)
        optimal = float(sweep_df.loc[sweep_df["f1"].idxmax(), "threshold"])

    logger.info(
        "Threshold sweep: optimal threshold by %s = %.3f", optimize, optimal
    )
    return sweep_df, optimal


def plot_threshold_analysis(
    sweep_df: pd.DataFrame,
    optimal_threshold: float,
    config: dict,
    filename: str = "threshold_analysis.png",
) -> None:
    """Plot precision, recall, F1, and specificity against decision threshold.

    Parameters
    ----------
    sweep_df           : output of :func:`threshold_sweep`.
    optimal_threshold  : value to mark with a vertical dashed line.
    config             : parsed config dict.
    filename           : output filename within the figures directory.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("tab10")

    metrics = ["precision", "recall", "f1", "specificity"]
    for metric, color in zip(metrics, palette):
        ax.plot(sweep_df["threshold"], sweep_df[metric],
                label=metric.capitalize(), color=color, lw=2)

    ax.axvline(optimal_threshold, color="black", linestyle="--", lw=1.5,
               label=f"Optimal ({optimal_threshold:.2f})")
    ax.set(
        xlabel="Decision Threshold",
        ylabel="Score",
        title="Threshold Analysis — Precision / Recall / F1 / Specificity",
        xlim=(sweep_df["threshold"].min(), sweep_df["threshold"].max()),
        ylim=(0, 1.05),
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    out = _figures_dir(config) / filename
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved threshold analysis → %s", out)

"""
modeling.py — Training, evaluation, calibration, and model persistence.

Public API — Baseline phase
----------------------------
    load_features(config)                    → X, y, feature_names
    make_splits(X, y, config)                → X_train, X_val, X_test, y_train, y_val, y_test
    build_baselines(config)                  → {name: unfitted estimator}
    cross_validate_model(model, X, y, cfg)   → dict of CV metric arrays
    evaluate(model, X, y, threshold)         → dict of scalar metrics
    plot_roc_curves(results, y_val, dir)
    plot_pr_curves(results, y_val, dir)
    plot_confusion_matrix(y_true, y_pred, name, dir)
    plot_calibration_curves(results, y_val, dir)
    compare_models(eval_results)             → pd.DataFrame summary
    select_best_model(summary_df, metric)    → model name string

Public API — Tuning / calibration scaffold (Phase 4)
------------------------------------------------------
    tune_model(model, param_dist, X, y, config)    → best fitted estimator
    calibrate_model(model, X_calib, y_calib, cfg)  → calibrated estimator

Pipeline order
--------------
    X, y, feat_names = load_features(config)
    X_tr, X_val, X_te, y_tr, y_val, y_te = make_splits(X, y, config)
    baselines = build_baselines(config)
    # fit each model on X_tr, evaluate on X_val
    summary = compare_models(eval_results)
    best_name = select_best_model(summary)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.utils import get_logger, save_model

logger = get_logger(__name__)

# XGBoost is optional — fall back to sklearn HistGradientBoostingClassifier
try:
    import xgboost as xgb  # type: ignore
    _XGBOOST_AVAILABLE = True
    logger.info("XGBoost detected — will use XGBClassifier as gradient-boosting backend.")
except ImportError:
    _XGBOOST_AVAILABLE = False
    logger.info(
        "XGBoost not installed — using sklearn HistGradientBoostingClassifier instead. "
        "Install with: pip install xgboost"
    )


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_features(
    config: dict,
    base_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the processed feature dataset and separate X from y.

    Also sanitises column names (spaces → underscores) to prevent downstream
    issues with certain estimators and serialisation formats.

    Parameters
    ----------
    config:
        Parsed config dict (from ``load_config``).
    base_dir:
        Optional project root directory.  When provided, relative paths in
        ``config['paths']`` are resolved against it.  Useful when calling
        from notebooks in a subdirectory.  Defaults to the current working
        directory.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns except target and excluded columns).
    y : pd.Series
        Binary target vector.
    feature_names : list[str]
        Ordered list of feature column names used in X.
    """
    raw_path = config["paths"]["features_data"]
    if base_dir is not None:
        path = Path(base_dir) / raw_path
    else:
        path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Features file not found: {path.resolve()}\n"
            "Run the preprocessing pipeline first:\n"
            "  clean_data → create_features → encode_features → save CSV"
        )

    df = pd.read_csv(path)
    logger.info("Loaded feature dataset: %d rows × %d columns", *df.shape)

    # Sanitise column names: spaces → underscores
    original_cols = df.columns.tolist()
    df.columns = df.columns.str.replace(" ", "_", regex=False)
    renamed = [
        (o, n) for o, n in zip(original_cols, df.columns.tolist()) if o != n
    ]
    if renamed:
        logger.info("Sanitised %d column name(s): %s", len(renamed), renamed)

    target = config["data"]["target_column"]
    model_cfg = config.get("model", {})
    exclude = model_cfg.get("exclude_columns", [])
    # Sanitise exclude list too (spaces → underscores, to match sanitised df)
    exclude = [c.replace(" ", "_") for c in exclude]
    present_exclude = [c for c in exclude if c in df.columns]

    if present_exclude:
        logger.info(
            "Excluding %d column(s) flagged as potential leakage: %s",
            len(present_exclude),
            present_exclude,
        )
        df.drop(columns=present_exclude, inplace=True)

    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found. "
            f"Available: {df.columns.tolist()}"
        )

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].copy()
    y = df[target].copy()

    neg, pos = (y == 0).sum(), (y == 1).sum()
    logger.info(
        "Feature matrix: %d rows × %d features | "
        "Target: %d pos (%.1f%%) / %d neg (%.1f%%)",
        X.shape[0], X.shape[1],
        pos, 100 * pos / len(y),
        neg, 100 * neg / len(y),
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
           pd.Series, pd.Series, pd.Series]:
    """Stratified 60 / 20 / 20 train / validation / test split.

    Parameters
    ----------
    X, y : feature matrix and target vector.
    config : parsed config dict.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    data_cfg = config.get("data", {})
    seed     = config.get("random_seed", 42)
    test_sz  = data_cfg.get("test_size", 0.20)
    val_sz   = data_cfg.get("val_size",  0.20)
    stratify = data_cfg.get("stratify",  True)

    strat = y if stratify else None

    # Step 1: hold out test (20 %)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_sz, stratify=strat, random_state=seed
    )

    # Step 2: split remaining 80 % into train (60 %) and val (20 %)
    # val fraction of the trainval portion = val_sz / (1 - test_sz)
    val_frac = val_sz / (1.0 - test_sz)
    strat2 = y_trainval if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac, stratify=strat2, random_state=seed
    )

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(y_train), len(y_val), len(y_test),
    )
    for split_name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        logger.info(
            "  %s positive rate: %.1f%%",
            split_name, 100 * y_split.mean(),
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# 3. Build baseline estimators
# ---------------------------------------------------------------------------

def build_baselines(config: dict) -> dict[str, Any]:
    """Instantiate the three baseline estimators from config hyperparameters.

    Gradient boosting backend:
        - Uses ``xgboost.XGBClassifier`` if XGBoost is installed.
        - Otherwise falls back to ``sklearn.HistGradientBoostingClassifier``.

    Parameters
    ----------
    config : parsed config dict.

    Returns
    -------
    dict mapping model name (str) → unfitted sklearn-compatible estimator.
    """
    seed    = config.get("random_seed", 42)
    m_cfg   = config.get("model", {})
    lr_cfg  = m_cfg.get("logistic_regression", {})
    rf_cfg  = m_cfg.get("random_forest", {})
    gb_cfg  = m_cfg.get("gradient_boosting", {})

    # --- Logistic Regression -------------------------------------------------
    logit = LogisticRegression(
        C=lr_cfg.get("C", 1.0),
        max_iter=lr_cfg.get("max_iter", 1000),
        class_weight=lr_cfg.get("class_weight", "balanced"),
        solver=lr_cfg.get("solver", "lbfgs"),
        random_state=seed,
    )

    # --- Random Forest -------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 300),
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 10),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        n_jobs=rf_cfg.get("n_jobs", -1),
        random_state=seed,
    )

    # --- Gradient Boosting ---------------------------------------------------
    if _XGBOOST_AVAILABLE:
        gb = xgb.XGBClassifier(
            n_estimators=gb_cfg.get("n_estimators", 300),
            learning_rate=gb_cfg.get("learning_rate", 0.05),
            max_depth=gb_cfg.get("max_depth", 6),
            subsample=gb_cfg.get("subsample", 0.8),
            colsample_bytree=gb_cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=gb_cfg.get("scale_pos_weight", 0.35),
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=seed,
        )
        gb_name = "XGBoost"
    else:
        gb = HistGradientBoostingClassifier(
            max_iter=gb_cfg.get("max_iter", 300),
            learning_rate=gb_cfg.get("learning_rate", 0.05),
            max_depth=gb_cfg.get("max_depth", 6),
            min_samples_leaf=gb_cfg.get("min_samples_leaf", 20),
            class_weight=gb_cfg.get("class_weight", "balanced"),
            random_state=seed,
        )
        gb_name = "GradientBoosting"

    models = {
        "LogisticRegression": logit,
        "RandomForest": rf,
        gb_name: gb,
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
    """Run stratified k-fold cross-validation on the training set.

    Parameters
    ----------
    model : unfitted sklearn-compatible estimator.
    X_train, y_train : training data.
    config : parsed config dict.

    Returns
    -------
    dict mapping metric name → array of per-fold scores.
    """
    cv_folds = config.get("model", {}).get("cv_folds", 5)
    seed     = config.get("random_seed", 42)
    cv       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    scoring = {
        "roc_auc":           "roc_auc",
        "average_precision": "average_precision",
        "f1":                "f1",
    }

    logger.info(
        "Cross-validating %s (%d folds)...",
        type(model).__name__, cv_folds,
    )
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv, scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    for metric, scores in cv_results.items():
        if metric.startswith("test_"):
            name = metric[5:]
            logger.info(
                "  CV %s: %.4f ± %.4f",
                name, scores.mean(), scores.std(),
            )

    return cv_results


# ---------------------------------------------------------------------------
# 5. Evaluation on a fixed split
# ---------------------------------------------------------------------------

def evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute a full set of classification metrics on a single split.

    Parameters
    ----------
    model : fitted estimator with ``predict_proba``.
    X, y  : feature matrix and true labels.
    threshold : decision threshold for converting probabilities to class labels.

    Returns
    -------
    dict with keys: roc_auc, pr_auc, accuracy, precision, recall, f1, brier.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc":   roc_auc_score(y, y_prob),
        "pr_auc":    average_precision_score(y, y_prob),
        "accuracy":  accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall":    recall_score(y, y_pred, zero_division=0),
        "f1":        f1_score(y, y_pred, zero_division=0),
        "brier":     brier_score_loss(y, y_prob),
    }
    return metrics


# ---------------------------------------------------------------------------
# 6. Plotting helpers
# ---------------------------------------------------------------------------

def _figures_dir(config: dict) -> Path:
    """Return (and create) the figures output directory.

    Resolves against ``config['_base_dir']`` when present (set by notebook
    callers to anchor relative paths to the project root).
    """
    raw = config.get("paths", {}).get("figures_dir", "reports/figures/")
    base = config.get("_base_dir")
    p = Path(base) / raw if base else Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_roc_curves(
    fitted_models: dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> None:
    """Plot ROC curves for all baseline models on the validation set.

    Parameters
    ----------
    fitted_models : {name: fitted estimator}.
    X_val, y_val  : validation split.
    config        : parsed config dict.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = sns.color_palette("tab10", len(fitted_models))

    for (name, model), color in zip(fitted_models.items(), palette):
        y_prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc = roc_auc_score(y_val, y_prob)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Baseline Models (Validation Set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    out = _figures_dir(config) / "roc_curve_baselines.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve → %s", out)


def plot_pr_curves(
    fitted_models: dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> None:
    """Plot Precision-Recall curves for all baseline models.

    The dashed baseline represents a no-skill classifier (prevalence line).
    PR-AUC is the primary selection metric for imbalanced datasets.

    Parameters
    ----------
    fitted_models : {name: fitted estimator}.
    X_val, y_val  : validation split.
    config        : parsed config dict.
    """
    prevalence = y_val.mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = sns.color_palette("tab10", len(fitted_models))

    for (name, model), color in zip(fitted_models.items(), palette):
        y_prob = model.predict_proba(X_val)[:, 1]
        prec, rec, _ = precision_recall_curve(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)
        ax.plot(rec, prec, label=f"{name}  (PR-AUC={pr_auc:.3f})", color=color, lw=2)

    ax.axhline(prevalence, color="k", linestyle="--", lw=1,
               label=f"No-skill baseline ({prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Baseline Models (Validation Set)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    out = _figures_dir(config) / "pr_curve_baselines.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curve → %s", out)


def plot_confusion_matrix(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    config: dict,
    threshold: float = 0.5,
) -> None:
    """Plot and save a confusion matrix for a single model.

    Parameters
    ----------
    model      : fitted estimator.
    X, y       : evaluation split.
    model_name : used in the plot title and filename.
    config     : parsed config dict.
    threshold  : decision threshold.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Readmitted (0)", "Readmitted (1)"],
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name} (threshold={threshold})")

    out = _figures_dir(config) / "confusion_matrix_best_baseline.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", out)


def plot_calibration_curves(
    fitted_models: dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    n_bins: int = 10,
) -> None:
    """Plot reliability (calibration) curves for all baseline models.

    A well-calibrated model's curve follows the diagonal.  Deviations
    indicate over-confidence (above diagonal) or under-confidence (below).

    Parameters
    ----------
    fitted_models : {name: fitted estimator}.
    X_val, y_val  : validation split.
    config        : parsed config dict.
    n_bins        : number of bins for the calibration curve.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = sns.color_palette("tab10", len(fitted_models))

    for (name, model), color in zip(fitted_models.items(), palette):
        CalibrationDisplay.from_estimator(
            model, X_val, y_val,
            n_bins=n_bins, ax=ax, name=name,
            color=color,
        )

    ax.set_title("Calibration Curves — Baseline Models (Validation Set)")
    ax.legend(loc="upper left", fontsize=9)

    out = _figures_dir(config) / "calibration_baselines.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration curves → %s", out)


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str,
    config: dict,
    top_n: int = 20,
) -> None:
    """Plot feature importances for tree-based models.

    Uses ``feature_importances_`` (Random Forest, XGBoost) or
    ``coef_`` (Logistic Regression).  Silently skips if neither is available.

    Parameters
    ----------
    model        : fitted estimator.
    feature_names: ordered list matching X columns.
    model_name   : used in title and filename.
    config       : parsed config dict.
    top_n        : how many top features to display.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Feature Importances — {model_name}"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = f"|Coefficient| — {model_name}"
    else:
        logger.info("Model %s has no feature importance attribute — skipping plot.", model_name)
        return

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color=sns.color_palette("muted")[0],
    )
    ax.set_title(title)
    ax.set_xlabel("Importance")

    safe_name = model_name.lower().replace(" ", "_")
    out = _figures_dir(config) / f"feature_importance_{safe_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance → %s", out)


# ---------------------------------------------------------------------------
# 7. Model comparison and selection
# ---------------------------------------------------------------------------

def compare_models(eval_results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a metrics summary DataFrame from per-model evaluation dicts.

    Parameters
    ----------
    eval_results : {model_name: {metric_name: value}}.

    Returns
    -------
    pd.DataFrame with models as rows, metrics as columns, sorted by PR-AUC desc.
    """
    df = pd.DataFrame(eval_results).T
    df.index.name = "model"
    # Sort by PR-AUC (primary metric) descending
    if "pr_auc" in df.columns:
        df = df.sort_values("pr_auc", ascending=False)
    return df.round(4)


def select_best_model(
    summary: pd.DataFrame,
    metric: str = "pr_auc",
) -> str:
    """Return the name of the best model by the given metric.

    Parameters
    ----------
    summary : output of ``compare_models``.
    metric  : column to rank by (default: ``pr_auc``).

    Returns
    -------
    str : model name (index value of the best row).
    """
    if metric not in summary.columns:
        logger.warning(
            "Metric '%s' not in summary. Using first column: '%s'.",
            metric, summary.columns[0],
        )
        metric = summary.columns[0]
    best = summary[metric].idxmax()
    logger.info(
        "Best baseline model by %s: %s (%.4f)",
        metric, best, summary.loc[best, metric],
    )
    return best


# ---------------------------------------------------------------------------
# 8. Tuning scaffold (Phase 4)
# ---------------------------------------------------------------------------

def tune_model(
    model: Any,
    param_dist: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Any:
    """RandomizedSearchCV hyperparameter tuning scaffold.

    This function is a placeholder for Phase 4.  It is fully functional
    but uses a small ``n_iter`` by default for speed during prototyping.

    Parameters
    ----------
    model      : unfitted estimator.
    param_dist : parameter distributions for RandomizedSearchCV.
    X_train, y_train : training data.
    config     : parsed config dict.

    Returns
    -------
    Best fitted estimator.
    """
    from sklearn.model_selection import RandomizedSearchCV  # local import

    tuning_cfg = config.get("model", {}).get("tuning", {})
    seed       = config.get("random_seed", 42)
    n_iter     = tuning_cfg.get("n_iter", 20)
    cv_folds   = tuning_cfg.get("cv_folds", 5)
    scoring    = tuning_cfg.get("scoring", "average_precision")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    logger.info(
        "Starting RandomizedSearchCV: n_iter=%d, cv=%d, scoring=%s",
        n_iter, cv_folds, scoring,
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
        verbose=1,
    )
    search.fit(X_train, y_train)

    logger.info(
        "Best params: %s | Best CV %s: %.4f",
        search.best_params_, scoring, search.best_score_,
    )
    return search.best_estimator_


# ---------------------------------------------------------------------------
# 9. Calibration scaffold (Phase 4)
# ---------------------------------------------------------------------------

def calibrate_model(
    model: Any,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    config: dict,
) -> CalibratedClassifierCV:
    """Wrap a fitted estimator with probability calibration.

    Uses ``CalibratedClassifierCV`` in ``prefit`` mode so the original
    model weights are preserved and only the calibration layer is fitted.

    Parameters
    ----------
    model         : already-fitted estimator.
    X_calib, y_calib : calibration data (typically the validation set).
    config        : parsed config dict.

    Returns
    -------
    CalibratedClassifierCV fitted on X_calib.
    """
    cal_cfg = config.get("model", {}).get("calibration", {})
    method  = cal_cfg.get("method", "isotonic")

    logger.info("Calibrating model with method='%s' (prefit mode).", method)
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_calib, y_calib)
    logger.info("Calibration complete.")
    return calibrated

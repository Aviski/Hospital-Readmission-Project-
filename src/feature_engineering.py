"""
feature_engineering.py — Healthcare-focused feature engineering.

Public API
----------
    create_features(df, config)  – build derived features and return augmented DataFrame
"""

from __future__ import annotations

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate engineered features from a cleaned DataFrame.

    The function is designed to be **flexible**: each sub-routine checks
    whether the required source columns are present before attempting to
    create the feature, and logs a warning if they are not.

    Features added (when source data is available):
        - ``comorbidity_count``      – sum of binary comorbidity flags
        - ``age_group``              – age discretised into labelled bins
        - ``prior_admission_flag``   – binary indicator for any prior admission
        - ``discharge_disposition_cat`` – simplified discharge grouping

    Parameters
    ----------
    df:
        Cleaned DataFrame (output of ``clean_data``).
    config:
        Parsed config dict (from ``load_config``).

    Returns
    -------
    pd.DataFrame
        DataFrame with new feature columns appended (original columns retained).
    """
    df = df.copy()
    feat_cfg = config.get("features", {})

    df = _add_comorbidity_count(df, feat_cfg)
    df = _add_age_group(df, feat_cfg)
    df = _add_prior_admission_flag(df, feat_cfg)
    df = _add_discharge_disposition_group(df, feat_cfg)

    logger.info("Feature engineering complete. Shape: %d rows × %d columns",
                df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Sub-routines
# ---------------------------------------------------------------------------

def _add_comorbidity_count(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """Sum binary comorbidity flag columns into a single ``comorbidity_count``.

    Expects ``feat_cfg['comorbidity_cols']`` to contain a list of column names
    that are already binary (0/1).  If the list is empty the function attempts
    to auto-detect columns whose names contain common comorbidity keywords.
    """
    comorbidity_cols: list[str] = feat_cfg.get("comorbidity_cols", [])

    # Auto-detect only if a non-empty list was provided and all columns exist.
    # If comorbidity_cols is empty, skip silently — the dataset may already
    # include a composite score (e.g. Comorbidity_Index) that is more reliable
    # than summing auto-detected columns.
    if not comorbidity_cols:
        logger.info(
            "features.comorbidity_cols is empty — skipping 'comorbidity_count'. "
            "Use Comorbidity_Index / Chronic_Disease_Count directly in modelling."
        )
        return df

    available = [c for c in comorbidity_cols if c in df.columns]
    missing = set(comorbidity_cols) - set(available)
    if missing:
        logger.warning("Comorbidity column(s) not found and will be skipped: %s", missing)
    if not available:
        logger.warning("No comorbidity columns available — skipping 'comorbidity_count'.")
        return df

    df["comorbidity_count"] = df[available].sum(axis=1)
    logger.info("Created 'comorbidity_count' from %d column(s): %s",
                len(available), available)
    return df


def _add_age_group(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """Discretise a numeric age column into labelled bins (``age_group``).

    Configuration keys used:
        - ``features.age_col``     – source column name (default: ``"age"``)
        - ``features.age_bins``    – bin edges (default: [0, 18, 40, 65, 80, 120])
        - ``features.age_labels``  – bin labels (must be len(bins) - 1)
    """
    age_col: str = feat_cfg.get("age_col", "age")

    if age_col not in df.columns:
        logger.warning(
            "Age column '%s' not found — skipping 'age_group'. "
            "Set 'features.age_col' in config.yaml.",
            age_col,
        )
        return df

    bins = feat_cfg.get("age_bins", [0, 18, 40, 65, 80, 120])
    labels = feat_cfg.get("age_labels", ["<18", "18-39", "40-64", "65-79", "80+"])

    df["age_group"] = pd.cut(
        df[age_col],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    logger.info("Created 'age_group' from column '%s'", age_col)
    return df


def _add_prior_admission_flag(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """Create a binary flag indicating whether a patient had any prior admission.

    Configuration key used:
        - ``features.prior_admissions_col`` – source column name (numeric count)

    If the source column contains a numeric count of prior admissions, this
    function adds ``prior_admission_flag`` (1 if count > 0, else 0).
    """
    col: str | None = feat_cfg.get("prior_admissions_col")

    if col is None:
        # Attempt auto-detection
        candidates = [c for c in df.columns if "prior" in c.lower() and "admit" in c.lower()]
        if candidates:
            col = candidates[0]
            logger.info("Auto-detected prior admissions column: '%s'", col)

    if col is None or col not in df.columns:
        logger.warning(
            "Prior admissions column not found — skipping 'prior_admission_flag'. "
            "Set 'features.prior_admissions_col' in config.yaml."
        )
        return df

    df["prior_admission_flag"] = (df[col] > 0).astype(int)
    logger.info("Created 'prior_admission_flag' from column '%s'", col)
    return df


def _add_discharge_disposition_group(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """Map granular discharge disposition codes/labels into broader groups.

    Configuration key used:
        - ``features.discharge_disposition_col`` – source column name

    The mapping below covers typical CMS discharge disposition categories.
    Adjust the ``DISPOSITION_MAP`` dict to match the actual dataset values.
    """
    col: str | None = feat_cfg.get("discharge_disposition_col")

    if col is None:
        # Attempt auto-detection
        candidates = [c for c in df.columns if "discharge" in c.lower() or "disposition" in c.lower()]
        if candidates:
            col = candidates[0]
            logger.info("Auto-detected discharge disposition column: '%s'", col)

    if col is None or col not in df.columns:
        logger.warning(
            "Discharge disposition column not found — skipping grouping. "
            "Set 'features.discharge_disposition_col' in config.yaml."
        )
        return df

    # Mapping covers the current dataset values and common CMS equivalents.
    # Add entries here as new datasets introduce different label conventions.
    DISPOSITION_MAP: dict[str, str] = {
        # Home
        "home": "Home",
        "home health": "Home",
        "home with home health service": "Home",
        # Facility (post-acute / institutional care)
        "rehab": "Facility",
        "inpatient rehabilitation": "Facility",
        "nursing facility": "Facility",
        "skilled nursing facility": "Facility",
        "snf": "Facility",
        "long term care hospital": "Facility",
        "ltch": "Facility",
        # AMA / Other
        "ama": "AMA/Other",
        "left against medical advice": "AMA/Other",
        "expired": "AMA/Other",
        "hospice": "AMA/Other",
    }

    raw_values = df[col].astype(str).str.strip().str.lower()
    df["discharge_disposition_cat"] = raw_values.map(DISPOSITION_MAP).fillna("Other")

    logger.info(
        "Created 'discharge_disposition_cat' from column '%s'. "
        "Group counts:\n%s",
        col,
        df["discharge_disposition_cat"].value_counts().to_string(),
    )
    return df

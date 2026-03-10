"""
data_preparation.py — Loading, cleaning, and basic validation of the raw dataset.

Public API
----------
    load_raw_data(path)          – read a CSV into a DataFrame
    clean_data(df, config)       – impute, clip, and validate (no encoding)
    encode_features(df, config)  – one-hot encode categorical columns

Pipeline order
--------------
    df_raw      = load_raw_data(config['paths']['raw_data'])
    df_clean    = clean_data(df_raw, config)
    df_features = create_features(df_clean, config)   # src.feature_engineering
    df_encoded  = encode_features(df_features, config)

Keeping encoding as a separate final step ensures that feature engineering
functions always have access to the original categorical columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read a raw CSV file into a pandas DataFrame.

    Parameters
    ----------
    path:
        Path to the CSV file (raw data directory).

    Returns
    -------
    pd.DataFrame
        Loaded dataset with original dtypes inferred by pandas.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path.resolve()}")

    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Cleaning  (no one-hot encoding — that belongs in encode_features)
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean a raw DataFrame according to the project configuration.

    Steps performed:
        1. Drop columns listed in ``config['data']['drop_columns']``.
        2. Remove fully-duplicate rows.
        3. Validate that the target column is present and binary.
        4. Clip physically-impossible numeric values (``config['data']['clip_values']``).
        5. Impute missing numeric values with the column median.
        6. Impute missing categorical values with ``config['data']['categorical_impute_value']``.

    One-hot encoding is intentionally NOT performed here so that
    ``create_features`` (feature engineering) has access to the original
    categorical columns.  Call ``encode_features(df, config)`` after
    feature engineering is complete.

    Parameters
    ----------
    df:
        Raw DataFrame (not modified in place — a copy is returned).
    config:
        Parsed config dict (from ``load_config``).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()
    data_cfg = config.get("data", {})

    # ------------------------------------------------------------------
    # 1. Drop unwanted columns
    # ------------------------------------------------------------------
    drop_cols = [c for c in data_cfg.get("drop_columns", []) if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        logger.info("Dropped columns: %s", drop_cols)

    # ------------------------------------------------------------------
    # 2. Remove duplicate rows
    # ------------------------------------------------------------------
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Removed %d duplicate rows", n_dropped)

    # ------------------------------------------------------------------
    # 3. Basic validation
    # ------------------------------------------------------------------
    target = data_cfg.get("target_column", "readmitted_30d")
    _validate(df, target)

    # ------------------------------------------------------------------
    # 4. Clip physically-impossible values
    # ------------------------------------------------------------------
    clip_cfg: dict = data_cfg.get("clip_values", {})
    for col, bounds in clip_cfg.items():
        if col not in df.columns:
            logger.warning("clip_values: column '%s' not found — skipping.", col)
            continue
        lo, hi = bounds[0], bounds[1]
        n_clipped = (
            (lo is not None and df[col] < lo) | (hi is not None and df[col] > hi)
        ).sum()
        if n_clipped:
            df[col] = df[col].clip(lower=lo, upper=hi)
            logger.info("Clipped %d out-of-range values in '%s' to [%s, %s]",
                        n_clipped, col, lo, hi)

    # ------------------------------------------------------------------
    # 5. Impute numeric columns with median
    # ------------------------------------------------------------------
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != target
    ]
    missing_numeric = {c: df[c].isna().sum() for c in numeric_cols if df[c].isna().any()}
    if missing_numeric:
        logger.info("Imputing %d numeric column(s) with median: %s",
                    len(missing_numeric), list(missing_numeric.keys()))
        for col in missing_numeric:
            df[col].fillna(df[col].median(), inplace=True)
    else:
        logger.info("No missing numeric values — imputation skipped.")

    # ------------------------------------------------------------------
    # 6. Impute categorical columns with a placeholder string
    # ------------------------------------------------------------------
    cat_fill = data_cfg.get("categorical_impute_value", "Unknown")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing_cat = {c: df[c].isna().sum() for c in cat_cols if df[c].isna().any()}
    if missing_cat:
        logger.info("Imputing %d categorical column(s) with '%s': %s",
                    len(missing_cat), cat_fill, list(missing_cat.keys()))
        for col in missing_cat:
            df[col].fillna(cat_fill, inplace=True)
    else:
        logger.info("No missing categorical values — imputation skipped.")

    logger.info("Cleaning complete. Shape: %d rows × %d columns", df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Encoding  (run AFTER feature engineering)
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """One-hot encode categorical columns.

    This step is intentionally separated from ``clean_data`` so that
    ``create_features`` always operates on data that still contains the
    original categorical columns (e.g. Discharge_Disposition).

    Parameters
    ----------
    df:
        DataFrame after cleaning *and* feature engineering.
    config:
        Parsed config dict.

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical columns replaced by OHE dummy columns.
        Boolean dummy columns are cast to ``int8`` for compatibility with
        scikit-learn and serialisation formats.
    """
    data_cfg = config.get("data", {})
    target = data_cfg.get("target_column", "readmitted_30d")

    # First pass: explicitly configured columns (if any)
    explicit_cols: list[str] = data_cfg.get("categorical_columns", [])
    explicit_cols = [c for c in explicit_cols if c in df.columns]

    if explicit_cols:
        logger.info("One-hot encoding configured column(s): %s", explicit_cols)
        df = pd.get_dummies(df, columns=explicit_cols, drop_first=False)

    # Second pass: any remaining object/category columns not yet encoded
    # (e.g. discharge_disposition_cat added by feature engineering)
    remaining_cat = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target
    ]
    if remaining_cat:
        logger.info("One-hot encoding remaining categorical column(s): %s", remaining_cat)
        df = pd.get_dummies(df, columns=remaining_cat, drop_first=False)

    # Cast bool dummies to int8 for sklearn compatibility
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype("int8")

    logger.info("Encoding complete. Shape: %d rows × %d columns", df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, target: str) -> None:
    """Run basic sanity checks and log warnings for any issues found."""
    if target not in df.columns:
        logger.warning(
            "Target column '%s' not found in dataset. "
            "Available columns: %s",
            target,
            df.columns.tolist(),
        )
        return

    # Target should be binary (0 / 1)
    unique_vals = df[target].dropna().unique()
    if set(map(int, unique_vals)) - {0, 1}:
        logger.warning(
            "Target column '%s' contains unexpected values: %s. "
            "Expected binary 0/1.",
            target,
            unique_vals,
        )

    rate = df[target].mean()
    logger.info(
        "Readmission rate: %.1f%%  (%d positive / %d total)",
        rate * 100, int(df[target].sum()), len(df),
    )

    total_missing = df.isna().sum().sum()
    if total_missing:
        logger.info("Total missing values before imputation: %d", total_missing)

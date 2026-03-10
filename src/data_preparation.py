"""
data_preparation.py — Loading, cleaning, and basic validation of the raw dataset.

Public API
----------
    load_raw_data(path)          – read a CSV into a DataFrame
    clean_data(df, config)       – impute, validate, and optionally encode
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean a raw DataFrame according to the project configuration.

    Steps performed:
        1. Drop columns listed in ``config['data']['drop_columns']``.
        2. Remove fully-duplicate rows.
        3. Validate that the target column is present and binary.
        4. Impute missing numeric values with the column median.
        5. Impute missing categorical values with ``config['data']['categorical_impute_value']``.
        6. Optionally one-hot encode categorical columns.

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
    # 4. Impute numeric columns with median
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Exclude the target from imputation
    numeric_cols = [c for c in numeric_cols if c != target]

    missing_numeric = {c: df[c].isna().sum() for c in numeric_cols if df[c].isna().any()}
    if missing_numeric:
        logger.info("Imputing %d numeric column(s) with median: %s",
                    len(missing_numeric), list(missing_numeric.keys()))
        for col in missing_numeric:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # ------------------------------------------------------------------
    # 5. Impute categorical columns with a placeholder string
    # ------------------------------------------------------------------
    cat_fill = data_cfg.get("categorical_impute_value", "Unknown")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    missing_cat = {c: df[c].isna().sum() for c in cat_cols if df[c].isna().any()}
    if missing_cat:
        logger.info("Imputing %d categorical column(s) with '%s': %s",
                    len(missing_cat), cat_fill, list(missing_cat.keys()))
        for col in missing_cat:
            df[col].fillna(cat_fill, inplace=True)

    # ------------------------------------------------------------------
    # 6. Optional one-hot encoding
    # ------------------------------------------------------------------
    encode_cols: list[str] = data_cfg.get("categorical_columns", [])
    if not encode_cols:
        # Auto-detect: all remaining object/category columns except target
        encode_cols = [c for c in cat_cols if c != target]

    if encode_cols:
        logger.info("One-hot encoding %d column(s): %s", len(encode_cols), encode_cols)
        df = pd.get_dummies(df, columns=encode_cols, drop_first=False)

    logger.info("Cleaning complete. Shape: %d rows × %d columns", df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, target: str) -> None:
    """Run basic sanity checks and log warnings for any issues found."""
    # Target column presence
    if target not in df.columns:
        logger.warning(
            "Target column '%s' not found in dataset. "
            "Available columns: %s",
            target,
            df.columns.tolist(),
        )
        return

    # Target should be binary (0 / 1 or True / False)
    unique_vals = df[target].dropna().unique()
    if set(map(int, unique_vals)) - {0, 1}:
        logger.warning(
            "Target column '%s' contains unexpected values: %s. "
            "Expected binary 0/1.",
            target,
            unique_vals,
        )

    # Class balance
    rate = df[target].mean()
    logger.info("Readmission rate: %.1f%%  (%d positive / %d total)",
                rate * 100, df[target].sum(), len(df))

    # Overall missing-value summary
    total_missing = df.isna().sum().sum()
    if total_missing:
        logger.info("Total missing values before imputation: %d", total_missing)

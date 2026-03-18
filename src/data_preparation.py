"""
data_preparation.py — Loading, cleaning, and basic validation of the raw dataset.

Public API
----------
    load_raw_data(path)          – read a CSV into a DataFrame
    clean_data(df, config)       – map target, impute, validate
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

import json
from pathlib import Path

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def _normalize_target_label(value: object) -> str | None:
    """Normalize a raw target label for strict validation and mapping."""
    if pd.isna(value):
        return None
    return str(value).strip().lower()


def _load_expected_feature_columns(config: dict) -> list[str] | None:
    """Load the persisted training feature schema from metadata when available."""
    metadata = _load_feature_metadata(config)
    if metadata is None:
        return None

    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(
        isinstance(col, str) for col in feature_columns
    ):
        return None
    return feature_columns


def _load_feature_metadata(config: dict) -> dict | None:
    """Load persisted feature metadata when available."""
    metadata_rel = config.get("paths", {}).get("features_metadata")
    if not metadata_rel:
        return None

    base_dir = config.get("_base_dir")
    metadata_path = Path(base_dir) / metadata_rel if base_dir else Path(metadata_rel)
    if not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Could not read feature metadata at %s: %s", metadata_path, exc)
        return None

    if not isinstance(metadata, dict):
        return None
    return metadata


def _load_expected_categorical_levels(config: dict) -> dict[str, set[str]] | None:
    """Load persisted training category levels keyed by column name."""
    metadata = _load_feature_metadata(config)
    if metadata is None:
        return None

    raw_levels = metadata.get("categorical_levels")
    if not isinstance(raw_levels, dict):
        return None

    levels: dict[str, set[str]] = {}
    for column, values in raw_levels.items():
        if not isinstance(column, str) or not isinstance(values, list):
            continue
        valid_values = {str(value) for value in values if isinstance(value, str)}
        if valid_values:
            levels[column] = valid_values
    return levels or None


def _warn_on_unseen_categories(
    df: pd.DataFrame,
    expected_levels: dict[str, set[str]] | None,
    target: str,
) -> None:
    """Warn when inference data contains categorical levels absent from training metadata."""
    if not expected_levels:
        return

    categorical_columns = [
        c for c in df.select_dtypes(include=["object", "category", "string"]).columns
        if c != target and c in expected_levels
    ]
    for column in categorical_columns:
        observed_values = {
            str(value)
            for value in pd.Series(df[column]).dropna().astype(str).unique().tolist()
        }
        unseen_values = sorted(value for value in observed_values if value not in expected_levels[column])
        if unseen_values:
            logger.warning(
                "Unseen categorical value(s) in '%s': %s",
                column,
                unseen_values,
            )


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
        0. Map target column from string to binary (yes→1, no→0).
        1. Drop columns listed in ``config['data']['drop_columns']``.
        2. Remove fully-duplicate rows.
        3. Validate schema: required columns present, target is 0/1.
        4. Impute missing numeric values with the column median.
        5. Impute missing categorical values (object/category dtype) with
           the column mode.  NOTE: the string "Missing" in medical_specialty
           is a valid category — it is NOT treated as NaN.
        6. Post-imputation validation: raise if any missing values remain.

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
    target = data_cfg.get("target_column", "readmitted")

    # ------------------------------------------------------------------
    # 0. Map target from string to binary
    # ------------------------------------------------------------------
    if target in df.columns:
        if pd.api.types.is_numeric_dtype(df[target]):
            missing_count = int(df[target].isna().sum())
            if missing_count:
                raise ValueError(
                    f"Target '{target}' contains {missing_count} missing value(s)."
                )
            unique_vals = set(pd.Series(df[target]).dropna().unique())
            if not unique_vals.issubset({0, 1}):
                raise ValueError(
                    f"Target '{target}' must contain only 0/1 when numeric. "
                    f"Found: {sorted(unique_vals)}"
                )
            df[target] = df[target].astype("int8")
        else:
            pos_val = _normalize_target_label(
                data_cfg.get("target_positive_value", "yes")
            )
            neg_val = _normalize_target_label(
                data_cfg.get("target_negative_value", "no")
            )
            if not pos_val or not neg_val or pos_val == neg_val:
                raise ValueError(
                    "Target mapping config must define distinct non-empty "
                    "positive and negative values."
                )

            normalized_target = df[target].map(_normalize_target_label)
            missing_mask = normalized_target.isna()
            if missing_mask.any():
                raise ValueError(
                    f"Target '{target}' contains {int(missing_mask.sum())} missing value(s)."
                )

            allowed_values = {pos_val, neg_val}
            invalid_mask = ~normalized_target.isin(allowed_values)
            if invalid_mask.any():
                invalid_values = sorted(
                    df.loc[invalid_mask, target].astype(str).drop_duplicates().tolist()
                )
                preview = invalid_values[:5]
                suffix = " ..." if len(invalid_values) > 5 else ""
                raise ValueError(
                    f"Unexpected target label(s) in '{target}': {preview}{suffix}. "
                    f"Expected only '{pos_val}' or '{neg_val}' (case/whitespace-insensitive)."
                )

            df[target] = normalized_target.map({neg_val: 0, pos_val: 1}).astype("int8")
            logger.info(
                "Mapped target '%s': '%s'→1, '%s'→0",
                target,
                pos_val,
                neg_val,
            )

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
    # 3. Schema validation
    # ------------------------------------------------------------------
    _validate(df, target, config)

    # ------------------------------------------------------------------
    # 4. Impute numeric columns with median
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
            df[col] = df[col].fillna(df[col].median())
    else:
        logger.info("No missing numeric values — imputation skipped.")

    # ------------------------------------------------------------------
    # 5. Impute categorical columns with mode (never imputes "Missing")
    # ------------------------------------------------------------------
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target
    ]
    missing_cat = {c: df[c].isna().sum() for c in cat_cols if df[c].isna().any()}
    if missing_cat:
        logger.info("Imputing %d categorical column(s) with mode: %s",
                    len(missing_cat), list(missing_cat.keys()))
        for col in missing_cat:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
    else:
        logger.info("No missing categorical values — imputation skipped.")

    # ------------------------------------------------------------------
    # 6. Post-imputation validation
    # ------------------------------------------------------------------
    still_missing = df.isna().sum().sum()
    if still_missing > 0:
        raise ValueError(
            f"Imputation incomplete — {still_missing} missing values remain."
        )

    remaining_obj = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if remaining_obj:
        logger.info("Categorical columns to be OHE'd: %s", remaining_obj)

    logger.info("Cleaning complete. Shape: %d rows × %d columns", df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Encoding  (run AFTER feature engineering)
# ---------------------------------------------------------------------------

def encode_features(
    df: pd.DataFrame,
    config: dict,
    expected_feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """One-hot encode categorical columns.

    This step is intentionally separated from ``clean_data`` so that
    ``create_features`` always operates on data that still contains the
    original categorical columns.

    Uses ``drop_first=True`` on both passes to remove reference categories
    and avoid perfect multicollinearity in linear models.

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
        scikit-learn and serialisation formats. When an expected training
        feature schema is supplied (or discoverable from metadata for
        target-less scoring data), columns are deterministically reindexed
        to that schema with missing dummies filled as 0.
    """
    df = df.copy()
    data_cfg = config.get("data", {})
    target = data_cfg.get("target_column", "readmitted")
    original_has_target = target in df.columns
    expected_category_levels = _load_expected_categorical_levels(config)

    _warn_on_unseen_categories(df, expected_category_levels, target)

    # First pass: explicitly configured columns (if any)
    explicit_cols: list[str] = data_cfg.get("categorical_columns", [])
    explicit_cols = [c for c in explicit_cols if c in df.columns]

    if explicit_cols:
        logger.info("One-hot encoding configured column(s): %s", explicit_cols)
        df = pd.get_dummies(df, columns=explicit_cols, drop_first=True)

    # Second pass: any remaining object/category columns not yet encoded
    # (e.g. columns created by feature engineering)
    remaining_cat = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target
    ]
    if remaining_cat:
        logger.info("One-hot encoding remaining categorical column(s): %s", remaining_cat)
        df = pd.get_dummies(df, columns=remaining_cat, drop_first=True)

    # Cast bool dummies to int8 for sklearn compatibility
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype("int8")

    # Post-encoding validation
    remaining_after = [
        c for c in df.select_dtypes(["object", "category"]).columns
        if c != target
    ]
    if remaining_after:
        raise ValueError(
            f"Encoding incomplete — categorical columns remain: {remaining_after}"
        )

    if expected_feature_columns is None and not original_has_target:
        expected_feature_columns = _load_expected_feature_columns(config)

    if expected_feature_columns:
        target_series = df[target].copy() if original_has_target else None
        encoded_feature_columns = [c for c in df.columns if c != target]
        missing_cols = [c for c in expected_feature_columns if c not in encoded_feature_columns]
        extra_cols = [c for c in encoded_feature_columns if c not in expected_feature_columns]

        df = df.reindex(columns=expected_feature_columns, fill_value=0)
        if target_series is not None:
            df[target] = target_series

        if extra_cols:
            logger.warning(
                "Aligned encoded features to expected schema and dropped %d unexpected column(s); added=%d",
                len(extra_cols),
                len(missing_cols),
            )
        elif missing_cols:
            logger.info(
                "Aligned encoded features to expected schema: added=%d dropped=0",
                len(missing_cols),
            )

    logger.info("Encoding complete. Shape: %d rows × %d columns", df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, target: str, config: dict) -> None:
    """Validate schema and raise on critical failures."""
    data_cfg = config.get("data", {})
    required = data_cfg.get("required_columns", [])

    # Schema check — raise on missing required columns
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Schema mismatch — required columns not found: {missing_cols}"
        )

    # Target presence
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Target must be binary 0/1 after mapping
    unique_vals = set(df[target].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"Target '{target}' must contain only 0/1 after mapping. "
            f"Found: {unique_vals}"
        )

    # Duplicate warning
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        logger.warning("%d duplicate rows found.", n_dups)

    # Class balance info
    rate = df[target].mean()
    logger.info(
        "Class balance — positive rate: %.1f%%  (%d / %d)",
        rate * 100, int(df[target].sum()), len(df),
    )

    total_missing = df.isna().sum().sum()
    if total_missing:
        logger.info("Total missing values before imputation: %d", total_missing)

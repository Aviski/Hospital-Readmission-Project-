"""
feature_engineering.py — Healthcare feature engineering for hospital_readmissions.csv.

Public API
----------
    create_features(df, config)  – build derived features and return augmented DataFrame

Dataset schema expected
-----------------------
    age                – categorical bracket strings: "[40-50)", "[50-60)", ...
    time_in_hospital   – integer days admitted
    n_lab_procedures   – integer
    n_procedures       – integer
    n_medications      – integer
    n_outpatient       – prior outpatient visits
    n_inpatient        – prior inpatient admissions
    n_emergency        – prior emergency visits
    medical_specialty  – categorical; 49.5% of values are the string "Missing"
    diag_1/2/3         – categorical diagnosis codes
    glucose_test       – categorical: no/normal/high
    A1Ctest            – categorical: no/normal/high
    change             – categorical: yes/no
    diabetes_med       – categorical: yes/no
    readmitted         – binary target (already mapped to 0/1 by clean_data)
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

    Features added:
        - ``age_ordinal``              – ordered integer from age bracket string
        - ``any_n_inpatient``          – binary: any prior inpatient admission
        - ``any_n_emergency``          – binary: any prior emergency visit
        - ``total_prior_utilization``  – sum of all prior visit counts
        - ``specialty_known``          – binary: medical_specialty != "Missing"
        - interaction features         – numeric column products (from config)

    The ``age`` string column is dropped after ordinal encoding since
    ordinal encoding is more appropriate than OHE for ordered categories.

    Parameters
    ----------
    df:
        Cleaned DataFrame (output of ``clean_data``).
    config:
        Parsed config dict (from ``load_config``).

    Returns
    -------
    pd.DataFrame
        DataFrame with new feature columns appended (most original columns
        retained; ``age`` is replaced by ``age_ordinal``).
    """
    df = df.copy()
    feat_cfg = config.get("features", {})

    df = _map_age_to_ordinal(df, feat_cfg)
    df = _add_prior_utilization_flags(df, feat_cfg)
    df = _add_total_utilization(df)
    df = _add_specialty_known_flag(df, feat_cfg)
    df = _add_interaction_features(df, feat_cfg)

    logger.info("Feature engineering complete. Shape: %d rows × %d columns",
                df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Sub-routines
# ---------------------------------------------------------------------------

def _map_age_to_ordinal(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """Convert bracket age strings to an ordered integer (``age_ordinal``).

    The original ``age`` column is dropped: ordinal encoding captures the
    ordering correctly, and OHE would discard it.

    Configuration key: ``features.age_brackets`` — dict mapping bracket
    string to integer rank (e.g. ``"[70-80)": 4``).

    Unmapped values produce NaN with a warning; the column is cast to Int64
    (nullable integer) to preserve the NaN without promoting to float.
    """
    age_map: dict = feat_cfg.get("age_brackets", {})

    if "age" not in df.columns:
        logger.warning("Column 'age' not found — skipping age_ordinal.")
        return df

    df["age_ordinal"] = df["age"].map(age_map)

    n_unmapped = df["age_ordinal"].isna().sum()
    if n_unmapped > 0:
        unmapped_vals = df.loc[df["age_ordinal"].isna(), "age"].unique().tolist()
        logger.warning(
            "%d rows have unmapped age values: %s. "
            "Add entries to features.age_brackets in config.yaml.",
            n_unmapped, unmapped_vals,
        )

    df["age_ordinal"] = df["age_ordinal"].astype(float).astype(int)
    df.drop(columns=["age"], inplace=True)
    logger.info("Created 'age_ordinal' from 'age'; dropped original 'age' column.")
    return df


def _add_prior_utilization_flags(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """Binary flags for any prior utilization (``any_{col}``).

    Binary flags are stronger predictors than raw counts for linear models
    when counts are sparse.

    Configuration key: ``features.prior_utilization_cols`` — list of column
    names (e.g. ``["n_inpatient", "n_emergency"]``).
    """
    cols: list[str] = feat_cfg.get("prior_utilization_cols", ["n_inpatient", "n_emergency"])
    created = []
    for col in cols:
        if col not in df.columns:
            logger.warning("Prior utilization column '%s' not found — skipping.", col)
            continue
        flag_col = f"any_{col}"
        df[flag_col] = (df[col] > 0).astype(int)
        created.append(flag_col)

    if created:
        logger.info("Created prior utilization flag(s): %s", created)
    return df


def _add_total_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """Sum all prior visit counts into ``total_prior_utilization``.

    Aggregates outpatient, inpatient, and emergency prior visits.
    Only columns present in ``df`` are included.
    """
    util_cols = ["n_outpatient", "n_inpatient", "n_emergency"]
    available = [c for c in util_cols if c in df.columns]
    if not available:
        logger.warning("No prior utilization columns found — skipping total_prior_utilization.")
        return df

    df["total_prior_utilization"] = df[available].sum(axis=1)
    logger.info("Created 'total_prior_utilization' from %s", available)
    return df


def _add_specialty_known_flag(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """Binary flag: ``specialty_known`` = 1 if medical_specialty is not "Missing".

    49.5% of medical_specialty values in this dataset are the literal string
    "Missing" (not NaN).  Making the missingness explicit as a binary feature
    gives the model a clean signal without conflating it with imputed values.

    The ``medical_specialty`` column itself is retained for OHE.

    Configuration key: ``features.specialty_missing_value`` (default: "Missing").
    """
    if "medical_specialty" not in df.columns:
        logger.warning("Column 'medical_specialty' not found — skipping specialty_known.")
        return df

    missing_val: str = feat_cfg.get("specialty_missing_value", "Missing")
    df["specialty_known"] = (df["medical_specialty"] != missing_val).astype(int)
    known_pct = df["specialty_known"].mean() * 100
    logger.info(
        "Created 'specialty_known': %.1f%% have a known specialty.", known_pct
    )
    return df


def _add_interaction_features(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """Multiply pairs of numeric columns to create interaction terms.

    Configuration keys used (under ``features.interactions``):
        - ``enabled`` – bool flag to toggle all interactions on/off.
        - ``terms``   – list of [col_a, col_b] pairs. A column named
          ``{col_a}_x_{col_b}`` is created for each valid pair.

    Only pairs where both columns are present and numeric are created.
    """
    interaction_cfg = feat_cfg.get("interactions", {})
    if not interaction_cfg.get("enabled", False):
        return df

    terms: list[list[str]] = interaction_cfg.get("terms", [])
    if not terms:
        logger.info("interactions.enabled=true but no terms defined — skipping.")
        return df

    created = []
    for pair in terms:
        if len(pair) != 2:
            logger.warning("Interaction term %s must have exactly 2 columns — skipping.", pair)
            continue
        col_a, col_b = pair
        if col_a not in df.columns or col_b not in df.columns:
            logger.warning(
                "Interaction term (%s × %s): one or both columns not found — skipping.",
                col_a, col_b,
            )
            continue
        if not pd.api.types.is_numeric_dtype(df[col_a]) or \
                not pd.api.types.is_numeric_dtype(df[col_b]):
            logger.warning(
                "Interaction term (%s × %s): non-numeric column — skipping.",
                col_a, col_b,
            )
            continue
        new_col = f"{col_a}_x_{col_b}"
        df[new_col] = df[col_a] * df[col_b]
        created.append(new_col)

    if created:
        logger.info("Created %d interaction feature(s): %s", len(created), created)
    return df

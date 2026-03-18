"""
cleaning_validation.py - Validate that preprocessing produced sane outputs.

Usage
-----
    python -m src.cleaning_validation

The validator runs the project's preprocessing stages in order:
    load_raw_data -> clean_data -> create_features -> encode_features

It then performs explicit checks and exits non-zero if any required check fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

from src.data_preparation import load_raw_data
from src.pipeline import build_feature_artifacts
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    severity: str = "error"


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _check_target(df: pd.DataFrame, target: str) -> CheckResult:
    if target not in df.columns:
        return CheckResult("target_present", False, f"Missing target column '{target}'")

    target_series = pd.Series(df[target]).dropna()
    numeric_target = pd.to_numeric(target_series, errors="coerce")

    invalid_mask = numeric_target.isna()
    if invalid_mask.any():
        invalid_values = sorted(target_series[invalid_mask].astype(str).unique().tolist())
        return CheckResult(
            "target_binary",
            False,
            f"Target contains non-numeric values after cleaning: {invalid_values}",
        )

    non_integer_mask = numeric_target.mod(1).ne(0)
    if non_integer_mask.any():
        non_integer_values = sorted(numeric_target[non_integer_mask].unique().tolist())
        return CheckResult(
            "target_binary",
            False,
            f"Target contains non-integer numeric values: {non_integer_values}",
        )

    unique_vals = set(numeric_target.astype(int).unique().tolist())
    if not unique_vals.issubset({0, 1}):
        return CheckResult(
            "target_binary",
            False,
            f"Target contains non-binary values: {sorted(unique_vals)}",
        )

    rate = float(df[target].mean())
    return CheckResult("target_binary", True, f"Target present and binary; positive rate={rate:.3f}")


def _check_duplicates(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> CheckResult:
    raw_dupes = int(raw_df.duplicated().sum())
    clean_dupes = int(clean_df.duplicated().sum())
    if clean_dupes != 0:
        return CheckResult(
            "duplicates_removed",
            False,
            f"Cleaned data still contains {clean_dupes} duplicate rows",
        )
    return CheckResult(
        "duplicates_removed",
        True,
        f"Raw duplicates={raw_dupes}; cleaned duplicates={clean_dupes}",
    )


def _check_missing_after_cleaning(clean_df: pd.DataFrame, target: str) -> list[CheckResult]:
    results: list[CheckResult] = []

    numeric_cols = [c for c in clean_df.select_dtypes(include="number").columns if c != target]
    cat_cols = clean_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    missing_numeric = int(clean_df[numeric_cols].isna().sum().sum()) if numeric_cols else 0
    missing_cat = int(clean_df[cat_cols].isna().sum().sum()) if cat_cols else 0

    results.append(
        CheckResult(
            "missing_numeric_after_cleaning",
            missing_numeric == 0,
            f"Remaining numeric missing values={missing_numeric}",
        )
    )
    results.append(
        CheckResult(
            "missing_categorical_after_cleaning",
            missing_cat == 0,
            f"Remaining categorical missing values={missing_cat}",
        )
    )
    return results


def _check_feature_generation(df_features: pd.DataFrame) -> list[CheckResult]:
    optional_expected = [
        "age_ordinal",
        "any_n_inpatient",
        "any_n_emergency",
        "total_prior_utilization",
        "specialty_known",
        "n_inpatient_x_time_in_hospital",
        "n_medications_x_time_in_hospital",
    ]
    results: list[CheckResult] = []

    for col in optional_expected:
        results.append(
            CheckResult(
                f"feature_{col}",
                col in df_features.columns,
                "present" if col in df_features.columns else "not created",
                severity="warning",
            )
        )

    return results


def _check_encoding(df_encoded: pd.DataFrame, config: dict, target: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    configured_cats = config.get("data", {}).get("categorical_columns", [])

    remaining_objects = [
        c for c in df_encoded.select_dtypes(include=["object", "category", "string"]).columns
        if c != target
    ]
    results.append(
        CheckResult(
            "remaining_categorical_after_encoding",
            len(remaining_objects) == 0,
            f"Remaining categorical columns={remaining_objects}",
        )
    )

    missing_dummy_groups = []
    for col in configured_cats:
        if col in df_encoded.columns:
            missing_dummy_groups.append(col)
            continue
        prefix = f"{col}_"
        if not any(name.startswith(prefix) for name in df_encoded.columns):
            missing_dummy_groups.append(col)

    results.append(
        CheckResult(
            "configured_categoricals_encoded",
            len(missing_dummy_groups) == 0,
            f"Missing encoded groups={missing_dummy_groups}",
        )
    )
    return results


def _check_artifact_row_counts(
    df_analysis: pd.DataFrame,
    df_encoded: pd.DataFrame,
) -> CheckResult:
    return CheckResult(
        "analysis_encoded_row_count_match",
        len(df_analysis) == len(df_encoded),
        f"analysis_rows={len(df_analysis)}, encoded_rows={len(df_encoded)}",
    )


def _check_artifact_index_alignment(
    df_analysis: pd.DataFrame,
    df_encoded: pd.DataFrame,
) -> CheckResult:
    aligned = df_analysis.index.equals(df_encoded.index)
    detail = (
        "row_id index alignment preserved"
        if aligned
        else "row_id index alignment mismatch between analysis and encoded artifacts"
    )
    return CheckResult("analysis_encoded_index_match", aligned, detail)


def run_validation(config_path: str | Path | None = None) -> tuple[list[CheckResult], dict[str, pd.DataFrame]]:
    root = _resolve_project_root()
    cfg_path = Path(config_path) if config_path else root / "config" / "config.yaml"
    config = load_config(cfg_path)
    config["_base_dir"] = str(root)
    set_seed(int(config.get("random_seed", 42)))

    raw_path = root / config["paths"]["raw_data"]
    target = config["data"]["target_column"]

    raw_df = load_raw_data(raw_path)
    artifacts = build_feature_artifacts(config, base_dir=root)
    clean_df = artifacts.clean
    feat_df = artifacts.analysis
    encoded_df = artifacts.encoded

    results: list[CheckResult] = []
    results.append(_check_target(clean_df, target))
    results.append(_check_duplicates(raw_df, clean_df))
    results.append(_check_artifact_row_counts(feat_df, encoded_df))
    results.append(_check_artifact_index_alignment(feat_df, encoded_df))
    results.extend(_check_missing_after_cleaning(clean_df, target))
    results.extend(_check_feature_generation(feat_df))
    results.extend(_check_encoding(encoded_df, config, target))

    return results, {
        "raw": raw_df,
        "clean": clean_df,
        "features": feat_df,
        "encoded": encoded_df,
    }


def _print_summary(results: list[CheckResult], frames: dict[str, pd.DataFrame]) -> None:
    print("Cleaning Validation Summary")
    print("=" * 80)
    print(f"raw shape      : {frames['raw'].shape}")
    print(f"clean shape    : {frames['clean'].shape}")
    print(f"analysis shape : {frames['features'].shape}")
    print(f"encoded shape  : {frames['encoded'].shape}")
    print("")

    for result in results:
        status = "PASS" if result.passed else ("WARN" if result.severity == "warning" else "FAIL")
        print(f"[{status}] {result.name}: {result.detail}")

    required_failures = [r for r in results if not r.passed and r.severity != "warning"]
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    print("")
    print(f"required failures : {len(required_failures)}")
    print(f"warnings          : {len(warnings)}")


def main() -> int:
    try:
        results, frames = run_validation()
    except Exception as exc:
        print("Cleaning Validation Summary")
        print("=" * 80)
        print(f"[FAIL] preprocessing_pipeline: {exc}")
        return 1

    _print_summary(results, frames)
    required_failures = [r for r in results if not r.passed and r.severity != "warning"]
    return 1 if required_failures else 0


if __name__ == "__main__":
    sys.exit(main())

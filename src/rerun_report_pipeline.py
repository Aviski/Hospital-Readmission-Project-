from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.interpretation import (
    compute_error_groups,
    compute_shap_values,
    plot_error_diagnosis_distribution,
    plot_error_distributions,
    plot_error_utilization_scatter,
    plot_false_positive_vs_negative,
    plot_shap_bar,
    plot_shap_dependence,
    plot_shap_summary,
    plot_shap_waterfall,
    summarise_errors,
)
from src.modeling import (
    build_baselines,
    build_param_distributions,
    calibrate_model,
    compare_models,
    cross_validate_model,
    evaluate,
    make_splits,
    plot_calibration_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_analysis,
    select_best_model,
    threshold_sweep,
    tune_model,
)
from src.pipeline import FeatureArtifacts, build_feature_artifacts
from src.utils import get_logger, load_config, save_model, set_seed

logger = get_logger(__name__)


def _configure_cli_warning_filters() -> None:
    """Reduce repetitive third-party warning noise while keeping actionable warnings visible."""
    warnings.filterwarnings(
        "ignore",
        message=r".*'penalty' was deprecated in version 1\.8.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Inconsistent values: penalty=l1 with l1_ratio=0\.0.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Passing `palette` without assigning `hue` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "default",
        category=ConvergenceWarning,
    )


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_output_path(root: Path, config: dict, key: str) -> Path:
    path = root / config["paths"][key]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_model_path(root: Path, config: dict, filename: str) -> Path:
    model_dir = root / config["paths"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / filename


def _resolve_figures_subdir(config: dict, subfolder: str) -> Path:
    base_dir = Path(config["_base_dir"])
    figures_dir = base_dir / config["paths"]["figures_dir"] / subfolder
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _load_runtime_config(config_path: str | None) -> tuple[dict, Path, Path]:
    root = _resolve_project_root()
    resolved_config = Path(config_path) if config_path else root / "config" / "config.yaml"
    config = load_config(resolved_config)
    config["_base_dir"] = str(root)
    set_seed(int(config.get("random_seed", 42)))
    return config, resolved_config, root


def _write_feature_artifacts(
    artifacts: FeatureArtifacts,
    config: dict,
    config_path: Path,
    root: Path,
) -> None:
    encoded_path = _resolve_output_path(root, config, "features_data")
    analysis_path = _resolve_output_path(root, config, "analysis_features_data")
    metadata_path = _resolve_output_path(root, config, "features_metadata")
    raw_data_path = _resolve_output_path(root, config, "raw_data")

    artifacts.encoded.to_csv(encoded_path, index_label="row_id")
    artifacts.analysis.to_csv(analysis_path, index_label="row_id")

    metadata = {
        "config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "raw_csv_size": raw_data_path.stat().st_size,
        "raw_csv_mtime": raw_data_path.stat().st_mtime,
        "cache_version": config.get("cache_version"),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "encoded_rows": int(artifacts.encoded.shape[0]),
        "encoded_cols": int(artifacts.encoded.shape[1]),
        "analysis_rows": int(artifacts.analysis.shape[0]),
        "analysis_cols": int(artifacts.analysis.shape[1]),
        "feature_columns": artifacts.feature_columns,
        "categorical_levels": {
            column: sorted(
                {
                    str(value)
                    for value in pd.Series(artifacts.analysis[column]).dropna().astype(str).unique().tolist()
                }
            )
            for column in artifacts.analysis.select_dtypes(include=["object", "category", "string"]).columns
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Saved encoded features -> %s", encoded_path)
    logger.info("Saved analysis features -> %s", analysis_path)
    logger.info("Saved feature metadata -> %s", metadata_path)


def _fit_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_columns: list[str],
    config: dict,
    root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    config["_figures_subfolder"] = "modeling"
    baselines = build_baselines(config)

    fitted_models: dict[str, Any] = {}
    eval_results: dict[str, dict[str, float]] = {}
    for name, model in baselines.items():
        cross_validate_model(model, X_train, y_train, config)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        eval_results[name] = evaluate(model, X_val, y_val, threshold=0.5)
        logger.info("Validation metrics (%s): %s", name, eval_results[name])

    summary = compare_models(eval_results)
    summary_path = _resolve_output_path(root, config, "metrics_out")
    summary.to_csv(summary_path)
    logger.info("Saved baseline metrics summary -> %s", summary_path)

    best_name = select_best_model(summary, metric="roc_auc")
    save_model(
        fitted_models[best_name],
        _resolve_model_path(root, config, "best_baseline_model.pkl"),
    )

    plot_roc_curves(
        fitted_models,
        X_val,
        y_val,
        config,
        title="ROC Curves - Baseline Models",
        filename="roc_curve_baselines.png",
    )
    plot_pr_curves(
        fitted_models,
        X_val,
        y_val,
        config,
        title="Precision-Recall Curves - Baseline Models",
        filename="pr_curve_baselines.png",
    )
    plot_calibration_curves(
        fitted_models,
        X_val,
        y_val,
        config,
        filename="calibration_baselines.png",
    )
    plot_confusion_matrix(
        fitted_models[best_name],
        X_val,
        y_val,
        best_name,
        config,
        threshold=0.5,
        filename="confusion_matrix_best_baseline.png",
    )
    for name in ("LogisticRegression", "RandomForest"):
        if name in fitted_models:
            plot_feature_importance(fitted_models[name], feature_columns, name, config)

    _plot_baseline_metrics_comparison(summary, y_val, config)
    return fitted_models, summary, best_name


def _plot_baseline_metrics_comparison(
    summary: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> None:
    metrics_to_plot = ["roc_auc", "pr_auc", "recall", "precision", "f1"]
    available = [metric for metric in metrics_to_plot if metric in summary.columns]
    if not available:
        logger.warning("No baseline metrics available for comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    summary[available].plot.bar(ax=ax, rot=0, colormap="Set2")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model Comparison - Validation Set")
    ax.axhline(y_val.mean(), color="black", linestyle="--", lw=1, label="Positive rate (no-skill)")
    ax.legend(loc="lower right", fontsize=9)

    out = _resolve_figures_subdir(config, "modeling") / "metrics_comparison_baselines.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved baseline metrics comparison -> %s", out)


def _fit_tuned_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
    config: dict,
    root: Path,
) -> tuple[Any, pd.DataFrame, str, float]:
    config["_figures_subfolder"] = "tuning"
    tuned_models: dict[str, Any] = {}
    tuned_eval_results: dict[str, dict[str, float]] = {}

    for name, model in build_baselines(config).items():
        tuned = tune_model(
            model,
            build_param_distributions(name, config),
            X_train,
            y_train,
            config,
        )["estimator"]
        tuned_models[name] = tuned
        tuned_eval_results[name] = evaluate(tuned, X_val, y_val, threshold=0.5)
        logger.info("Validation metrics after tuning (%s): %s", name, tuned_eval_results[name])

    plot_roc_curves(
        tuned_models,
        X_val,
        y_val,
        config,
        title="ROC Curves - Tuned Models",
        filename="roc_curve_tuned_models.png",
    )
    plot_pr_curves(
        tuned_models,
        X_val,
        y_val,
        config,
        title="Precision-Recall Curves - Tuned Models",
        filename="pr_curve_tuned_models.png",
    )
    for name, model in tuned_models.items():
        safe_name = name.lower().replace(" ", "")
        plot_confusion_matrix(
            model,
            X_val,
            y_val,
            name,
            config,
            threshold=0.5,
            filename=f"confusion_matrix_tuned_{safe_name}.png",
        )

    tuning_summary = compare_models(tuned_eval_results)
    best_name = select_best_model(tuning_summary, metric="roc_auc")
    calibrated_model = calibrate_model(tuned_models[best_name], X_val, y_val, config, save_plot=True)

    sweep_df, optimal_threshold = threshold_sweep(calibrated_model, X_val, y_val, config)
    plot_threshold_analysis(sweep_df, optimal_threshold, config, filename="threshold_analysis.png")
    calibrated_model.threshold = optimal_threshold

    plot_confusion_matrix(
        calibrated_model,
        X_val,
        y_val,
        f"{best_name} calibrated",
        config,
        threshold=optimal_threshold,
        filename="confusion_matrix_calibrated_optimal.png",
    )

    final_metrics = pd.DataFrame(
        {
            "val_set": evaluate(calibrated_model, X_val, y_val, threshold=optimal_threshold),
            "test_set": evaluate(calibrated_model, X_test, y_test, threshold=optimal_threshold),
        }
    ).T
    final_metrics.index.name = "split"

    tuned_metrics_path = _resolve_output_path(root, config, "metrics_tuned")
    final_metrics.to_csv(tuned_metrics_path)
    logger.info("Saved tuned metrics summary -> %s", tuned_metrics_path)

    save_model(
        {
            "model": calibrated_model,
            "threshold": calibrated_model.threshold,
            "model_name": best_name,
            "val_metrics": final_metrics.loc["val_set"].to_dict(),
            "test_metrics": final_metrics.loc["test_set"].to_dict(),
            "feature_columns": feature_columns,
        },
        _resolve_model_path(root, config, "best_tuned_model.pkl"),
    )
    return calibrated_model, final_metrics, best_name, optimal_threshold


def _run_interpretation(
    model: Any,
    analysis_features: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    shap_samples: int,
    config: dict,
) -> None:
    X_val_analysis = analysis_features.loc[X_val.index].copy()
    shap_values, X_sample = compute_shap_values(
        model,
        X_val,
        config,
        max_samples=shap_samples,
    )
    plot_shap_summary(shap_values, X_sample, config)
    plot_shap_bar(shap_values, config)
    plot_shap_dependence(shap_values, X_sample, config)
    plot_shap_waterfall(shap_values, config)

    threshold = getattr(model, "threshold", 0.5)
    groups = compute_error_groups(model, X_val, y_val, threshold=threshold)
    summarise_errors(groups, X_val_analysis, config)
    plot_error_distributions(groups, X_val_analysis, config)
    plot_error_utilization_scatter(groups, X_val_analysis, config)
    plot_false_positive_vs_negative(groups, X_val_analysis, config)
    plot_error_diagnosis_distribution(groups, X_val_analysis, config)


def run_pipeline(
    config_path: str | None = None,
    skip_interpretation: bool = False,
    shap_samples: int = 500,
) -> dict[str, Any]:
    config, resolved_config, root = _load_runtime_config(config_path)
    artifacts = build_feature_artifacts(config, base_dir=root)
    _write_feature_artifacts(artifacts, config, resolved_config, root)

    target = config["data"]["target_column"]
    X = artifacts.encoded.drop(columns=[target]).copy()
    y = artifacts.encoded[target].copy()
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y, config)

    _, baseline_summary, best_baseline_name = _fit_baselines(
        X_train,
        y_train,
        X_val,
        y_val,
        artifacts.feature_columns,
        config,
        root,
    )
    best_tuned_model, tuned_metrics, best_tuned_name, optimal_threshold = _fit_tuned_model(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        artifacts.feature_columns,
        config,
        root,
    )

    if not skip_interpretation:
        _run_interpretation(
            best_tuned_model,
            artifacts.analysis,
            X_val,
            y_val,
            shap_samples=shap_samples,
            config=config,
        )

    config.pop("_figures_subfolder", None)
    return {
        "best_baseline": best_baseline_name,
        "baseline_summary": baseline_summary,
        "best_tuned": best_tuned_name,
        "optimal_threshold": optimal_threshold,
        "tuned_metrics": tuned_metrics,
        "project_root": root,
        "skip_interpretation": skip_interpretation,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the features, retrain the models, regenerate the report figures, "
            "and refresh the saved metrics artifacts."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a config YAML file. Defaults to config/config.yaml.",
    )
    parser.add_argument(
        "--skip-interpretation",
        action="store_true",
        help="Skip SHAP and error-analysis outputs for a faster rerun.",
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=500,
        help="Number of validation samples to explain when SHAP is enabled.",
    )
    return parser


def main() -> int:
    _configure_cli_warning_filters()
    args = _build_parser().parse_args()
    results = run_pipeline(
        config_path=args.config,
        skip_interpretation=args.skip_interpretation,
        shap_samples=max(1, args.shap_samples),
    )

    print("Rerun Summary")
    print("=" * 80)
    print(f"Best baseline model : {results['best_baseline']}")
    print(f"Best tuned model    : {results['best_tuned']}")
    print(f"Optimal threshold   : {results['optimal_threshold']:.2f}")
    print(f"Interpretation run  : {not results['skip_interpretation']}")
    print("")
    print("Final metrics")
    print(results["tuned_metrics"].round(4).to_string())
    print("")
    print(f"Artifacts refreshed under: {results['project_root']}")
    if results["skip_interpretation"]:
        print("Note: SHAP and error-analysis figures were not refreshed in this run.")
    print("Note: reports/final_report.md is a static narrative and is not rewritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

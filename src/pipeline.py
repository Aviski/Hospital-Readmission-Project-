from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data_preparation import clean_data, encode_features, load_raw_data
from src.feature_engineering import create_features
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureArtifacts:
    clean: pd.DataFrame
    analysis: pd.DataFrame
    encoded: pd.DataFrame
    feature_columns: list[str]


def build_feature_artifacts(
    config: dict,
    base_dir: str | Path | None = None,
) -> FeatureArtifacts:
    raw_path = config["paths"]["raw_data"]
    path = Path(base_dir) / raw_path if base_dir else Path(raw_path)
    target = config["data"]["target_column"]

    logger.info("Building feature artifacts from %s", path)
    df_raw = load_raw_data(path)
    df_clean = clean_data(df_raw, config)
    df_clean.index.name = "row_id"

    df_feats = create_features(df_clean, config)
    df_analysis = df_feats.drop(columns=[target]).copy()
    df_analysis.index.name = "row_id"

    df_encoded = encode_features(df_feats, config)
    df_encoded.index.name = "row_id"

    feature_columns = [c for c in df_encoded.columns if c != target]
    logger.info(
        "Built artifacts: clean=%s analysis=%s encoded=%s",
        df_clean.shape,
        df_analysis.shape,
        df_encoded.shape,
    )
    return FeatureArtifacts(
        clean=df_clean,
        analysis=df_analysis,
        encoded=df_encoded,
        feature_columns=feature_columns,
    )

"""
utils.py — Shared helpers for the Hospital Readmission Risk pipeline.

Provides:
    get_logger   – configures and returns a standard Python logger
    load_config  – loads config/config.yaml into a plain dict
    save_model   – serialises a model to disk with joblib
    load_model   – deserialises a model from disk with joblib
    set_seed     – sets random seeds for reproducibility
"""

import logging
import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout with a timestamp prefix.

    Parameters
    ----------
    name:
        Name for the logger, typically ``__name__`` of the calling module.
    level:
        Logging level (default: ``logging.INFO``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Load a YAML configuration file and return it as a nested dict.

    Parameters
    ----------
    path:
        Path to the YAML file. Defaults to ``config/config.yaml``.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    return config


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str | Path) -> None:
    """Serialise a model (or any picklable object) to disk using joblib.

    Parameters
    ----------
    model:
        The fitted model (or pipeline) to save.
    path:
        Destination file path.  Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logging.getLogger(__name__).info("Model saved to %s", path)


def load_model(path: str | Path) -> Any:
    """Deserialise a model from disk using joblib.

    Parameters
    ----------
    path:
        Path to the serialised model file.

    Returns
    -------
    Any
        The deserialised object.

    Raises
    ------
    FileNotFoundError
        If no file exists at ``path``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")

    model = joblib.load(path)
    logging.getLogger(__name__).info("Model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and (if available) PyTorch / TF.

    Parameters
    ----------
    seed:
        Integer seed value.  Use the ``random_seed`` entry from config.yaml.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logging.getLogger(__name__).debug("Random seed set to %d", seed)

"""
Utility module for the Career Predictor pipeline.

Provides:
- Logging configuration with rotating file handler
- Seed setting for reproducibility
- Directory helpers
"""

import logging
import os
import random
import hashlib
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np


def get_logger(name: str, log_dir: Path = None, level: str = 'INFO') -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name: Logger name (typically __name__ of the calling module).
        log_dir: Directory for log files. If None, uses config default.
        level: Logging level string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    log_format = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / f'{name}.log',
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # XGBoost uses random_state parameter, set during model init
    # No global seed setter needed for XGBoost


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure.

    Returns:
        The same path (for chaining).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_data_hash(filepath: Path) -> str:
    """
    Compute SHA256 hash of a data file for versioning.

    Args:
        filepath: Path to the data file.

    Returns:
        SHA256 hex digest string.
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def format_metrics(metrics: dict) -> str:
    """
    Format a metrics dictionary into a readable string.

    Args:
        metrics: Dictionary of metric_name → value.

    Returns:
        Formatted multi-line string.
    """
    lines = ["=" * 50, "  MODEL EVALUATION METRICS", "=" * 50]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key:.<30} {value:.4f}")
        else:
            lines.append(f"  {key:.<30} {value}")
    lines.append("=" * 50)
    return "\n".join(lines)

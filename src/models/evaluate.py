"""GNN model evaluation for DCN fault detection.

Loads a trained model checkpoint and a test dataset, runs inference, and
computes evaluation metrics appropriate for the three labelling schemes
(binary detection, fault type classification, fault localisation).

Metrics reported:
- Binary detection:       AUROC, F1, precision, recall
- Fault classification:   macro-F1, per-class F1, confusion matrix
- Fault localisation:     top-k accuracy, mean rank of true fault location

Results are written to the output directory as JSON and CSV so they can be
compared across model checkpoints and dataset configurations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def evaluate(
    checkpoint_path: Path,
    test_data_dir: Path,
    output_dir: Path,
    label_scheme: str = "binary",
) -> dict[str, float]:
    """Evaluate a trained model on the test split and save results.

    Args:
        checkpoint_path: Path to a .pt model checkpoint produced by train.py.
        test_data_dir: Directory containing the processed test graph dataset.
        output_dir: Directory where metric files are written.
        label_scheme: One of 'binary', 'classification', 'localization'.
            Must match the scheme used during training.

    Returns:
        Dictionary mapping metric names to scalar values (for programmatic
        use and logging).
    """
    raise NotImplementedError


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    label_scheme: str = "binary",
) -> dict[str, float]:
    """Compute evaluation metrics from prediction arrays.

    Args:
        y_true: Ground-truth integer labels, shape (N,).
        y_pred: Predicted integer labels, shape (N,).
        y_score: Predicted scores or probabilities for AUROC computation,
            shape (N,) for binary or (N, C) for multi-class. Optional.
        label_scheme: Determines which metrics are computed.

    Returns:
        Dictionary of metric name → scalar value.
    """
    raise NotImplementedError

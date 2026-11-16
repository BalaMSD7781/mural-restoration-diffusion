"""Pure NumPy metrics for binary damage-mask evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np


def binary_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """Return Dice, IoU, precision, and recall for two binary masks."""
    predicted = np.asarray(prediction).astype(bool)
    expected = np.asarray(target).astype(bool)
    if predicted.shape != expected.shape:
        raise ValueError(
            f"Prediction and target must have the same shape; "
            f"got {predicted.shape} and {expected.shape}."
        )

    true_positive = float(np.count_nonzero(predicted & expected))
    false_positive = float(np.count_nonzero(predicted & ~expected))
    false_negative = float(np.count_nonzero(~predicted & expected))
    union = true_positive + false_positive + false_negative

    return {
        "dice": (
            2.0 * true_positive
            / (2.0 * true_positive + false_positive + false_negative)
            if true_positive + false_positive + false_negative > 0
            else 1.0
        ),
        "iou": true_positive / union if union > 0 else 1.0,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 1.0
        ),
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 1.0
        ),
    }

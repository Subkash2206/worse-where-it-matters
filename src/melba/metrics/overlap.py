"""Overlap metrics for binary segmentation masks.

All functions accept binary masks (bools) or label/probability maps together
with an explicit threshold.  Conventions are stated per function because the
empty-mask cases are exactly where the previous code base was ambiguous.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "as_binary",
    "check_pair",
    "confusion_counts",
    "dice",
    "jaccard",
    "precision",
    "recall",
    "volume_mm3",
]


def as_binary(mask: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Convert a mask/probability map to a boolean array.

    * boolean input is returned unchanged (a copy is *not* made);
    * floating point input is thresholded at ``threshold`` (default ``0.5``);
    * integer input is thresholded when ``threshold`` is given, otherwise any
      non-zero value counts as foreground.
    """
    array = np.asarray(mask)
    if array.dtype == bool:
        return array
    if threshold is not None:
        return array > threshold
    if np.issubdtype(array.dtype, np.floating):
        return array > 0.5
    return array != 0


def check_pair(pred: np.ndarray, target: np.ndarray) -> None:
    """Validate that two binary masks can be compared elementwise."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {pred.shape} vs {target.shape}")
    if pred.ndim < 1:
        raise ValueError("masks must have at least one dimension")


def confusion_counts(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    """Return ``(true_positive, false_positive, false_negative)`` foreground counts."""
    check_pair(pred, target)
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    return tp, fp, fn


def dice(pred: np.ndarray, target: np.ndarray, *, empty_value: float = 1.0) -> float:
    """Dice similarity coefficient.

    Two empty masks are conventionally a perfect match (``empty_value=1.0``);
    exactly one empty mask scores ``0.0``.
    """
    tp, fp, fn = confusion_counts(pred, target)
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return float(empty_value)
    return float(2 * tp / denominator)


def jaccard(pred: np.ndarray, target: np.ndarray, *, empty_value: float = 1.0) -> float:
    """Intersection over union (IoU)."""
    tp, fp, fn = confusion_counts(pred, target)
    union = tp + fp + fn
    if union == 0:
        return float(empty_value)
    return float(tp / union)


def precision(pred: np.ndarray, target: np.ndarray) -> float:
    """Positive predictive value; ``nan`` when the prediction is empty."""
    tp, fp, _ = confusion_counts(pred, target)
    if tp + fp == 0:
        return float("nan")
    return float(tp / (tp + fp))


def recall(pred: np.ndarray, target: np.ndarray) -> float:
    """Sensitivity; ``nan`` when the reference mask is empty."""
    tp, _, fn = confusion_counts(pred, target)
    if tp + fn == 0:
        return float("nan")
    return float(tp / (tp + fn))


def volume_mm3(mask: np.ndarray, spacing: tuple[float, ...] | float | None = None) -> float:
    """Physical volume of a binary mask in mm^3 (or pixel units if spacing is 1)."""
    array = as_binary(mask)
    if spacing is None:
        spacing = (1.0,) * array.ndim
    elif np.isscalar(spacing):
        spacing = (float(spacing),) * array.ndim
    spacing = tuple(float(value) for value in spacing)
    if len(spacing) != array.ndim:
        raise ValueError(f"spacing has {len(spacing)} entries for a {array.ndim}-D mask")
    if any(value <= 0 for value in spacing):
        raise ValueError(f"spacing entries must be positive, got {spacing}")
    return float(np.count_nonzero(array) * np.prod(spacing))

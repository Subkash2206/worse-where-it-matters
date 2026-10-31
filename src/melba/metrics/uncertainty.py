"""Per-view uncertainty and reliability descriptors for geometric TTA.

The runner keeps the individual (inverse-transformed) view probabilities and
derives, per case and method:

* ``entropy_mean`` — mean binary predictive entropy of the mean probability map;
* ``variance_mean`` — mean voxelwise variance across views;
* ``entropy_mean_roi`` / ``variance_mean_roi`` — the same two restricted to the
  predicted foreground plus a boundary band (default 3 pixels);
* ``view_disagreement`` — fraction of pixels where the binarised views do not all
  agree;
* ``per_view_dice_mean`` / ``per_view_dice_std`` — mean and standard deviation of
  the Dice coefficient of each binarised view against the consensus map.

These are *not* ground-truth metrics, so they are computed outside
:func:`melba.metrics.compute_case_metrics` but written to the same long-format CSV.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .overlap import dice

__all__ = [
    "UNCERTAINTY_METRICS",
    "binarized_views",
    "compute_view_uncertainty",
    "foreground_band",
    "per_view_consensus_dice",
    "predictive_entropy_map",
    "view_disagreement_fraction",
    "voxelwise_variance",
]

UNCERTAINTY_METRICS: tuple[str, ...] = (
    "entropy_mean",
    "variance_mean",
    "entropy_mean_roi",
    "variance_mean_roi",
    "view_disagreement",
    "per_view_dice_mean",
    "per_view_dice_std",
)

_EPS = 1e-7


def _validate(per_view: np.ndarray) -> np.ndarray:
    array = np.asarray(per_view, dtype=np.float64)
    if array.ndim < 3:
        raise ValueError(
            f"per-view probabilities must have shape (V, C, *spatial), got {array.shape}"
        )
    if array.shape[0] < 1:
        raise ValueError("per-view probabilities must contain at least one view")
    if array.shape[1] != 1:
        raise ValueError(
            f"uncertainty descriptors are defined for single-channel predictions, got "
            f"{array.shape[1]} channels"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("per-view probabilities contain non-finite values")
    return array


def predictive_entropy_map(mean_probability: np.ndarray, *, eps: float = _EPS) -> np.ndarray:
    """Binary predictive entropy (natural log) of a probability map."""
    probability = np.clip(np.asarray(mean_probability, dtype=np.float64), eps, 1.0 - eps)
    return -(probability * np.log(probability) + (1.0 - probability) * np.log(1.0 - probability))


def voxelwise_variance(per_view: np.ndarray) -> np.ndarray:
    """Population variance across views for every voxel (same shape as one view)."""
    return _validate(per_view).var(axis=0, ddof=0)


def foreground_band(foreground: np.ndarray, band_px: int = 3) -> np.ndarray:
    """Foreground plus a ``band_px``-pixel band around it (empty mask stays empty)."""
    mask = np.asarray(foreground, dtype=bool)
    band_px = int(band_px)
    if band_px < 0:
        raise ValueError(f"band_px must be non-negative, got {band_px}")
    if not mask.any() or band_px == 0:
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    return ndimage.binary_dilation(mask, structure=structure, iterations=band_px)


def binarized_views(per_view: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Boolean view stack ``(V, *spatial)`` at ``threshold``."""
    return _validate(per_view)[:, 0] > float(threshold)


def view_disagreement_fraction(per_view: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of pixels where the binarised views are not unanimous."""
    masks = binarized_views(per_view, threshold)
    n_views = masks.shape[0]
    if n_views == 1:
        return 0.0
    votes = masks.sum(axis=0)
    disagree = (votes > 0) & (votes < n_views)
    return float(disagree.mean())


def per_view_consensus_dice(
    per_view: np.ndarray, threshold: float = 0.5
) -> tuple[float, float]:
    """Mean and standard deviation of per-view Dice against the consensus map."""
    array = _validate(per_view)
    masks = binarized_views(array, threshold)
    consensus = array.mean(axis=0)[0] > float(threshold)
    scores = np.array([dice(mask, consensus) for mask in masks], dtype=np.float64)
    return float(scores.mean()), float(scores.std(ddof=0))


def compute_view_uncertainty(
    per_view: np.ndarray,
    *,
    threshold: float = 0.5,
    boundary_band_px: int = 3,
) -> dict[str, float]:
    """Compute every descriptor in :data:`UNCERTAINTY_METRICS` for one case/method."""
    array = _validate(per_view)
    mean_probability = array.mean(axis=0, dtype=np.float64)  # (1, *spatial)
    entropy = predictive_entropy_map(mean_probability)
    variance = array.var(axis=0, ddof=0)

    roi = foreground_band(mean_probability[0] > float(threshold), boundary_band_px)
    if roi.any():
        entropy_roi = float(entropy[0][roi].mean())
        variance_roi = float(variance[0][roi].mean())
    else:
        entropy_roi = float("nan")
        variance_roi = float("nan")

    dice_mean, dice_std = per_view_consensus_dice(array, threshold)
    return {
        "entropy_mean": float(entropy.mean()),
        "variance_mean": float(variance.mean()),
        "entropy_mean_roi": entropy_roi,
        "variance_mean_roi": variance_roi,
        "view_disagreement": view_disagreement_fraction(array, threshold),
        "per_view_dice_mean": dice_mean,
        "per_view_dice_std": dice_std,
    }

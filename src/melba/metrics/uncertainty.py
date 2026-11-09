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
  the Dice coefficient of each binarised view against the consensus map;
* ``view_pairwise_dice`` — mean Dice over every *pair* of binarised views (1.0 for a
  single view), the view-agreement score used as a failure detector by
  :mod:`melba.eval.uncertainty_audit`.

These are *not* ground-truth metrics, so they are computed outside
:func:`melba.metrics.compute_case_metrics` but written to the same long-format CSV.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import ndimage

from .overlap import dice

__all__ = [
    "UNCERTAINTY_METRICS",
    "binarized_views",
    "compute_view_uncertainty",
    "foreground_band",
    "mean_pairwise_dice",
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
    "view_pairwise_dice",
)

_EPS = 1e-7


def _validate(per_view: np.ndarray) -> np.ndarray:
    # Conversion to float64 only when needed: the runner passes float32 view stacks of
    # 15 x 3.7 M voxels, and an unconditional astype would copy ~450 MB on *every*
    # descriptor (that copy used to dominate the uncertainty block of the evaluation).
    array = np.asarray(per_view)
    if array.dtype != np.float64:
        array = array.astype(np.float64)
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


def mean_pairwise_dice(per_view: np.ndarray, threshold: float = 0.5) -> float:
    """Mean Dice over every unordered pair of binarised views.

    Unlike :func:`per_view_consensus_dice` (each view against the *consensus*), this
    compares the views with each other, so it estimates how much the TTA members
    agree without letting the averaged map blur their disagreement away.  A single
    view trivially agrees with itself and returns 1.0 (i.e. no measurable spread).
    """
    masks = binarized_views(per_view, threshold)
    if masks.shape[0] < 2:
        return 1.0
    pairs = combinations(range(masks.shape[0]), 2)
    scores = [dice(masks[first], masks[second]) for first, second in pairs]
    return float(np.mean(scores))


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
        "view_pairwise_dice": mean_pairwise_dice(array, threshold),
    }


def _band_torch(mask, band_px: int):
    """``foreground_band`` in torch: ``band_px`` iterations of a 6-neighbour dilation."""
    import torch
    import torch.nn.functional as functional

    out = mask
    if band_px <= 0 or not bool(out.any()):
        return out
    for _ in range(int(band_px)):
        grown = out
        for axis in range(out.ndim):
            for shift in (-1, 1):
                pad = [0] * (2 * out.ndim)
                # pad the side the content is shifted *away* from, then crop the opposite
                # end: left pad + drop last == every voxel moves one step up the axis
                pad[2 * (out.ndim - 1 - axis) + (0 if shift < 0 else 1)] = 1
                padded = functional.pad(out[None, None].to(torch.float32), pad)
                moved = padded[0, 0]
                index = [slice(None)] * out.ndim
                index[axis] = slice(0, -1) if shift < 0 else slice(1, None)
                grown = grown | (moved[tuple(index)] > 0.0)
        out = grown
    return out


def compute_view_uncertainty_torch(
    per_view,
    *,
    threshold: float = 0.5,
    boundary_band_px: int = 3,
) -> dict[str, float]:
    """The same eight descriptors, computed on the tensor's device (GPU).

    Copy of :func:`compute_view_uncertainty` in torch: the descriptors are reductions over
    a ``(V, 1, *spatial)`` stack, so doing them on the GPU before the views are moved to
    the host removes the ~1 s of float64 copies and elementwise work per case and method
    that a 15-view 240^3 evaluation spent on the CPU.  Values agree with the numpy path to
    floating-point tolerance (tested).
    """
    import torch

    views = per_view.to(torch.float64)
    if views.ndim < 3:
        raise ValueError(f"per-view probabilities must have shape (V, C, *spatial), got {tuple(views.shape)}")
    if views.shape[1] != 1:
        raise ValueError(
            f"uncertainty descriptors are defined for single-channel predictions, got "
            f"{views.shape[1]} channels"
        )
    if not bool(torch.isfinite(views).all()):
        raise ValueError("per-view probabilities contain non-finite values")

    mean_probability = views.mean(dim=0)  # (1, *spatial)
    probability = mean_probability.clamp(_EPS, 1.0 - _EPS)
    entropy = -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())
    variance = views.var(dim=0, unbiased=False)

    roi = _band_torch(mean_probability[0] > float(threshold), boundary_band_px)
    if bool(roi.any()):
        entropy_roi = float(entropy[0][roi].mean().item())
        variance_roi = float(variance[0][roi].mean().item())
    else:
        entropy_roi = float("nan")
        variance_roi = float("nan")

    masks = views[:, 0] > float(threshold)  # (V, *spatial) bool
    n_views = int(masks.shape[0])
    votes = masks.sum(dim=0)
    if n_views == 1:
        disagreement = 0.0
    else:
        disagree = (votes > 0) & (votes < n_views)
        disagreement = float(disagree.to(torch.float64).mean().item())

    flat = masks.reshape(n_views, -1).to(torch.float64)
    sizes = flat.sum(dim=1)
    consensus = mean_probability[0].reshape(-1) > float(threshold)
    consensus_size = consensus.to(torch.float64).sum()
    intersection = flat @ consensus.to(torch.float64)
    denominator = sizes + consensus_size
    scores = torch.where(
        denominator > 0,
        2.0 * intersection / denominator.clamp_min(1e-8),
        torch.ones_like(denominator),
    )
    dice_mean = float(scores.mean().item())
    dice_std = float(scores.std(unbiased=False).item())

    if n_views < 2:
        pairwise = 1.0
    else:
        gram = flat @ flat.T
        index = torch.arange(n_views, device=gram.device)
        off_diagonal = index[:, None] != index[None, :]
        pair_denominator = sizes[:, None] + sizes[None, :]
        pair_dice = torch.where(
            pair_denominator > 0,
            2.0 * gram / pair_denominator.clamp_min(1e-8),
            torch.ones_like(pair_denominator),
        )
        pairwise = float(pair_dice[off_diagonal].mean().item())

    return {
        "entropy_mean": float(entropy.mean().item()),
        "variance_mean": float(variance.mean().item()),
        "entropy_mean_roi": entropy_roi,
        "variance_mean_roi": variance_roi,
        "view_disagreement": disagreement,
        "per_view_dice_mean": dice_mean,
        "per_view_dice_std": dice_std,
        "view_pairwise_dice": pairwise,
    }


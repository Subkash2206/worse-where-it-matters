"""Surface-distance metrics.

These replace the previous implementation, which took the maximum directed
Hausdorff distance over *all foreground pixels* on an anisotropic grid without
any spacing information.  Here the metrics are computed between **surfaces**
(boundary voxels), with physical spacing, and HD95 is a percentile over the pooled
bidirectional surface distances.

Conventions
-----------
* surfaces are the boundary voxels of the binary mask (``mask & ~erode(mask)``);
* distances use ``scipy.ndimage.distance_transform_edt`` with ``sampling=spacing``
  so anisotropic voxels are handled correctly;
* ``hd`` is the symmetric Hausdorff distance (100th percentile);
* ``hd95`` follows the MONAI convention:
  ``max(p95(pred -> gt), p95(gt -> pred))``;
* ``hd95_pooled`` is the medpy-style percentile over the pooled distances and is
  kept for reference only;
* ``assd`` is the mean of the pooled distances;
* ``boundary_f1`` is the tolerance-based boundary F1 score, i.e. the normalised
  surface Dice at ``tolerance_mm``;
* two empty masks are a perfect match (distance 0, boundary F1 1); if exactly one
  mask is empty the metric is undefined and ``empty_value`` (``nan`` by default)
  is returned.  The evaluation registry can replace ``nan`` by a penalty.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import ndimage

from .overlap import as_binary

__all__ = [
    "average_surface_distance",
    "boundary_f1",
    "foreground_bbox",
    "grid_diagonal_mm",
    "hausdorff_distance",
    "hausdorff_distance_95",
    "hausdorff_distance_95_pooled",
    "normalize_spacing",
    "surface_distances",
    "surface_mask",
    "surface_metric_bundle",
]


def normalize_spacing(
    spacing: Iterable[float] | float | None, ndim: int
) -> tuple[float, ...]:
    """Validate a spacing value and broadcast scalars to ``ndim`` entries."""
    if spacing is None:
        return (1.0,) * ndim
    if np.isscalar(spacing):
        spacing = (float(spacing),) * ndim  # type: ignore[arg-type]
    values = tuple(float(value) for value in spacing)  # type: ignore[union-attr]
    if len(values) != ndim:
        raise ValueError(f"spacing has {len(values)} entries for a {ndim}-D mask")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"spacing entries must be finite and positive, got {values}")
    return values


def surface_mask(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Return the boundary voxels of a binary mask.

    ``connectivity=1`` uses the 6-neighbourhood in 3D (4-neighbourhood in 2D).
    Voxels on the array border count as surface, i.e. the outside of the array is
    background (``border_value=0``).
    """
    array = as_binary(mask)
    if not array.any():
        return np.zeros_like(array, dtype=bool)
    footprint = ndimage.generate_binary_structure(array.ndim, connectivity)
    eroded = ndimage.binary_erosion(array, structure=footprint, border_value=0)
    return array & ~eroded


def grid_diagonal_mm(
    shape: tuple[int, ...], spacing: tuple[float, ...] | float | None = None
) -> float:
    """Diagonal of the image grid in physical units (used as empty-mask penalty)."""
    values = normalize_spacing(spacing, len(shape))
    return float(math.sqrt(sum(((size - 1) * step) ** 2 for size, step in zip(shape, values))))


def surface_distances(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Iterable[float] | float | None = None,
    connectivity: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Bidirectional surface distances.

    Returns
    -------
    (d_pred_to_gt, d_gt_to_pred):
        Flat float64 arrays with the distance from each prediction surface voxel
        to the nearest reference surface voxel and vice versa.  Both are empty
        when either mask is empty (the metric is then undefined).
    """
    pred_bin = as_binary(pred)
    target_bin = as_binary(target)
    if pred_bin.shape != target_bin.shape:
        raise ValueError(f"pred and target shapes differ: {pred_bin.shape} vs {target_bin.shape}")
    values = normalize_spacing(spacing, pred_bin.ndim)

    pred_surface = surface_mask(pred_bin, connectivity)
    target_surface = surface_mask(target_bin, connectivity)
    if not pred_surface.any() or not target_surface.any():
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # distance_transform_edt measures the distance to the nearest zero (False)
    # element, so inverting the surface mask gives the distance to that surface.
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=values)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=values)
    return (
        distance_to_target[pred_surface].astype(np.float64, copy=False),
        distance_to_pred[target_surface].astype(np.float64, copy=False),
    )


def _empty_distance_value(pred_size: int, target_size: int, empty_value: float) -> float:
    """Distance for degenerate cases: 0 if both masks are empty, else ``empty_value``."""
    if pred_size == 0 and target_size == 0:
        return 0.0
    return float(empty_value)


def _pooled(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Iterable[float] | float | None,
    connectivity: int,
) -> tuple[np.ndarray, int, int]:
    pred_bin = as_binary(pred)
    target_bin = as_binary(target)
    d_pred, d_target = surface_distances(pred_bin, target_bin, spacing, connectivity)
    return np.concatenate([d_pred, d_target]), int(pred_bin.sum()), int(target_bin.sum())


def foreground_bbox(
    *masks: np.ndarray, margin: int = 1
) -> tuple[tuple[slice, ...], bool]:
    """Bounding box (as slices) of the union foreground of ``masks``, plus a clip flag.

    Returns ``(slices, touches_border)``.  ``touches_border`` is ``True`` when the padded
    box had to be clipped at the array edge, i.e. when some foreground voxel sits on the
    array border (there the surface definition already treats the outside as background,
    so computing inside the clipped box is still exact).  An empty union returns slices
    covering the whole array.
    """
    arrays = [as_binary(mask) for mask in masks]
    if not arrays:
        raise ValueError("foreground_bbox needs at least one mask")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("all masks must have the same shape")
    union = np.zeros(shape, dtype=bool)
    for array in arrays:
        union |= array
    if not union.any():
        return tuple(slice(0, size) for size in shape), False
    index = np.argwhere(union)
    starts = np.maximum(index.min(axis=0) - int(margin), 0)
    stops = np.minimum(index.max(axis=0) + int(margin) + 1, np.array(shape))
    touches_border = bool(np.any(starts == 0) or np.any(stops == np.array(shape)))
    return (
        tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops)),
        touches_border,
    )


def surface_metric_bundle(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Iterable[float] | float | None = None,
    *,
    connectivity: int = 1,
    percentile: float = 95.0,
    tolerance_mm: float = 2.0,
    empty_value: float = float("nan"),
) -> dict[str, float]:
    """HD100, HD95, pooled HD95, ASSD and boundary F1 from **one** pair of transforms.

    Each of those five metrics needs the same two surface distance transforms
    (``pred -> gt`` and ``gt -> pred``); computing them once and reducing the same
    distributions five times is what makes the surface block of the evaluation affordable.
    The values are identical to calling :func:`hausdorff_distance`,
    :func:`hausdorff_distance_95`, :func:`hausdorff_distance_95_pooled`,
    :func:`average_surface_distance` and :func:`boundary_f1` separately.
    """
    if not 0 < float(percentile) <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    if tolerance_mm < 0:
        raise ValueError(f"tolerance_mm must be non-negative, got {tolerance_mm}")
    pred_bin = as_binary(pred)
    target_bin = as_binary(target)
    if pred_bin.shape != target_bin.shape:
        raise ValueError(f"pred and target shapes differ: {pred_bin.shape} vs {target_bin.shape}")

    d_pred, d_target = surface_distances(pred_bin, target_bin, spacing, connectivity)
    if d_pred.size == 0:
        # degenerate: both empty is a perfect match, exactly one empty is undefined
        empty = _empty_distance_value(int(pred_bin.sum()), int(target_bin.sum()), empty_value)
        both_empty = not pred_bin.any() and not target_bin.any()
        return {
            "hd": empty,
            "hd95": empty,
            "hd95_pooled": empty,
            "assd": empty,
            "boundary_f1": 1.0 if both_empty else float(empty_value),
        }

    pooled = np.concatenate([d_pred, d_target])
    within = int(np.count_nonzero(d_pred <= tolerance_mm)) + int(
        np.count_nonzero(d_target <= tolerance_mm)
    )
    return {
        "hd": float(np.max(pooled)),
        "hd95": float(
            max(np.percentile(d_pred, percentile), np.percentile(d_target, percentile))
        ),
        "hd95_pooled": float(np.percentile(pooled, percentile)),
        "assd": float(np.mean(pooled)),
        "boundary_f1": float(within / pooled.size),
    }


def hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Iterable[float] | float | None = None,
    connectivity: int = 1,
    empty_value: float = float("nan"),
) -> float:
    """Symmetric Hausdorff distance between surfaces (HD100), in physical units."""
    pooled, pred_size, target_size = _pooled(pred, target, spacing, connectivity)
    if pooled.size == 0:
        return _empty_distance_value(pred_size, target_size, empty_value)
    return float(np.max(pooled))


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Iterable[float] | float | None = None,
    percentile: float = 95.0,
    connectivity: int = 1,
    empty_value: float = float("nan"),
) -> float:
    """HD95 with the MONAI convention: ``max(p95(pred->gt), p95(gt->pred))``.

    This matches :func:`monai.metrics.compute_hausdorff_distance` with
    ``percentile=95`` and ``spacing`` set, i.e. each directed surface-distance
    distribution is reduced separately and the larger value is reported.
    """
    if not 0 < float(percentile) <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    pred_bin = as_binary(pred)
    target_bin = as_binary(target)
    if pred_bin.shape != target_bin.shape:
        raise ValueError(f"pred and target shapes differ: {pred_bin.shape} vs {target_bin.shape}")
    if not pred_bin.any() or not target_bin.any():
        return _empty_distance_value(int(pred_bin.sum()), int(target_bin.sum()), empty_value)

    d_pred, d_target = surface_distances(pred_bin, target_bin, spacing, connectivity)
    return float(
        max(np.percentile(d_pred, percentile), np.percentile(d_target, percentile))
    )


def hausdorff_distance_95_pooled(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Iterable[float] | float | None = None,
    percentile: float = 95.0,
    connectivity: int = 1,
    empty_value: float = float("nan"),
) -> float:
    """HD95 over the pooled bidirectional distances (medpy-style, reference only)."""
    if not 0 < float(percentile) <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    pooled, pred_size, target_size = _pooled(pred, target, spacing, connectivity)
    if pooled.size == 0:
        return _empty_distance_value(pred_size, target_size, empty_value)
    return float(np.percentile(pooled, percentile))


def average_surface_distance(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Iterable[float] | float | None = None,
    connectivity: int = 1,
    empty_value: float = float("nan"),
) -> float:
    """Average symmetric surface distance (ASSD), in physical units."""
    pooled, pred_size, target_size = _pooled(pred, target, spacing, connectivity)
    if pooled.size == 0:
        return _empty_distance_value(pred_size, target_size, empty_value)
    return float(np.mean(pooled))


def boundary_f1(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Iterable[float] | float | None = None,
    tolerance_mm: float = 2.0,
    connectivity: int = 1,
    empty_value: float = float("nan"),
) -> float:
    """Tolerance-based boundary F1 (equivalently, surface Dice at ``tolerance_mm``).

    A surface voxel is a true positive when it lies within ``tolerance_mm`` of the
    other surface.  The score is symmetric and lies in [0, 1].
    """
    if tolerance_mm < 0:
        raise ValueError(f"tolerance_mm must be non-negative, got {tolerance_mm}")
    pred_bin = as_binary(pred)
    target_bin = as_binary(target)
    if pred_bin.shape != target_bin.shape:
        raise ValueError(f"pred and target shapes differ: {pred_bin.shape} vs {target_bin.shape}")
    pred_empty = not bool(pred_bin.any())
    target_empty = not bool(target_bin.any())
    if pred_empty and target_empty:
        return 1.0
    if pred_empty or target_empty:
        return float(empty_value)

    d_pred, d_target = surface_distances(pred_bin, target_bin, spacing, connectivity)
    n_pred, n_target = d_pred.size, d_target.size
    within = int(np.count_nonzero(d_pred <= tolerance_mm)) + int(
        np.count_nonzero(d_target <= tolerance_mm)
    )
    return float(within / (n_pred + n_target))


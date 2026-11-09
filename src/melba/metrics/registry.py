"""Per-case metric computation used by the evaluation runner.

One call returns a ``{metric_name: value}`` dictionary plus an ``empty_flag`` for a
single case, which the runner writes as several rows of the long-format CSV.
Aggregation never happens here: means across cases are computed later from the
per-case records.

Ground-truth metrics are computed by :func:`compute_case_metrics`; the per-view
uncertainty descriptors live in :mod:`melba.metrics.uncertainty` because they need
the individual TTA view predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .overlap import as_binary, dice, jaccard, precision, recall, volume_mm3
from .surface import (
    average_surface_distance,
    boundary_f1,
    foreground_bbox,
    grid_diagonal_mm,
    hausdorff_distance,
    hausdorff_distance_95,
    hausdorff_distance_95_pooled,
    normalize_spacing,
    surface_metric_bundle,
)
from .uncertainty import UNCERTAINTY_METRICS

__all__ = [
    "AVAILABLE_METRICS",
    "DISTANCE_METRICS",
    "EMPTY_FLAGS",
    "EMPTY_POLICIES",
    "GT_METRICS",
    "MetricSettings",
    "OVERLAP_METRICS",
    "SURFACE_METRICS",
    "UNCERTAINTY_METRICS",
    "VOLUME_METRICS",
    "compute_case_metrics",
    "empty_flag",
]

OVERLAP_METRICS: tuple[str, ...] = ("dice", "jaccard", "precision", "recall")
DISTANCE_METRICS: tuple[str, ...] = ("hd", "hd95", "hd95_pooled", "assd", "boundary_f1")
SURFACE_METRICS = DISTANCE_METRICS
VOLUME_METRICS: tuple[str, ...] = (
    "pred_volume_mm3",
    "gt_volume_mm3",
    "volume_difference_mm3",
)
GT_METRICS: tuple[str, ...] = OVERLAP_METRICS + DISTANCE_METRICS + VOLUME_METRICS
AVAILABLE_METRICS: tuple[str, ...] = GT_METRICS + UNCERTAINTY_METRICS
EMPTY_POLICIES: tuple[str, ...] = ("nan", "penalize")
EMPTY_FLAGS: tuple[str, ...] = ("both_nonempty", "both_empty", "pred_empty", "gt_empty")
#: similarity scores that are conventionally 1.0 when *both* masks are empty (Dice and
#: Jaccard by definition, boundary F1 because there is no boundary to disagree on).  With
#: ``score_empty_reference=False`` they become undefined instead of "perfect".
SCORING_METRICS: tuple[str, ...] = ("dice", "jaccard", "boundary_f1")


def empty_flag(pred: np.ndarray, target: np.ndarray) -> str:
    """Classify a prediction/reference pair by which masks are empty."""
    pred_empty = not bool(np.asanyarray(pred).any())
    target_empty = not bool(np.asanyarray(target).any())
    if pred_empty and target_empty:
        return "both_empty"
    if pred_empty:
        return "pred_empty"
    if target_empty:
        return "gt_empty"
    return "both_nonempty"


@dataclass(frozen=True)
class MetricSettings:
    """Which metrics to compute and how to treat degenerate (empty) masks."""

    names: tuple[str, ...] = AVAILABLE_METRICS
    hd_percentile: float = 95.0
    connectivity: int = 1
    boundary_tolerance_mm: float = 2.0
    empty_policy: str = "nan"
    # ``True`` (legacy): two empty masks are a perfect match, so Dice/Jaccard = 1.0 when
    # the reference is empty (and the prediction too).  ``False``: a case whose reference
    # is empty is never scored as a success -- the overlap metrics become ``nan`` (they are
    # undefined, the case is flagged through ``empty_flag``) and the case is left out of
    # the mean instead of inflating it.  BraTS-style datasets must use ``False``: ~3% of
    # the cases have no enhancing tumour, and "Dice 1.0 for a correct empty prediction"
    # silently adds ~0.03-0.05 to the mean.
    score_empty_reference: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "score_empty_reference", bool(self.score_empty_reference))
        unknown = [name for name in self.names if name not in AVAILABLE_METRICS]
        if unknown:
            raise ValueError(f"Unknown metric(s) {unknown}; available: {list(AVAILABLE_METRICS)}")
        if not 0 < float(self.hd_percentile) <= 100:
            raise ValueError(f"hd_percentile must be in (0, 100], got {self.hd_percentile}")
        if int(self.connectivity) not in (1, 2, 3):
            raise ValueError(f"connectivity must be 1, 2 or 3, got {self.connectivity}")
        if float(self.boundary_tolerance_mm) < 0:
            raise ValueError("boundary_tolerance_mm must be non-negative")
        if self.empty_policy not in EMPTY_POLICIES:
            raise ValueError(
                f"empty_policy must be one of {EMPTY_POLICIES}, got {self.empty_policy!r}"
            )

    @property
    def gt_names(self) -> tuple[str, ...]:
        """Metric names computed against the reference mask."""
        return tuple(name for name in self.names if name in GT_METRICS)

    @property
    def uncertainty_names(self) -> tuple[str, ...]:
        """Per-view descriptors computed by the runner from the view stack."""
        return tuple(name for name in self.names if name in UNCERTAINTY_METRICS)


    @classmethod
    def from_config(cls, config: Any) -> "MetricSettings":
        """Build settings from the ``metrics`` section of a MELBA config.

        Unknown keys are rejected rather than silently ignored, so a typo such as
        ``empty_polcy`` cannot quietly fall back to the default policy.
        """
        allowed = {field.name for field in fields(cls)}
        section: Mapping[str, Any] = config.section("metrics")
        unknown = sorted(set(section) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown key(s) {unknown} in the 'metrics' config section; "
                f"allowed: {sorted(allowed)}"
            )
        return cls(**dict(section))


def compute_case_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Iterable[float] | float | None = None,
    threshold: float = 0.5,
    settings: MetricSettings | None = None,
    *,
    penalty_shape: Sequence[int] | None = None,
    restrict_to_bbox: bool = True,
    bbox_margin: int | None = None,
) -> dict[str, Any]:
    """Compute the configured ground-truth metrics for one prediction/reference pair.

    Parameters
    ----------
    pred:
        Probability map (or binary map) on the reference grid.
    target:
        Reference mask on the same grid.
    spacing:
        Physical voxel/pixel spacing used by every distance and volume metric.
    threshold:
        Threshold applied to ``pred`` (and to a non-binary ``target``).
    settings:
        Metric selection and conventions.  With ``empty_policy="nan"`` (default)
        metrics that are undefined for empty masks are returned as ``nan`` and the
        case is flagged through ``empty_flag``; ``"penalize"`` replaces them by the
        image diagonal (sensitivity analysis only).
    penalty_shape:
        Shape used for that image diagonal.  When the arrays were cropped (see
        ``restrict_to_bbox``) the caller passes the *original* shape, so a penalised
        value cannot depend on how much background was cropped away.
    restrict_to_bbox:
        Compute inside the union bounding box of both masks (plus ``bbox_margin``)
        instead of the whole array.  This is exact - the box contains every foreground
        voxel of both masks and a background rim, so no surface voxel changes and no
        background voxel contributes to these metrics - and it is what keeps a
        240x240x155 evaluation affordable.  The box is asserted to contain both masks;
        if it does not (should be impossible) the full array is used instead.
    bbox_margin:
        Background rim kept around the box, in voxels.  Defaults to a rim wide enough for
        the boundary-F1 tolerance, so cropped and full-grid values agree exactly.

    Returns
    -------
    dict
        One entry per requested ground-truth metric plus ``empty_flag``.
    """
    settings = settings or MetricSettings()
    pred_bin = as_binary(pred, threshold)
    target_bin = as_binary(target, threshold)
    if pred_bin.shape != target_bin.shape:
        raise ValueError(f"pred and target shapes differ: {pred_bin.shape} vs {target_bin.shape}")

    spacing_values = normalize_spacing(spacing, pred_bin.ndim)
    full_shape = tuple(int(size) for size in (penalty_shape or pred_bin.shape))
    # the margin keeps a rim of background inside the box: with it the box can never clip
    # a surface voxel, so every metric is bit-identical to the full-array computation
    margin = (
        int(bbox_margin)
        if bbox_margin is not None
        else max(1, int(math.ceil(float(settings.boundary_tolerance_mm) / min(spacing_values))))
    )
    box = tuple(slice(0, size) for size in pred_bin.shape)
    cropped = False
    if restrict_to_bbox and any(size > 1 for size in pred_bin.shape):
        candidate, _touches_border = foreground_bbox(pred_bin, target_bin, margin=margin)
        cropped_pred = pred_bin[candidate]
        cropped_target = target_bin[candidate]
        # exactness guard: the box must contain the masks, otherwise use the whole array
        if (
            int(cropped_pred.sum()) == int(pred_bin.sum())
            and int(cropped_target.sum()) == int(target_bin.sum())
        ):
            box = candidate
            pred_bin = np.ascontiguousarray(cropped_pred)
            target_bin = np.ascontiguousarray(cropped_target)
            cropped = True

    penalty = grid_diagonal_mm(full_shape, spacing_values)
    flag = empty_flag(pred_bin, target_bin)
    both_empty = flag == "both_empty"

    # one pair of surface distance transforms, reused by HD100/HD95/HD95_pooled/ASSD/BF1
    surface_names = {"hd", "hd95", "hd95_pooled", "assd", "boundary_f1"}
    surface_values: dict[str, float] | None = None
    if surface_names.intersection(settings.gt_names):
        surface_values = surface_metric_bundle(
            pred_bin,
            target_bin,
            spacing_values,
            connectivity=settings.connectivity,
            percentile=settings.hd_percentile,
            tolerance_mm=settings.boundary_tolerance_mm,
        )

    results: dict[str, Any] = {}
    for name in settings.gt_names:
        if name == "dice":
            value = dice(pred_bin, target_bin)
        elif name == "jaccard":
            value = jaccard(pred_bin, target_bin)
        elif name == "precision":
            value = precision(pred_bin, target_bin)
        elif name == "recall":
            value = recall(pred_bin, target_bin)
        elif name in surface_names and surface_values is not None:
            value = surface_values[name]
        elif name == "hd":
            value = hausdorff_distance(pred_bin, target_bin, spacing_values, settings.connectivity)
        elif name == "hd95":
            value = hausdorff_distance_95(
                pred_bin,
                target_bin,
                spacing_values,
                percentile=settings.hd_percentile,
                connectivity=settings.connectivity,
            )
        elif name == "hd95_pooled":
            value = hausdorff_distance_95_pooled(
                pred_bin,
                target_bin,
                spacing_values,
                percentile=settings.hd_percentile,
                connectivity=settings.connectivity,
            )
        elif name == "assd":
            value = average_surface_distance(
                pred_bin, target_bin, spacing_values, settings.connectivity
            )
        elif name == "boundary_f1":
            value = boundary_f1(
                pred_bin,
                target_bin,
                spacing_values,
                tolerance_mm=settings.boundary_tolerance_mm,
                connectivity=settings.connectivity,
            )
        elif name == "pred_volume_mm3":
            value = volume_mm3(pred_bin, spacing_values)
        elif name == "gt_volume_mm3":
            value = volume_mm3(target_bin, spacing_values)
        elif name == "volume_difference_mm3":
            value = volume_mm3(pred_bin, spacing_values) - volume_mm3(target_bin, spacing_values)
        else:  # pragma: no cover - guarded by MetricSettings validation
            raise ValueError(f"Unhandled metric {name!r}")

        if settings.empty_policy == "penalize" and math.isnan(value):
            # Exactly one mask was empty (both-empty cases are handled above):
            # distance metrics are undefined, so the worst-case value is used
            # instead of silently dropping the case; precision/recall become 0.
            if name in DISTANCE_METRICS:
                value = 0.0 if name == "boundary_f1" else penalty
            elif not both_empty:
                value = 0.0
        if not settings.score_empty_reference and flag in ("gt_empty", "both_empty"):
            # A case without reference foreground must not be counted as a success: the
            # similarity scores are undefined (``nan``) and stay out of the mean, or fall
            # to the worst case under the penalising policy.  ``empty_flag`` keeps the case
            # visible in the per-case CSV.
            if name in SCORING_METRICS:
                value = 0.0 if settings.empty_policy == "penalize" else float("nan")
        results[name] = float(value)

    results["empty_flag"] = flag
    return results


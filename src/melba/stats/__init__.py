"""Paired, non-parametric analysis of long-format per-case results."""

from __future__ import annotations

from .detection import (
    analyze_detection,
    mcnemar_exact,
    missed_detection_rate,
    transition_table,
)
from .paired import (
    DISTANCE_RESTRICTED_METRICS,
    METRIC_DIRECTIONS,
    analyze_long,
    analyze_pooled_folds,
    analyze_sensitivity,
    bootstrap_ci_mean,
    holm_adjust,
    metric_direction,
    paired_comparison,
    paired_values,
    pool_folds,
    rank_biserial,
)

__all__ = [
    "DISTANCE_RESTRICTED_METRICS",
    "METRIC_DIRECTIONS",
    "analyze_detection",
    "analyze_long",
    "analyze_pooled_folds",
    "analyze_sensitivity",
    "bootstrap_ci_mean",
    "holm_adjust",
    "mcnemar_exact",
    "metric_direction",
    "missed_detection_rate",
    "paired_comparison",
    "paired_values",
    "pool_folds",
    "rank_biserial",
    "transition_table",
]

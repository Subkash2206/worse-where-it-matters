"""Paired, non-parametric analysis of long-format per-case results."""

from __future__ import annotations

from .paired import (
    METRIC_DIRECTIONS,
    analyze_long,
    analyze_pooled_folds,
    bootstrap_ci_mean,
    holm_adjust,
    metric_direction,
    paired_comparison,
    paired_values,
    pool_folds,
    rank_biserial,
)

__all__ = [
    "METRIC_DIRECTIONS",
    "analyze_long",
    "analyze_pooled_folds",
    "bootstrap_ci_mean",
    "holm_adjust",
    "metric_direction",
    "paired_comparison",
    "paired_values",
    "pool_folds",
    "rank_biserial",
]

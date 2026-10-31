"""Paired statistical analysis of per-case segmentation results.

The previous code base used a paired t-test on Dice/HD95 deltas, which are
heavy-tailed and bounded, and it treated frames from the same video as independent
observations.  This module instead provides

* the Wilcoxon signed-rank test as the primary test (exact for small samples),
  with a sign test as an explicit fallback;
* a seeded percentile bootstrap confidence interval for the mean delta;
* the matched-pairs rank-biserial correlation as an effect size;
* Holm-Bonferroni correction across metrics;
* ``unit="group"`` aggregation, which averages frames within a video/patient
  before testing, so the effective sample size is the number of groups.

The paired t-test is still computed, but only as a clearly labelled secondary
column for reviewers who expect it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "analyze_long",
    "bootstrap_ci_mean",
    "holm_adjust",
    "paired_comparison",
    "paired_values",
    "rank_biserial",
]

_BOOTSTRAP_CHUNK = 512


def bootstrap_ci_mean(
    values: Iterable[float],
    *,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (deterministic for a given seed)."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (float("nan"), float("nan"))
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    rng = np.random.default_rng(int(seed))
    means = np.empty(n_bootstrap, dtype=float)
    n = array.size
    for start in range(0, n_bootstrap, _BOOTSTRAP_CHUNK):
        stop = min(start + _BOOTSTRAP_CHUNK, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = array[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return (float(low), float(high))


def rank_biserial(diff: Iterable[float]) -> float:
    """Matched-pairs rank-biserial correlation in [-1, 1].

    Positive values mean the treatment is better on most pairs; zero means the
    differences are symmetric.  Zero differences are discarded, matching the
    Wilcoxon signed-rank convention.
    """
    values = np.asarray(list(diff), dtype=float)
    values = values[np.isfinite(values)]
    values = values[values != 0]
    if values.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(values))
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    total = positive + negative
    if total == 0:  # pragma: no cover - defensive
        return 0.0
    return float((positive - negative) / total)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs are ignored and stay NaN)."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    count = int(valid.sum())
    if count == 0:
        return adjusted

    order = np.argsort(values[valid], kind="stable")
    sorted_values = values[valid][order]
    running = 0.0
    for rank, value in enumerate(sorted_values):
        candidate = (count - rank) * float(value)
        running = max(running, candidate)
        sorted_values[rank] = min(1.0, running)
    adjusted[np.flatnonzero(valid)[order]] = sorted_values
    return adjusted


def _sign_test(diff: np.ndarray, alternative: str) -> tuple[float, float]:
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return (0.0, 1.0)
    positives = int(np.count_nonzero(nonzero > 0))
    result = stats.binomtest(positives, nonzero.size, 0.5, alternative=alternative)
    return (float(positives), float(result.pvalue))


def _paired_test(diff: np.ndarray, test: str, alternative: str) -> tuple[float, float, str]:
    """Return ``(statistic, p_value, test_name)`` for the primary test."""
    test = (test or "wilcoxon").strip().lower()
    if test == "sign":
        statistic, p_value = _sign_test(diff, alternative)
        return statistic, p_value, "sign"

    if test != "wilcoxon":
        raise ValueError(f"Unknown primary test {test!r}; expected 'wilcoxon' or 'sign'")

    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        # All differences are exactly zero: no evidence of any effect.
        return 0.0, 1.0, "wilcoxon"
    try:
        result = stats.wilcoxon(nonzero, alternative=alternative, correction=False)
    except ValueError:  # pragma: no cover - tiny/degenerate samples
        statistic, p_value = _sign_test(diff, alternative)
        return statistic, p_value, "sign (fallback)"
    return float(result.statistic), float(result.pvalue), "wilcoxon"


#: default metric directions; distance metrics improve when they decrease
METRIC_DIRECTIONS: dict[str, str] = {
    "dice": "higher",
    "jaccard": "higher",
    "precision": "higher",
    "recall": "higher",
    "boundary_f1": "higher",
    "hd": "lower",
    "hd95": "lower",
    "hd95_pooled": "lower",
    "assd": "lower",
}


def metric_direction(metric: str) -> str:
    return METRIC_DIRECTIONS.get(str(metric).replace("_penalized", ""), "higher")


#: distance metrics whose primary paired comparison is restricted to cases that are
#: non-empty under both methods (see :func:`paired_values`)
DISTANCE_RESTRICTED_METRICS: tuple[str, ...] = ("hd", "hd95", "hd95_pooled", "assd")


def _preferred_method(delta: float, direction: str, treatment: str, control: str) -> str:
    if not math.isfinite(delta) or abs(delta) < 1e-12:
        return "tie"
    return treatment if (delta > 0) == (direction == "higher") else control


def _paired_frame(
    frame: pd.DataFrame,
    metric: str,
    treatment: str,
    control: str,
    *,
    unit: str,
    require_both_nonempty: bool,
) -> tuple[pd.DataFrame, int]:
    """Build the paired frame and count the cases that contributed to it."""
    required = {"case_id", "group_id", "method", "metric", "value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Long-format frame is missing column(s) {missing}")
    if unit not in {"case", "group"}:
        raise ValueError(f"unit must be 'case' or 'group', got {unit!r}")
    key = "case_id" if unit == "case" else "group_id"
    if require_both_nonempty and "empty_flag" not in frame.columns:
        raise ValueError(
            "require_both_nonempty needs an 'empty_flag' column in the long-format frame"
        )

    subset = frame.loc[frame["metric"] == metric]
    if subset.empty:
        raise ValueError(f"No rows for metric {metric!r}")
    available = set(subset["method"].unique())
    for role, method in (("control", control), ("treatment", treatment)):
        if method not in available:
            raise ValueError(
                f"{role} method {method!r} not found for metric {metric!r}; "
                f"available: {sorted(available)}"
            )

    columns = ["case_id", key, "method", "value"]
    if "empty_flag" in frame.columns:
        columns.append("empty_flag")
    base = subset.loc[subset["method"].isin([control, treatment]), list(dict.fromkeys(columns))].copy()
    base["value"] = base["value"].astype(float)
    base = base[np.isfinite(base["value"])]
    if require_both_nonempty:
        base = base[base["empty_flag"] == "both_nonempty"]

    paired_cases = base.groupby("case_id")["method"].nunique()
    n_cases_used = int((paired_cases == 2).sum())

    values = base[[key, "method", "value"]]
    if unit == "group":
        values = values.groupby([key, "method"], as_index=False)["value"].mean()
    pivot = (
        values.pivot(index=key, columns="method", values="value")
        .dropna(subset=[control, treatment])
        .rename_axis("unit_id")
        .reset_index()
    )
    ordered = pivot.sort_values(
        "unit_id", key=lambda column: column.astype(str)
    ).reset_index(drop=True)
    return ordered[["unit_id", control, treatment]], n_cases_used


def paired_values(
    frame: pd.DataFrame,
    metric: str,
    treatment: str,
    control: str,
    *,
    unit: str = "group",
    require_both_nonempty: bool = False,
) -> pd.DataFrame:
    """Return complete pairs ``(unit_id, control, treatment)`` for one metric.

    With ``unit="group"`` each case's value is averaged within its group before
    pairing, so the test's sample size is the number of videos/patients rather than
    the number of (near-duplicate) frames.

    With ``require_both_nonempty=True`` a case enters the comparison only when the
    reference is non-empty **and both methods** predicted a non-empty mask.  This is
    required for distance metrics: under ``empty_policy: nan`` an empty prediction
    would otherwise drop out of the mean, so a method that misses lesions would be
    rewarded with a lower HD95/ASSD (survivorship bias).
    """
    pairs, _ = _paired_frame(
        frame,
        metric,
        treatment,
        control,
        unit=unit,
        require_both_nonempty=require_both_nonempty,
    )
    return pairs


def paired_comparison(
    control: Iterable[float],
    treatment: Iterable[float],
    *,
    primary_test: str = "wilcoxon",
    alternative: str = "two-sided",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare paired control/treatment values with an appropriate test."""
    control_values = np.asarray(list(control), dtype=float)
    treatment_values = np.asarray(list(treatment), dtype=float)
    if control_values.shape != treatment_values.shape:
        raise ValueError(
            f"Paired arrays must have the same length, got {control_values.shape} and "
            f"{treatment_values.shape}"
        )

    finite = np.isfinite(control_values) & np.isfinite(treatment_values)
    n_dropped = int((~finite).sum())
    control_values = control_values[finite]
    treatment_values = treatment_values[finite]
    diff = treatment_values - control_values
    n_pairs = int(diff.size)

    row: dict[str, Any] = {
        "n_pairs": n_pairs,
        "n_dropped": n_dropped,
        "mean_control": float(control_values.mean()) if n_pairs else float("nan"),
        "mean_treatment": float(treatment_values.mean()) if n_pairs else float("nan"),
        "delta_mean": float(diff.mean()) if n_pairs else float("nan"),
        "delta_median": float(np.median(diff)) if n_pairs else float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
        "test": primary_test,
        "alternative": alternative,
        "effect_size": float("nan"),
        "effect_size_name": "rank_biserial",
        "secondary_test": "t_test_rel",
        "t_statistic": float("nan"),
        "p_value_secondary": float("nan"),
    }
    if n_pairs == 0:
        return row

    low, high = bootstrap_ci_mean(diff, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    statistic, p_value, test_name = _paired_test(diff, primary_test, alternative)
    row.update(
        {
            "ci_low": low,
            "ci_high": high,
            "statistic": statistic,
            "p_value": p_value,
            "test": test_name,
            "effect_size": rank_biserial(diff),
        }
    )

    if n_pairs >= 2 and float(np.std(diff)) > 0:
        # A paired t-test is undefined when every delta is identical; in that case
        # the secondary columns stay NaN instead of emitting an infinite statistic.
        t_result = stats.ttest_rel(treatment_values, control_values)
        row["t_statistic"] = float(t_result.statistic)
        row["p_value_secondary"] = float(t_result.pvalue)
    return row


def analyze_long(
    frame: pd.DataFrame,
    *,
    treatment: str,
    control: str = "baseline",
    metrics: Sequence[str] | None = None,
    unit: str = "group",
    primary_test: str = "wilcoxon",
    alternative: str = "two-sided",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
    distance_restriction: str = "both_nonempty",
) -> pd.DataFrame:
    """Run the paired analysis for every metric and return a long-format table.

    Positive ``delta_mean`` always means ``treatment - control``; the ``preferred``
    column translates that into "which method is better" using the metric's
    direction.  ``p_value_adjusted`` is the Holm-corrected primary p-value.

    ``distance_restriction="both_nonempty"`` (default) restricts distance metrics
    (``hd``, ``hd95``, ``hd95_pooled``, ``assd``) to cases that are non-empty under
    both methods, so a missed lesion cannot silently buy a better distance.  The
    number of contributing cases is reported as ``n_cases_used`` per comparison;
    use :func:`analyze_sensitivity` for the all-cases penalised variant.
    """
    if distance_restriction not in {"both_nonempty", "all"}:
        raise ValueError(
            f"distance_restriction must be 'both_nonempty' or 'all', got {distance_restriction!r}"
        )
    if metrics is None:
        metrics = sorted(frame["metric"].dropna().unique())
    metrics = list(metrics)
    if not metrics:
        raise ValueError("No metrics to analyse")

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        restrict = (
            metric in DISTANCE_RESTRICTED_METRICS and distance_restriction == "both_nonempty"
        )
        pairs, n_cases_used = _paired_frame(
            frame,
            metric,
            treatment,
            control,
            unit=unit,
            require_both_nonempty=restrict,
        )
        row = paired_comparison(
            pairs[control],
            pairs[treatment],
            primary_test=primary_test,
            alternative=alternative,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
        direction = metric_direction(metric)
        row.update(
            {
                "metric": metric,
                "metric_direction": direction,
                "treatment": treatment,
                "control": control,
                "unit": unit,
                "n_units": int(len(pairs)),
                "restriction": "both_nonempty" if restrict else "all_cases",
                "n_cases_used": int(n_cases_used),
                "preferred": _preferred_method(row["delta_mean"], direction, treatment, control),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if correction == "holm":
        result["p_value_adjusted"] = holm_adjust(result["p_value"].to_numpy(dtype=float))
    elif correction in (None, "none"):
        result["p_value_adjusted"] = result["p_value"]
    else:
        raise ValueError(f"Unknown correction {correction!r}; expected 'holm' or 'none'")

    column_order = [
        "metric",
        "metric_direction",
        "treatment",
        "control",
        "unit",
        "n_units",
        "restriction",
        "n_cases_used",
        "n_pairs",
        "n_dropped",
        "mean_control",
        "mean_treatment",
        "delta_mean",
        "delta_median",
        "ci_low",
        "ci_high",
        "effect_size",
        "effect_size_name",
        "statistic",
        "test",
        "p_value",
        "p_value_adjusted",
        "preferred",
        "alternative",
        "t_statistic",
        "p_value_secondary",
        "secondary_test",
    ]
    return result[column_order]


def analyze_sensitivity(
    frame: pd.DataFrame,
    *,
    treatment: str,
    control: str = "baseline",
    metrics: Sequence[str] | None = None,
    unit: str = "group",
    primary_test: str = "wilcoxon",
    alternative: str = "two-sided",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
) -> pd.DataFrame:
    """All-cases comparison of distance metrics with empty predictions penalised.

    Uses the ``<metric>_penalized`` columns written by the evaluation runner (empty
    prediction -> image diagonal), so no case is dropped.  This is the secondary
    sensitivity analysis next to the primary both-non-empty comparison.
    """
    available = set(frame["metric"].dropna().unique())
    if metrics is None:
        metrics = [
            metric
            for metric in DISTANCE_RESTRICTED_METRICS
            if f"{metric}_penalized" in available
        ]
        if not metrics:
            raise ValueError(
                "Frame contains no penalised distance metrics; re-run evaluation with "
                "evaluation.emit_penalized_metrics: true"
            )
    penalized_metrics = [f"{metric}_penalized" for metric in metrics]
    missing = [metric for metric in penalized_metrics if metric not in available]
    if missing:
        raise ValueError(
            f"Missing penalised metric(s) {missing}; re-run evaluation with "
            "evaluation.emit_penalized_metrics: true"
        )

    table = analyze_long(
        frame,
        treatment=treatment,
        control=control,
        metrics=penalized_metrics,
        unit=unit,
        primary_test=primary_test,
        alternative=alternative,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        correction=correction,
        distance_restriction="all",
    )
    table["metric"] = table["metric"].str.replace("_penalized", "", regex=False)
    table["metric_direction"] = table["metric"].map(metric_direction)
    table["preferred"] = [
        _preferred_method(delta, direction, treatment, control)
        for delta, direction in zip(table["delta_mean"], table["metric_direction"])
    ]
    table["variant"] = "penalized_all_cases"
    table["restriction"] = "all_cases_penalized"
    return table


def pool_folds(
    frames: Sequence[pd.DataFrame],
    *,
    fold_column: str = "fold",
    case_column: str = "case_id",
    group_column: str = "group_id",
) -> pd.DataFrame:
    """Concatenate per-fold per-case results into one pooled long-format frame.

    Validates the two properties that make cross-fold pooling legitimate:

    * each case is evaluated in exactly one fold (every frame is tested once);
    * each group (video/patient) appears in exactly one fold, so video-level
      aggregation over the pooled data has one independent value per video.
    """
    if not frames:
        raise ValueError("pool_folds needs at least one per-case frame")

    prepared: list[pd.DataFrame] = []
    for index, frame in enumerate(frames):
        missing = [
            column
            for column in (case_column, group_column, "method", "metric", "value")
            if column not in frame
        ]
        if missing:
            raise ValueError(f"Frame {index} is missing column(s) {missing}")
        frame = frame.copy()
        if fold_column not in frame:
            frame[fold_column] = 0
        prepared.append(frame)

    pooled = pd.concat(prepared, ignore_index=True)
    pooled[fold_column] = pooled[fold_column].astype(int)

    per_case = pooled[[case_column, fold_column]].drop_duplicates()
    duplicated_cases = per_case.loc[per_case[case_column].duplicated(), case_column].unique()
    if len(duplicated_cases):
        preview = ", ".join(str(case) for case in duplicated_cases[:5])
        raise ValueError(
            f"{len(duplicated_cases)} case(s) appear in more than one fold (first: {preview}); "
            "pooling requires each case to be tested exactly once"
        )

    per_group = pooled[[group_column, fold_column]].drop_duplicates()
    duplicated_groups = per_group.loc[per_group[group_column].duplicated(), group_column].unique()
    if len(duplicated_groups):
        preview = ", ".join(str(group) for group in duplicated_groups[:5])
        raise ValueError(
            f"{len(duplicated_groups)} group(s) appear in more than one fold (first: {preview}); "
            "each video/patient must be a test unit exactly once"
        )
    return pooled


def analyze_pooled_folds(
    frames: Sequence[pd.DataFrame],
    *,
    treatment: str,
    control: str = "baseline",
    metrics: Sequence[str] | None = None,
    unit: str = "group",
    primary_test: str = "wilcoxon",
    alternative: str = "two-sided",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
) -> pd.DataFrame:
    """Test on per-case results pooled over all folds, aggregated per video.

    With ``unit="group"`` every video contributes exactly one paired value
    (mean over its frames), so the sample size is the number of videos rather than
    the number of frames, and no video is counted twice.
    """
    pooled = pool_folds(frames)
    table = analyze_long(
        pooled,
        treatment=treatment,
        control=control,
        metrics=metrics,
        unit=unit,
        primary_test=primary_test,
        alternative=alternative,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        correction=correction,
    )
    table.insert(5, "n_folds", int(pooled["fold"].nunique()))
    table.insert(6, "n_groups_pooled", int(pooled["group_id"].nunique()))
    table.insert(7, "n_cases_pooled", int(pooled["case_id"].nunique()))
    return table




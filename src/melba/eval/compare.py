"""Cross-model comparison of two experiments on the same folds.

``scripts/aggregate.py`` describes one grid; this module compares two (e.g. the
ResNet-50 UNet and SwinUNETR runs of the same 5x3 protocol):

* one side-by-side table of the headline methods and metrics;
* the same paired tests, missed detections and tail/failure rates per model;
* the **residual-error test**: is the TTA gain larger where the single-view prediction
  is worse?  Correlating the baseline error with ``treatment - control`` is biased
  towards a negative rho by regression to the mean, so the primary tests are

  - **Oldham's method**: ``Spearman((control + treatment) / 2, treatment - control)``,
    which is unbiased when the two methods have equal variance;
  - a **paired-swap permutation null**: within each case control and treatment are
    exchanged, video by video, and the statistic is recomputed (the null hypothesis is
    exchangeability of the two methods);

  and the raw ``Spearman(control, delta)`` is kept as a secondary, explicitly biased
  column.  Every CI resamples whole videos (frames of a video are not independent);
* the **between-model gain difference**: per video, is the TTA gain of one model
  different from the other's (Wilcoxon + paired bootstrap CI on the difference)?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from melba.eval.aggregate import seed_averaged_frame, string_ids
from melba.eval.report import CORE_METRICS, TailThresholds, spearman, tail_table
from melba.stats import (
    analyze_detection,
    analyze_long,
    bootstrap_ci_mean,
    metric_direction,
    pool_folds,
)

__all__ = [
    "METHODS",
    "ModelRuns",
    "TABLE_METRICS",
    "cluster_bootstrap_correlation",
    "deltas_table",
    "gain_difference_table",
    "load_model",
    "method_table",
    "missed_detection_table",
    "paired_swap_null",
    "residual_error_table",
    "tail_table_for_models",
]

#: methods shown side by side
METHODS: tuple[str, ...] = ("baseline", "tta4", "tta8", "ensemble3")

#: metrics shown side by side
TABLE_METRICS: tuple[str, ...] = ("dice", "boundary_f1", "hd95")


@dataclass(frozen=True)
class ModelRuns:
    """Per-case records of one experiment (all folds/seeds of one grid)."""

    name: str
    frames_by_seed: Mapping[int, Sequence[pd.DataFrame]]
    ensemble_frame: pd.DataFrame | None = None

    def means(self) -> pd.DataFrame:
        """Seed-averaged long-format frame (one value per case/method/metric)."""
        return seed_averaged_frame(dict(self.frames_by_seed))

    def combined(self) -> pd.DataFrame:
        """Seed-averaged frame plus the ensemble rows (case ids normalised to ``str``)."""
        frame = string_ids(self.means().drop(columns=["n_seeds"], errors="ignore"))
        if self.ensemble_frame is not None and not self.ensemble_frame.empty:
            frame = pd.concat([frame, string_ids(self.ensemble_frame)], ignore_index=True)
        return frame

    def pooled(self) -> pd.DataFrame:
        """All folds pooled; every case and video appears exactly once."""
        return pool_folds([self.combined()])


def load_model(
    config: Any,
    *,
    name: str | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    folds: Sequence[int] | None = None,
    ensemble: bool = True,
    ground_truth: Mapping[int, Mapping[str, Mapping[str, Any]]] | None = None,
) -> ModelRuns:
    """Load one experiment's per-case results (and its ``ensemble3`` rows).

    ``ground_truth`` can be loaded once and shared by both models: the folds and the
    reference masks are identical for every model trained on the same split files.
    """
    from melba.eval.aggregate import discover_runs, ensemble_frame, frames_by_seed

    runs = discover_runs(config, folds=folds, seeds=seeds)
    if not runs:
        raise ValueError(
            f"no per-case results found for {config.get('experiment.name')}; run its grid first"
        )
    frames = frames_by_seed(runs)
    ensembles: pd.DataFrame | None = None
    if ensemble:
        ensembles = ensemble_frame(
            config,
            runs,
            name="ensemble3",
            source_method="baseline",
            ground_truth=ground_truth,
        )
    return ModelRuns(
        name=str(name or config.get("experiment.name", "model")),
        frames_by_seed=frames,
        ensemble_frame=ensembles,
    )


def method_table(
    models: Sequence[ModelRuns],
    *,
    methods: Sequence[str] = METHODS,
    metrics: Sequence[str] = TABLE_METRICS,
    unit: str = "group_id",
) -> pd.DataFrame:
    """Side-by-side mean ± std over the per-video values (one value per video)."""
    rows: list[dict[str, Any]] = []
    for model in models:
        pooled = model.pooled()
        for method in methods:
            for metric in metrics:
                subset = pooled.loc[(pooled["method"] == method) & (pooled["metric"] == metric)]
                if subset.empty:
                    continue
                per_unit = subset.groupby(unit)["value"].mean()
                rows.append(
                    {
                        "model": model.name,
                        "method": method,
                        "metric": metric,
                        "n_cases": int(subset["case_id"].nunique()),
                        "n_units": int(per_unit.size),
                        "mean": float(per_unit.mean()),
                        "std": float(per_unit.std(ddof=1)) if per_unit.size > 1 else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def deltas_table(
    models: Sequence[ModelRuns],
    *,
    treatments: Sequence[str] = ("tta4", "tta8", "ensemble3"),
    control: str = "baseline",
    metrics: Sequence[str] = CORE_METRICS,
    unit: str = "group",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
) -> pd.DataFrame:
    """Paired tests vs the control for every model (same statistics as the grid report)."""
    rows: list[pd.DataFrame] = []
    for model in models:
        pooled = model.pooled()
        available = set(pooled["method"].unique())
        available_metrics = [metric for metric in metrics if metric in set(pooled["metric"].unique())]
        for treatment in treatments:
            if treatment not in available or control not in available or not available_metrics:
                continue
            table = analyze_long(
                pooled,
                treatment=treatment,
                control=control,
                metrics=available_metrics,
                unit=unit,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
                correction=correction,
            )
            table.insert(0, "model", model.name)
            rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def missed_detection_table(
    models: Sequence[ModelRuns],
    *,
    methods: Sequence[str] = ("baseline", "tta4", "tta8", "ensemble3"),
    control: str = "baseline",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
) -> pd.DataFrame:
    """Missed-detection rates, transitions and exact McNemar tests, per model."""
    rows: list[pd.DataFrame] = []
    for model in models:
        pooled = model.pooled()
        available = [method for method in methods if method in set(pooled["method"].unique())]
        table = analyze_detection(
            pooled,
            control=control,
            methods=available,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            correction=correction,
        )
        table.insert(0, "model", model.name)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def tail_table_for_models(
    models: Sequence[ModelRuns],
    *,
    control: str = "baseline",
    thresholds: TailThresholds = TailThresholds(),
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """HD95 percentiles and failure rates per model (delta vs that model's own baseline)."""
    rows: list[pd.DataFrame] = []
    for model in models:
        table = tail_table(
            model.combined(),
            control=control,
            thresholds=thresholds,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
        table.insert(0, "model", model.name)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()



def _paired_wide(
    frame: pd.DataFrame, *, metric: str, control: str, treatment: str
) -> pd.DataFrame:
    """One row per case with the control and treatment values of ``metric``."""
    subset = frame.loc[frame["metric"] == metric, ["case_id", "group_id", "method", "value"]]
    if subset.empty:
        return pd.DataFrame()
    wide = subset.pivot(index="case_id", columns="method", values="value")
    if control not in wide.columns or treatment not in wide.columns:
        return pd.DataFrame()
    groups = subset.drop_duplicates("case_id").set_index("case_id")["group_id"]
    wide["group_id"] = groups.reindex(wide.index)
    return wide[[control, treatment, "group_id"]].reset_index()


def cluster_bootstrap_correlation(
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence[Any],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Spearman rho with a **video-cluster** bootstrap CI and a bootstrap p-value.

    Videos are resampled as blocks, so frames of one video (and, in the pooled
    analysis, the two models' predictions for the same frame) stay together.
    """
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    group_values = np.asarray(list(groups))
    keep = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values, group_values = x_values[keep], y_values[keep], group_values[keep]
    point = spearman(x_values, y_values)[0]
    unique = np.unique(group_values)
    positions = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    for _ in range(int(n_bootstrap)):
        pick = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([positions[unique[index]] for index in pick])
        rho = spearman(x_values[rows], y_values[rows])[0]
        if np.isfinite(rho):
            draws.append(float(rho))
    result: dict[str, float] = {
        "n_cases": int(x_values.size),
        "n_clusters": int(len(unique)),
        "spearman_rho": float(point),
        "spearman_p": float(spearman(x_values, y_values)[1]),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_bootstrap": float("nan"),
    }
    if len(draws) >= 10:
        distribution = np.asarray(draws)
        alpha = (1.0 - float(ci)) / 2.0
        low, high = np.percentile(distribution, [100 * alpha, 100 * (1 - alpha)])
        p_bootstrap = 2.0 * min(
            float(np.mean(distribution <= 0.0)), float(np.mean(distribution >= 0.0))
        )
        result.update(
            {
                "ci_low": float(low),
                "ci_high": float(high),
                "p_bootstrap": float(min(1.0, p_bootstrap)),
            }
        )
    return result


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho as the Pearson correlation of the (mid-)ranks, NaN-safe and fast."""
    if x.size < 3:
        return float("nan")
    x_ranks = scipy_stats.rankdata(x)
    y_ranks = scipy_stats.rankdata(y)
    x_centred = x_ranks - x_ranks.mean()
    y_centred = y_ranks - y_ranks.mean()
    denominator = float(np.sqrt((x_centred @ x_centred) * (y_centred @ y_centred)))
    if denominator == 0.0:
        return float("nan")
    return float((x_centred @ y_centred) / denominator)


def paired_swap_null(
    control: Sequence[float],
    treatment: Sequence[float],
    groups: Sequence[Any],
    *,
    n_permutations: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Permutation null of both residual-error statistics under a **paired swap**.

    For every permutation each *video block* is swapped as a whole (all of its cases
    exchange control and treatment, so the within-video dependence is preserved),
    ``delta = treatment - control`` flips sign and both statistics are recomputed:

    * ``naive``: ``Spearman(control, delta)`` (biased by regression to the mean);
    * ``oldham``: ``Spearman((control + treatment) / 2, delta)``, whose x-axis does not
      move under the swap and is therefore centred under the null when the two methods
      are exchangeable.

    The p-value is two-sided and compares ``|rho_perm| >= |rho_observed|``.
    """
    control_values = np.asarray(list(control), dtype=float)
    treatment_values = np.asarray(list(treatment), dtype=float)
    group_values = np.asarray(list(groups))
    keep = np.isfinite(control_values) & np.isfinite(treatment_values)
    control_values = control_values[keep]
    treatment_values = treatment_values[keep]
    group_values = group_values[keep]
    delta = treatment_values - control_values
    mean_pair = (control_values + treatment_values) / 2.0
    result: dict[str, float] = {
        "n_permutations": int(n_permutations),
        "n_blocks": int(np.unique(group_values).size),
        "naive_observed": _rank_correlation(control_values, delta),
        "naive_null_mean": float("nan"),
        "naive_p_permutation": float("nan"),
        "oldham_observed": _rank_correlation(mean_pair, delta),
        "oldham_null_mean": float("nan"),
        "oldham_p_permutation": float("nan"),
    }
    unique, block_index = np.unique(group_values, return_inverse=True)
    if unique.size < 2 or control_values.size < 3 or int(n_permutations) < 1:
        return result
    rng = np.random.default_rng(int(seed))
    naive_draws = np.empty(int(n_permutations), dtype=float)
    oldham_draws = np.empty(int(n_permutations), dtype=float)
    for index in range(int(n_permutations)):
        flip = (rng.random(unique.size) < 0.5)[block_index]
        swapped_control = np.where(flip, treatment_values, control_values)
        swapped_delta = np.where(flip, -delta, delta)
        naive_draws[index] = _rank_correlation(swapped_control, swapped_delta)
        oldham_draws[index] = _rank_correlation(mean_pair, swapped_delta)
    for prefix, draws, observed in (
        ("naive", naive_draws, result["naive_observed"]),
        ("oldham", oldham_draws, result["oldham_observed"]),
    ):
        finite = draws[np.isfinite(draws)]
        if finite.size == 0:
            continue
        result[f"{prefix}_null_mean"] = float(finite.mean())
        if np.isfinite(observed):
            result[f"{prefix}_p_permutation"] = float(
                np.mean(np.abs(finite) >= abs(observed))
            )
    return result


def _residual_row(
    *,
    scope: str,
    metric: str,
    control: pd.Series | np.ndarray,
    treatment: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    n_bootstrap: int,
    n_permutations: int,
    ci: float,
    seed: int,
) -> dict[str, Any]:
    """One row of the residual-error table: naive (secondary), Oldham and both nulls."""
    control_values = np.asarray(control, dtype=float)
    treatment_values = np.asarray(treatment, dtype=float)
    group_values = np.asarray(groups)
    delta = treatment_values - control_values
    mean_pair = (control_values + treatment_values) / 2.0
    keep = np.isfinite(control_values) & np.isfinite(delta)
    control_values, treatment_values = control_values[keep], treatment_values[keep]
    delta, mean_pair, group_values = delta[keep], mean_pair[keep], group_values[keep]
    naive = cluster_bootstrap_correlation(
        control_values, delta, group_values, n_bootstrap=n_bootstrap, ci=ci, seed=seed
    )
    oldham = cluster_bootstrap_correlation(
        mean_pair, delta, group_values, n_bootstrap=n_bootstrap, ci=ci, seed=seed
    )
    permutation = paired_swap_null(
        control_values,
        treatment_values,
        group_values,
        n_permutations=n_permutations,
        seed=seed,
    )
    return {
        "scope": scope,
        "metric": metric,
        "restriction": "all_cases" if metric == "dice" else "both_nonempty",
        "n_cases": int(delta.size),
        "n_clusters": int(naive["n_clusters"]),
        "delta_mean": float(delta.mean()) if delta.size else float("nan"),
        # secondary: biased towards a negative rho by regression to the mean
        "rho_naive": naive["spearman_rho"],
        "p_naive": naive["spearman_p"],
        "ci_low_naive": naive["ci_low"],
        "ci_high_naive": naive["ci_high"],
        "p_bootstrap_naive": naive["p_bootstrap"],
        "null_mean_naive": permutation["naive_null_mean"],
        "p_permutation_naive": permutation["naive_p_permutation"],
        # primary: Oldham's regression-to-the-mean correction
        "rho_oldham": oldham["spearman_rho"],
        "p_oldham": oldham["spearman_p"],
        "ci_low_oldham": oldham["ci_low"],
        "ci_high_oldham": oldham["ci_high"],
        "p_bootstrap_oldham": oldham["p_bootstrap"],
        "null_mean_oldham": permutation["oldham_null_mean"],
        "p_permutation_oldham": permutation["oldham_p_permutation"],
        "n_permutations": permutation["n_permutations"],
    }


def residual_error_table(
    models: Sequence[ModelRuns],
    *,
    treatment: str = "tta4",
    control: str = "baseline",
    metrics: Sequence[str] = ("hd95", "dice"),
    n_bootstrap: int = 2000,
    n_permutations: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Residual-error test: does TTA help more where the single-view prediction is worse?

    ``rho_naive`` correlates the control value with ``treatment - control``; that rho is
    pulled negative by regression to the mean even when TTA has no effect, so it is
    reported as a **secondary** column.  The primary tests are ``rho_oldham``
    (Oldham's mean-vs-change correlation, with the same video-cluster bootstrap CI) and
    the paired-swap permutation null (``null_mean_*`` / ``p_permutation_*``).

    Sign convention: ``delta`` is always ``treatment - control`` in the metric's own
    units, so for HD95 *and* Dice a **negative** rho means the gain is larger where the
    control was worse (HD95 improves downwards, Dice upwards).  Rows are per model plus
    one pooled row over all models.
    """
    rows: list[dict[str, Any]] = []
    pooled_parts: list[pd.DataFrame] = []
    for model in models:
        combined = model.combined()
        for metric in metrics:
            wide = _paired_wide(combined, metric=metric, control=control, treatment=treatment)
            if wide.empty:
                continue
            keep = np.isfinite(wide[control]) & np.isfinite(wide[treatment])
            if int(keep.sum()) < 3:
                continue
            subset = wide.loc[keep]
            rows.append(
                _residual_row(
                    scope=f"model:{model.name}",
                    metric=metric,
                    control=subset[control],
                    treatment=subset[treatment],
                    groups=subset["group_id"],
                    n_bootstrap=n_bootstrap,
                    n_permutations=n_permutations,
                    ci=ci,
                    seed=seed,
                )
            )
            pooled_parts.append(
                pd.DataFrame(
                    {
                        "metric": metric,
                        "control": subset[control].to_numpy(),
                        "treatment": subset[treatment].to_numpy(),
                        "group_id": subset["group_id"].to_numpy(),
                    }
                )
            )
    if pooled_parts and len(models) > 1:
        pooled = pd.concat(pooled_parts, ignore_index=True)
        for metric in metrics:
            subset = pooled.loc[pooled["metric"] == metric]
            if len(subset) < 3:
                continue
            rows.append(
                _residual_row(
                    scope="pooled_models",
                    metric=metric,
                    control=subset["control"],
                    treatment=subset["treatment"],
                    groups=subset["group_id"],
                    n_bootstrap=n_bootstrap,
                    n_permutations=n_permutations,
                    ci=ci,
                    seed=seed,
                )
            )
    return pd.DataFrame(rows)


def gain_difference_table(
    models: Sequence[ModelRuns],
    *,
    treatment: str = "tta4",
    control: str = "baseline",
    metrics: Sequence[str] = ("dice", "hd95"),
    unit: str = "group_id",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Between-model test: is the video-level TTA gain of one model larger than another's?

    The gain is ``treatment - control`` in the metric's own units, averaged per video
    inside each model; the two models are then paired **by video** and the difference of
    gains is tested with a Wilcoxon signed-rank test and a paired bootstrap CI of the
    mean difference (videos resampled as blocks).  The first model is the reference, so
    with the UNet and SwinUNETR configs the table answers "is the SwinUNETR TTA gain
    larger than the UNet's?".

    ``delta_gain`` is ``compared - reference``; ``favours`` names the model with the
    larger *gain* (not the better score: a negative HD95 gain can still be an
    improvement), and ``better_model`` names the model with the better mean score.
    """
    if len(models) < 2:
        return pd.DataFrame()
    reference = models[0]
    rows: list[dict[str, Any]] = []
    for model in models[1:]:
        for metric in metrics:
            gains: dict[str, pd.Series] = {}
            scores: dict[str, pd.Series] = {}
            for candidate in (reference, model):
                wide = _paired_wide(
                    candidate.pooled(), metric=metric, control=control, treatment=treatment
                )
                if wide.empty:
                    gains, scores = {}, {}
                    break
                wide = wide.assign(
                    gain=wide[treatment] - wide[control],
                    treatment_value=wide[treatment],
                )
                gains[candidate.name] = wide.groupby(unit)["gain"].mean()
                scores[candidate.name] = wide.groupby(unit)["treatment_value"].mean()
            if not gains:
                continue
            paired = pd.concat(gains, axis=1, join="inner").dropna()
            paired_scores = pd.concat(scores, axis=1, join="inner").dropna()
            if len(paired) < 3:
                continue
            difference = (paired[model.name] - paired[reference.name]).to_numpy(dtype=float)
            mean_difference = float(np.mean(difference))
            low, high = bootstrap_ci_mean(
                difference, n_bootstrap=n_bootstrap, ci=ci, seed=seed
            )
            if np.all(difference == 0.0):
                p_value = 1.0
            else:
                p_value = float(
                    scipy_stats.wilcoxon(
                        difference, zero_method="wilcox", alternative="two-sided"
                    ).pvalue
                )
            direction = metric_direction(metric)
            gain_reference = float(paired[reference.name].mean())
            gain_compared = float(paired[model.name].mean())
            if abs(mean_difference) < 1e-12:
                favours = "tie"
            elif (mean_difference > 0) == (direction == "higher"):
                favours = model.name
            else:
                favours = reference.name
            score_reference = float(paired_scores[reference.name].mean())
            score_compared = float(paired_scores[model.name].mean())
            better_model = (
                model.name
                if (score_compared > score_reference) == (direction == "higher")
                else reference.name
            )
            rows.append(
                {
                    "metric": metric,
                    "reference": reference.name,
                    "compared": model.name,
                    "treatment": treatment,
                    "control": control,
                    "metric_direction": direction,
                    "n_units": int(len(paired)),
                    "gain_reference": gain_reference,
                    "gain_compared": gain_compared,
                    "delta_gain": mean_difference,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "p_value": p_value,
                    "favours": favours,
                    "score_reference": score_reference,
                    "score_compared": score_compared,
                    "better_model": better_model,
                }
            )
    return pd.DataFrame(rows)


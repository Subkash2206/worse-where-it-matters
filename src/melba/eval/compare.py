"""Cross-model comparison of two experiments on the same folds.

``scripts/aggregate.py`` describes one grid; this module compares two (e.g. the
ResNet-50 UNet and SwinUNETR runs of the same 5x3 protocol):

* one side-by-side table of the headline methods and metrics;
* the same paired tests, missed detections and tail/failure rates per model;
* the **residual-error test**: is the TTA gain larger where the single-view prediction
  is worse?  For every case ``Spearman(baseline error, delta)`` is computed, once per
  model and once pooled over both models, with a video-cluster bootstrap CI on rho
  (frames of a video, and the two models' predictions for the same frame, are not
  independent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from melba.eval.aggregate import seed_averaged_frame, string_ids
from melba.eval.report import CORE_METRICS, TailThresholds, spearman, tail_table
from melba.stats import analyze_detection, analyze_long, pool_folds

__all__ = [
    "METHODS",
    "ModelRuns",
    "TABLE_METRICS",
    "cluster_bootstrap_correlation",
    "deltas_table",
    "load_model",
    "method_table",
    "missed_detection_table",
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


def residual_error_table(
    models: Sequence[ModelRuns],
    *,
    treatment: str = "tta4",
    control: str = "baseline",
    metrics: Sequence[str] = ("hd95", "dice"),
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Residual-error test: does TTA help more where the baseline is worse?

    For each case the baseline error and the paired delta (treatment − control) are
    correlated (Spearman).  A **negative** rho for HD95 means larger TTA improvements on
    cases the baseline already handled badly; a **positive** rho for Dice means larger
    improvements where baseline Dice was low.  Rows are per model plus one pooled row
    over both models (cluster bootstrap by video).
    """
    rows: list[dict[str, Any]] = []
    pooled_parts: list[pd.DataFrame] = []
    for model in models:
        combined = model.combined()
        for metric in metrics:
            wide = _paired_wide(combined, metric=metric, control=control, treatment=treatment)
            if wide.empty:
                continue
            delta = wide[treatment] - wide[control]
            keep = np.isfinite(wide[control]) & np.isfinite(delta)
            if int(keep.sum()) < 3:
                continue
            result = cluster_bootstrap_correlation(
                wide.loc[keep, control],
                delta[keep],
                wide.loc[keep, "group_id"],
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
            )
            rows.append(
                {
                    "scope": f"model:{model.name}",
                    "metric": metric,
                    "restriction": "both_nonempty" if metric != "dice" else "all_cases",
                    "delta_mean": float(delta[keep].mean()),
                    **result,
                }
            )
            pooled_parts.append(
                pd.DataFrame(
                    {
                        "metric": metric,
                        "baseline_value": wide.loc[keep, control].to_numpy(),
                        "delta": delta[keep].to_numpy(),
                        "group_id": wide.loc[keep, "group_id"].to_numpy(),
                    }
                )
            )
    if pooled_parts:
        pooled = pd.concat(pooled_parts, ignore_index=True)
        for metric in metrics:
            subset = pooled.loc[pooled["metric"] == metric]
            if len(subset) < 3:
                continue
            result = cluster_bootstrap_correlation(
                subset["baseline_value"],
                subset["delta"],
                subset["group_id"],
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
            )
            rows.append(
                {
                    "scope": "pooled_models",
                    "metric": metric,
                    "restriction": "both_nonempty" if metric != "dice" else "all_cases",
                    "delta_mean": float(subset["delta"].mean()),
                    **result,
                }
            )
    return pd.DataFrame(rows)


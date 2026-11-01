"""Report tables for a completed cross-validation grid.

The tables here answer questions the per-case CSVs cannot answer directly:

* a per-method summary at **frame level and video level**, pooled over all videos and
  reported as mean ± std across seeds (``pooled_method_table``);
* per-fold means, so fold-to-fold variation is visible (``per_fold_table``);
* tail behaviour: 90th/95th percentile HD95 and failure rates (``tail_table``), with a
  video-cluster bootstrap CI on the change relative to the control;
* the all-cases penalised sensitivity table (``sensitivity_table``);
* whether the TTA uncertainty descriptors actually rank errors
  (``uncertainty_table``: Spearman against HD95 / 1-Dice and AUROC for detecting
  failures).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

__all__ = [
    "CORE_METRICS",
    "DISTANCE_METRICS",
    "METRIC_LABELS",
    "TailThresholds",
    "auroc",
    "cluster_bootstrap_change",
    "markdown_table",
    "per_fold_table",
    "pooled_method_table",
    "sensitivity_table",
    "spearman",
    "tail_table",
    "uncertainty_table",
    "write_markdown",
]

#: metrics reported in the headline tables
CORE_METRICS: tuple[str, ...] = ("dice", "boundary_f1", "hd95", "hd95_pooled", "assd")

#: metrics that are undefined when a prediction is empty (empty_policy: nan)
DISTANCE_METRICS: tuple[str, ...] = ("hd", "hd95", "hd95_pooled", "assd")

METRIC_LABELS: Mapping[str, str] = {
    "dice": "Dice",
    "boundary_f1": "BF1",
    "hd95": "HD95",
    "hd95_pooled": "HD95_pooled",
    "assd": "ASSD",
}


@dataclass(frozen=True)
class TailThresholds:
    """Failure definitions for the tail analysis (CVC-ClinicDB spacing is 1 px)."""

    hd95_px: float = 50.0
    dice: float = 0.5


def _series_for_level(frame: pd.DataFrame, method: str, metric: str, level: str) -> pd.Series:
    subset = frame.loc[(frame["method"] == method) & (frame["metric"] == metric)]
    if level == "frame":
        return subset["value"].dropna()
    if level == "video":
        return subset.groupby("group_id")["value"].mean().dropna()
    raise ValueError(f"level must be 'frame' or 'video', got {level!r}")


def pooled_method_table(
    frames_by_seed_map: Mapping[int, Sequence[pd.DataFrame]],
    *,
    metrics: Sequence[str] = CORE_METRICS,
    ensemble_frame: pd.DataFrame | None = None,
    levels: Sequence[str] = ("frame", "video"),
) -> pd.DataFrame:
    """Per method/metric: pooled mean ± std across seeds, at frame and video level.

    For each seed the folds are pooled (every video is tested exactly once), the mean is
    taken over all frames (frame level) or over the per-video means (video level), and
    the mean/standard deviation across seeds is reported.  Ensembles are a single
    realisation over the seeds, so their std is undefined and reported as NaN.
    """
    from melba.stats import pool_folds

    pooled_by_seed = {seed: pool_folds(frames) for seed, frames in frames_by_seed_map.items()}
    methods = sorted(
        {str(method) for frame in pooled_by_seed.values() for method in frame["method"].unique()}
    )
    if ensemble_frame is not None and not ensemble_frame.empty:
        methods = sorted(set(methods) | {str(m) for m in ensemble_frame["method"].unique()})

    rows: list[dict[str, Any]] = []
    reference = pooled_by_seed[sorted(pooled_by_seed)[0]]
    for method in methods:
        for metric in metrics:
            subset_reference = reference.loc[
                (reference["method"] == method) & (reference["metric"] == metric)
            ]
            for level in levels:
                values: list[float] = []
                seed_columns: dict[str, float] = {}
                is_ensemble = False
                for seed, pooled in sorted(pooled_by_seed.items()):
                    series = _series_for_level(pooled, method, metric, level)
                    if series.empty:
                        continue
                    values.append(float(series.mean()))
                    seed_columns[f"mean_seed{int(seed)}"] = float(series.mean())
                if ensemble_frame is not None and not ensemble_frame.empty:
                    series = _series_for_level(ensemble_frame, method, metric, level)
                    if not series.empty:
                        values = [float(series.mean())]
                        seed_columns = {}
                        is_ensemble = True
                        subset_reference = ensemble_frame.loc[
                            (ensemble_frame["method"] == method)
                            & (ensemble_frame["metric"] == metric)
                        ]
                if not values:
                    continue
                row: dict[str, Any] = {
                    "method": method,
                    "metric": metric,
                    "level": level,
                    "n_cases": int(subset_reference["case_id"].nunique()),
                    "n_videos": int(subset_reference["group_id"].nunique()),
                    "n_seeds": 0 if is_ensemble else len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                }
                row.update(seed_columns)
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["metric", "level", "method"]).reset_index(drop=True)


def per_fold_table(
    frames_by_seed_map: Mapping[int, Sequence[pd.DataFrame]],
    *,
    method: str = "baseline",
    metrics: Sequence[str] = ("dice",),
) -> pd.DataFrame:
    """Per fold: frame-level and video-level means, averaged (with std) across seeds."""
    rows: list[dict[str, Any]] = []
    for seed, frames in sorted(frames_by_seed_map.items()):
        for frame in frames:
            fold = int(frame["fold"].iloc[0]) if "fold" in frame else 0
            for metric in metrics:
                subset = frame.loc[(frame["method"] == method) & (frame["metric"] == metric)]
                if subset.empty:
                    continue
                per_video = subset.groupby("group_id")["value"].mean()
                rows.append(
                    {
                        "fold": fold,
                        "seed": int(seed),
                        "method": method,
                        "metric": metric,
                        "n_cases": int(subset["case_id"].nunique()),
                        "n_videos": int(per_video.size),
                        "frame_mean": float(subset["value"].mean()),
                        "video_mean": float(per_video.mean()),
                    }
                )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return (
        table.groupby(["fold", "metric"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_cases=("n_cases", "max"),
            n_videos=("n_videos", "max"),
            frame_mean=("frame_mean", "mean"),
            frame_std=("frame_mean", "std"),
            video_mean=("video_mean", "mean"),
            video_std=("video_mean", "std"),
        )
        .sort_values(["metric", "fold"])
        .reset_index(drop=True)
    )



def cluster_bootstrap_change(
    treatment: np.ndarray,
    control: np.ndarray,
    groups: np.ndarray,
    statistic,
    *,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``statistic(treatment) - statistic(control)``.

    Videos (not frames) are the resampling unit, because frames of one video are not
    independent.  Both arrays are aligned case by case, so the two methods are paired
    inside every resampled draw.
    """
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        draw = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([positions[unique[position]] for position in draw])
        if np.all(~np.isfinite(treatment[rows])) or np.all(~np.isfinite(control[rows])):
            deltas[index] = np.nan
            continue
        deltas[index] = statistic(treatment[rows]) - statistic(control[rows])
    finite = deltas[np.isfinite(deltas)]
    if finite.size < max(10, int(0.05 * int(n_bootstrap))):
        return float("nan"), float("nan")
    alpha = (1.0 - float(ci)) / 2.0
    low, high = np.percentile(finite, [100 * alpha, 100 * (1 - alpha)])
    return float(low), float(high)


def _pivot_cases(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per case: ``group_id`` plus one column per method for ``metric``.

    ``DataFrame.pivot`` is used instead of ``pivot_table`` on purpose: a case whose
    distance is undefined (empty prediction) must stay in the table, otherwise it would
    silently disappear from the failure counts and the percentiles.
    """
    subset = frame.loc[frame["metric"] == metric, ["case_id", "group_id", "method", "value"]]
    wide = subset.pivot(index="case_id", columns="method", values="value")
    groups = subset.drop_duplicates("case_id").set_index("case_id")["group_id"]
    wide["group_id"] = groups.reindex(wide.index)
    return wide.reset_index()


def _nan_percentile(values: np.ndarray, percentile: float) -> float:
    """Percentile over the finite values (NaN when nothing is finite)."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def tail_table(
    frame: pd.DataFrame,
    *,
    methods: Sequence[str] | None = None,
    control: str = "baseline",
    metric: str = "hd95",
    thresholds: TailThresholds = TailThresholds(),
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Tail behaviour per method: HD95 percentiles and failure rates, with delta CIs.

    A distance metric is undefined for an empty prediction (``empty_policy: nan``), so
    such a case counts as an HD95 failure; the dice failure rate is the fraction of
    cases with ``dice < thresholds.dice``.
    """
    hd = _pivot_cases(frame, metric)
    dice_frame = _pivot_cases(frame, "dice")
    available = [column for column in hd.columns if column not in {"case_id", "group_id"}]
    selected = [str(method) for method in (methods or available) if method in available]
    if control not in available:
        raise ValueError(f"control method {control!r} is not in the frame")

    n_cases = len(hd)
    n_videos = int(hd["group_id"].nunique())

    def hd_failures(values: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(values)
        return np.where(nan_mask, 1.0, (values > thresholds.hd95_px).astype(float))

    def dice_failures(values: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(values), 1.0, (values < thresholds.dice).astype(float))

    rows: list[dict[str, Any]] = []
    for method in selected:
        values = hd[method].to_numpy(dtype=float)
        dice_values = dice_frame[method].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "method": method,
            "n_cases": n_cases,
            "n_videos": n_videos,
            "n_cases_hd_defined": int(np.isfinite(values).sum()),
            "hd95_p90": _nan_percentile(values, 90),
            "hd95_p95": _nan_percentile(values, 95),
            "hd95_median": _nan_percentile(values, 50),
            "hd95_failure_rate": float(hd_failures(values).mean()),
            "dice_failure_rate": float(dice_failures(dice_values).mean()),
        }
        if method != control:
            control_values = hd[control].to_numpy(dtype=float)
            control_dice = dice_frame[control].to_numpy(dtype=float)
            groups = hd["group_id"].to_numpy()
            for name, statistic, treatment_array, control_array in (
                (
                    "delta_hd95_p90",
                    lambda array: _nan_percentile(array, 90),
                    values,
                    control_values,
                ),
                (
                    "delta_hd95_p95",
                    lambda array: _nan_percentile(array, 95),
                    values,
                    control_values,
                ),
                (
                    "delta_hd95_failure_rate",
                    lambda array: float(np.mean(hd_failures(array))),
                    values,
                    control_values,
                ),
                (
                    "delta_dice_failure_rate",
                    lambda array: float(np.mean(dice_failures(array))),
                    dice_values,
                    control_dice,
                ),
            ):
                row[f"{name}_estimate"] = statistic(treatment_array) - statistic(control_array)
                low, high = cluster_bootstrap_change(
                    treatment_array,
                    control_array,
                    groups,
                    statistic,
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                    seed=seed,
                )
                row[f"{name}_ci_low"] = low
                row[f"{name}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)



def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation and p-value over the finite pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan"), float("nan")
    if float(np.ptp(x[mask])) == 0.0 or float(np.ptp(y[mask])) == 0.0:
        return float("nan"), float("nan")  # a constant input has no defined rank correlation
    result = scipy_stats.spearmanr(x[mask], y[mask])
    return float(result.statistic), float(result.pvalue)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney rank formula (ties averaged)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    mask = np.isfinite(scores)
    scores, labels = scores[mask], labels[mask]
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = scipy_stats.rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def sensitivity_table(
    frame: pd.DataFrame,
    *,
    control: str = "baseline",
    treatments: Sequence[str] = ("tta2", "tta4", "tta8"),
    metrics: Sequence[str] = DISTANCE_METRICS,
    unit: str = "group",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
) -> pd.DataFrame:
    """All-cases comparison using the ``*_penalized`` columns (empty prediction -> diagonal).

    This is the sensitivity analysis for survivorship bias: if a method wins here as
    well as on the both-non-empty restriction, its distance advantage is not merely an
    artefact of dropping the cases it missed.
    """
    from melba.stats import analyze_sensitivity

    available = set(frame["method"].unique())
    tables: list[pd.DataFrame] = []
    for treatment in treatments:
        if treatment not in available or control not in available:
            continue
        table = analyze_sensitivity(
            frame,
            treatment=treatment,
            control=control,
            metrics=list(metrics),
            unit=unit,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            correction=correction,
        )
        # analyze_sensitivity already records the treatment column
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def uncertainty_table(
    frame: pd.DataFrame,
    *,
    method: str = "tta4",
    predictors: Sequence[str] = ("view_disagreement", "entropy_mean_roi"),
    metric: str = "hd95",
    thresholds: TailThresholds = TailThresholds(),
) -> pd.DataFrame:
    """Do the TTA uncertainty descriptors rank errors and detect failures?

    Per case (one value per method) the predictor is correlated with the distance
    metric and with ``1 - Dice`` (Spearman), and its AUROC is computed for the failure
    label ``Dice < thresholds.dice`` or a missed detection, which is the decision the
    descriptor would actually be used for.
    """
    rows: list[dict[str, Any]] = []
    cases = _case_table(frame, method=method, metric=metric)
    if cases.empty:
        return pd.DataFrame()
    failures = (cases["dice"] < thresholds.dice) | (cases["missed"].astype(bool))
    for predictor in predictors:
        if predictor not in cases.columns:
            continue
        values = cases[predictor].to_numpy(dtype=float)
        rho_hd, p_hd = spearman(values, cases[metric].to_numpy(dtype=float))
        rho_dice, p_dice = spearman(values, 1.0 - cases["dice"].to_numpy(dtype=float))
        rows.append(
            {
                "method": method,
                "predictor": predictor,
                "n_cases": int(len(cases)),
                "spearman_vs_hd95": rho_hd,
                "spearman_vs_hd95_p": p_hd,
                "spearman_vs_one_minus_dice": rho_dice,
                "spearman_vs_one_minus_dice_p": p_dice,
                "auroc_failure": auroc(values, failures.to_numpy(dtype=bool)),
                "n_failures": int(failures.sum()),
                "failure_definition": (
                    f"dice < {thresholds.dice} or missed detection"
                ),
            }
        )
    return pd.DataFrame(rows)


def _case_table(frame: pd.DataFrame, *, method: str, metric: str) -> pd.DataFrame:
    """One row per case with the method's metrics as columns (predictors, dice, hd95)."""
    subset = frame.loc[frame["method"] == method]
    if subset.empty:
        return pd.DataFrame()
    wide = subset.pivot(index=["case_id", "group_id"], columns="metric", values="value").reset_index()
    wide["missed"] = (
        subset.drop_duplicates("case_id")
        .set_index("case_id")["missed_detection"]
        .reindex(wide["case_id"])
        .to_numpy()
    )
    if metric not in wide.columns or "dice" not in wide.columns:
        return pd.DataFrame()
    return wide



def markdown_table(frame: pd.DataFrame | None, *, digits: int = 4, max_rows: int = 60) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no extra dependency)."""
    if frame is None or frame.empty:
        return "_(no rows)_"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.head(max_rows).iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(float(value)) else f"{float(value):.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(
    path: str,
    *,
    title: str,
    preamble: str = "",
    sections: Sequence[tuple[str, Any, str]] = (),
) -> str:
    """Write a markdown report: title, optional preamble and (heading, table, note) sections."""
    from pathlib import Path

    chunks: list[str] = [f"# {title}", ""]
    if preamble:
        chunks += [preamble.strip(), ""]
    for heading, body, note in sections:
        chunks += [f"## {heading}", ""]
        if isinstance(body, pd.DataFrame):
            chunks += [markdown_table(body), ""]
        elif body:
            chunks += [str(body).strip(), ""]
        if note:
            chunks += [note.strip(), ""]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    return str(target)


"""Audit of the TTA uncertainty descriptors as failure detectors.

``melba.metrics.uncertainty`` defines the ROI descriptors as the mean of the binary
entropy (or the across-view variance) **inside the predicted foreground dilated by
``boundary_band_px`` pixels**.  An empty prediction therefore has an *empty* ROI and
both ``entropy_mean_roi`` and ``variance_mean_roi`` are written as NaN, and a tiny
prediction gives a ROI of a few pixels whose entropy can be near zero even though the
case is a failure.  This module quantifies that behaviour instead of letting the NaNs
disappear from a correlation/AUROC:

* ROI diagnostics for the failures (empty / near-empty / size buckets, values);
* AUROC of every descriptor under three failure definitions (all, non-missed, missed
  only), with an explicit count of the cases actually used and a NaN bound;
* trivial baselines for the same task (predicted area, 1 - confidence, image entropy,
  single-view baseline entropy) and the seed-ensemble entropy/variance;
* video-cluster bootstrap CIs on every AUROC and paired tests between descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from melba.metrics.uncertainty import foreground_band, predictive_entropy_map

__all__ = [
    "DEFAULT_BAND_PX",
    "NEAR_EMPTY_PX",
    "MapStats",
    "auroc_ci",
    "auroc_difference",
    "auroc_nan_bounds",
    "map_statistics",
    "roi_diagnostics",
    "run_uncertainty_audit",
]

#: same band as the runner (``evaluation.uncertainty_boundary_band_px``)
DEFAULT_BAND_PX = 3
#: a ROI at or below this many pixels counts as "near empty" (~0.09 % of 288x384)
NEAR_EMPTY_PX = 100


@dataclass(frozen=True)
class MapStats:
    """Geometry/uncertainty facts derived from one (seed-averaged) probability map."""

    area_px: int
    roi_px: int
    entropy_roi: float
    entropy_image: float
    confidence: float
    variance_roi: float
    variance_image: float
    ambiguous_fraction: float


def map_statistics(
    mean_map: np.ndarray,
    seed_maps: Sequence[np.ndarray] | None = None,
    *,
    threshold: float = 0.5,
    band_px: int = DEFAULT_BAND_PX,
) -> MapStats:
    """Statistics of a mean probability map, using the runner's exact ROI definition."""
    probability = np.asarray(mean_map, dtype=np.float64)
    if probability.ndim != 2:
        raise ValueError(f"expected a 2-D probability map, got {probability.shape}")
    mask = probability > float(threshold)
    roi = foreground_band(mask, band_px)
    entropy = predictive_entropy_map(probability)
    variance_roi = float("nan")
    variance_image = float("nan")
    if seed_maps is not None and len(seed_maps) > 1:
        stack = np.stack([np.asarray(item, dtype=np.float64) for item in seed_maps], axis=0)
        variance = stack.var(axis=0, ddof=0)
        variance_image = float(variance.mean())
        if roi.any():
            variance_roi = float(variance[roi].mean())
    return MapStats(
        area_px=int(mask.sum()),
        roi_px=int(roi.sum()),
        entropy_roi=float(entropy[roi].mean()) if roi.any() else float("nan"),
        entropy_image=float(entropy.mean()),
        confidence=float(probability[mask].mean()) if mask.any() else float("nan"),
        variance_roi=variance_roi,
        variance_image=variance_image,
        ambiguous_fraction=float(((probability > 0.1) & (probability < 0.9)).mean()),
    )


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = scipy_stats.rankdata(scores)
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def auroc_ci(
    scores: Iterable[float],
    labels: Iterable[Any],
    groups: Iterable[Any],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """AUROC with a video-cluster bootstrap CI (frames of a video are dependent)."""
    values = np.asarray(list(scores), dtype=float)
    flags = np.asarray(list(labels)).astype(bool)
    group_values = np.asarray(list(groups))
    keep = np.isfinite(values)
    values, flags, group_values = values[keep], flags[keep], group_values[keep]
    point = _auroc(values, flags)
    unique = np.unique(group_values)
    positions = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    for _ in range(int(n_bootstrap)):
        pick = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([positions[unique[index]] for index in pick])
        value = _auroc(values[rows], flags[rows])
        if np.isfinite(value):
            draws.append(value)
    alpha = (1.0 - float(ci)) / 2.0
    result = {
        "auroc": point,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n_used": int(values.size),
        "n_failures": int(flags.sum()),
        "n_undefined": int((~keep).sum()),
    }
    if len(draws) >= 10:
        low, high = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])
        result["ci_low"] = float(low)
        result["ci_high"] = float(high)
    return result


def auroc_nan_bounds(scores: Iterable[float], labels: Iterable[Any]) -> tuple[float, float]:
    """AUROC bounds when undefined scores count as maximally certain vs maximally uncertain."""
    values = np.asarray(list(scores), dtype=float)
    flags = np.asarray(list(labels)).astype(bool)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lowest, highest = float(finite.min()), float(finite.max())
    span = max(highest - lowest, 1.0)
    as_certain = np.where(np.isfinite(values), values, lowest - span)
    as_uncertain = np.where(np.isfinite(values), values, highest + span)
    return _auroc(as_certain, flags), _auroc(as_uncertain, flags)


def auroc_difference(
    scores_a: Iterable[float],
    scores_b: Iterable[float],
    labels: Iterable[Any],
    groups: Iterable[Any],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Paired (video-cluster) bootstrap test of ``AUROC(a) - AUROC(b)``."""
    a = np.asarray(list(scores_a), dtype=float)
    b = np.asarray(list(scores_b), dtype=float)
    flags = np.asarray(list(labels)).astype(bool)
    group_values = np.asarray(list(groups))
    keep = np.isfinite(a) & np.isfinite(b)
    a, b, flags, group_values = a[keep], b[keep], flags[keep], group_values[keep]
    delta = _auroc(a, flags) - _auroc(b, flags)
    unique = np.unique(group_values)
    positions = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    for _ in range(int(n_bootstrap)):
        pick = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([positions[unique[index]] for index in pick])
        value = _auroc(a[rows], flags[rows]) - _auroc(b[rows], flags[rows])
        if np.isfinite(value):
            draws.append(value)
    result = {
        "delta_auroc": float(delta),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": float("nan"),
        "n_used": int(a.size),
    }
    if len(draws) >= 10:
        distribution = np.asarray(draws)
        alpha = (1.0 - float(ci)) / 2.0
        low, high = np.percentile(distribution, [100 * alpha, 100 * (1 - alpha)])
        p_value = 2.0 * min(
            float(np.mean(distribution <= 0.0)), float(np.mean(distribution >= 0.0))
        )
        result.update(
            {"ci_low": float(low), "ci_high": float(high), "p_value": float(min(1.0, p_value))}
        )
    return result



def _load_maps(runs: Sequence[Any], case_id: str, method: str) -> list[np.ndarray]:
    maps = []
    for run in runs:
        path = Path(run.map_dir) / f"{case_id}__{method}__mean.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing probability map {path}")
        maps.append(np.load(path).astype(np.float32))
    return maps


def _wide(frame: pd.DataFrame, method: str, metrics: Sequence[str]) -> pd.DataFrame:
    """One row per case with the named metrics of ``method`` as columns."""
    subset = frame.loc[frame["method"] == method, ["case_id", "metric", "value"]]
    subset = subset.loc[subset["metric"].isin(metrics)]
    return subset.pivot(index="case_id", columns="metric", values="value").reset_index()


def build_case_table(
    runs: Sequence[Any],
    averaged: pd.DataFrame,
    *,
    band_px: int = DEFAULT_BAND_PX,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-case predictors (CSV descriptors + map-derived baselines) and failure labels.

    ``tta4``/``baseline`` descriptors come from the long-format rows (already averaged
    over seeds by the caller); the map-derived predictors are recomputed from the saved
    probability maps with the runner's own ROI definition.
    """
    case_metrics = (
        "dice",
        "hd95",
        "entropy_mean",
        "entropy_mean_roi",
        "variance_mean",
        "variance_mean_roi",
        "view_disagreement",
    )
    tta4 = _wide(averaged, "tta4", case_metrics).rename(
        columns={
            "entropy_mean": "tta4_entropy_image",
            "entropy_mean_roi": "tta4_entropy_roi",
            "variance_mean": "tta4_variance_image",
            "variance_mean_roi": "tta4_variance_roi",
            "view_disagreement": "tta4_view_disagreement",
        }
    )
    baseline = _wide(averaged, "baseline", ("entropy_mean", "entropy_mean_roi")).rename(
        columns={
            "entropy_mean": "baseline_entropy_image",
            "entropy_mean_roi": "baseline_entropy_roi",
        }
    )
    meta = (
        averaged.loc[
            averaged["method"] == "tta4", ["case_id", "group_id", "fold", "missed_detection"]
        ]
        .drop_duplicates("case_id")
        .rename(columns={"missed_detection": "missed"})
    )
    table = meta.merge(tta4, on="case_id", how="left").merge(baseline, on="case_id", how="left")
    table["case_id"] = table["case_id"].astype(str)
    table["fold"] = table["fold"].astype(int)
    table["missed"] = table["missed"].astype(bool)

    runs_by_fold: dict[int, list[Any]] = {}
    for run in runs:
        runs_by_fold.setdefault(int(run.fold), []).append(run)
    for fold, fold_runs in runs_by_fold.items():
        ordered = sorted(fold_runs, key=lambda item: item.seed)
        for index in table.index[table["fold"] == fold]:
            case_id = str(table.at[index, "case_id"])
            tta4_maps = _load_maps(ordered, case_id, "tta4")
            baseline_maps = _load_maps(ordered, case_id, "baseline")
            tta4_stats = map_statistics(
                np.mean(tta4_maps, axis=0), tta4_maps, threshold=threshold, band_px=band_px
            )
            ensemble_stats = map_statistics(
                np.mean(baseline_maps, axis=0),
                baseline_maps,
                threshold=threshold,
                band_px=band_px,
            )
            table.at[index, "pred_area_px"] = tta4_stats.area_px
            table.at[index, "pred_roi_px"] = tta4_stats.roi_px
            table.at[index, "one_minus_confidence"] = (
                1.0 - tta4_stats.confidence if np.isfinite(tta4_stats.confidence) else np.nan
            )
            table.at[index, "map_entropy_roi_tta4"] = tta4_stats.entropy_roi
            table.at[index, "map_variance_roi_tta4"] = tta4_stats.variance_roi
            table.at[index, "map_ambiguous_fraction_tta4"] = tta4_stats.ambiguous_fraction
            table.at[index, "ensemble3_entropy_roi"] = ensemble_stats.entropy_roi
            table.at[index, "ensemble3_variance_roi"] = ensemble_stats.variance_roi
            table.at[index, "ensemble3_entropy_image"] = ensemble_stats.entropy_image
            table.at[index, "ensemble3_variance_image"] = ensemble_stats.variance_image
            table.at[index, "ensemble3_area_px"] = ensemble_stats.area_px
    return table



def _nanmedian(values: np.ndarray) -> float:
    """Median of the finite values (NaN, without a warning, when none is finite)."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("nan")


def roi_diagnostics(table: pd.DataFrame, *, near_empty_px: int = NEAR_EMPTY_PX) -> pd.DataFrame:
    """How the ROI descriptors behave on failures: empty, near-empty and larger ROIs."""
    rows: list[dict[str, Any]] = []
    failures = table.loc[table["failure_all"]]
    cohorts = {
        "all_cases": table,
        "failures_all": failures,
        "failures_non_missed": failures.loc[~failures["missed"]],
        "failures_missed": failures.loc[failures["missed"]],
    }
    for name, cohort in cohorts.items():
        if cohort.empty:
            continue
        areas = cohort["pred_area_px"].to_numpy(dtype=float)
        entropy_roi = cohort["tta4_entropy_roi"].to_numpy(dtype=float)
        near = entropy_roi[areas <= near_empty_px]
        rows.append(
            {
                "cohort": name,
                "n": int(len(cohort)),
                "n_empty_prediction": int((areas == 0).sum()),
                "n_near_empty_le_100px": int(((areas > 0) & (areas <= near_empty_px)).sum()),
                "n_entropy_roi_nan": int(np.isnan(entropy_roi).sum()),
                "median_area_px": float(np.median(areas)),
                "median_entropy_roi": _nanmedian(entropy_roi),
                "median_entropy_roi_empty_or_near": _nanmedian(near),
            }
        )
    buckets = [
        ("empty_0px", 0, 0),
        ("tiny_1_10px", 1, 10),
        ("small_11_100px", 11, 100),
        ("medium_101_1000px", 101, 1000),
        ("large_gt_1000px", 1001, 10**9),
    ]
    for label, low, high in buckets:
        selection = failures.loc[
            (failures["pred_area_px"] >= low) & (failures["pred_area_px"] <= high)
        ]
        if selection.empty:
            continue
        entropy_roi = selection["tta4_entropy_roi"].to_numpy(dtype=float)
        rows.append(
            {
                "cohort": f"failures_{label}",
                "n": int(len(selection)),
                "n_empty_prediction": int((selection["pred_area_px"] == 0).sum()),
                "n_near_empty_le_100px": int((selection["pred_area_px"] <= near_empty_px).sum()),
                "n_entropy_roi_nan": int(np.isnan(entropy_roi).sum()),
                "median_area_px": float(np.median(selection["pred_area_px"])),
                "median_entropy_roi": _nanmedian(entropy_roi),
                "median_entropy_roi_empty_or_near": float("nan"),
            }
        )
    return pd.DataFrame(rows)


PREDICTORS: dict[str, str] = {
    "tta4_entropy_roi": "tta4: entropy in predicted ROI (primary descriptor)",
    "tta4_variance_roi": "tta4: across-view variance in predicted ROI",
    "tta4_view_disagreement": "tta4: fraction of pixels where views disagree",
    "tta4_entropy_image": "tta4: entropy over the whole image",
    "map_entropy_roi_tta4": "entropy in ROI recomputed from the seed-averaged tta4 map",
    "map_variance_roi_tta4": "across-seed variance in ROI (tta4 maps)",
    "map_ambiguous_fraction_tta4": "fraction of pixels with 0.1 < p < 0.9 (tta4)",
    "pred_area_px": "trivial: predicted foreground area in pixels",
    "one_minus_confidence": "trivial: 1 - mean probability inside the predicted mask",
    "baseline_entropy_roi": "trivial: single-view (no TTA) entropy in its own ROI",
    "baseline_entropy_image": "trivial: single-view entropy over the whole image",
    "ensemble3_entropy_roi": "ensemble3: entropy in its ROI (3 baseline maps averaged)",
    "ensemble3_variance_roi": "ensemble3: across-seed variance in its ROI",
    "ensemble3_entropy_image": "ensemble3: entropy over the whole image",
    "ensemble3_variance_image": "ensemble3: across-seed variance over the whole image",
}



def run_uncertainty_audit(
    runs: Sequence[Any],
    averaged: pd.DataFrame,
    *,
    by_seed: Mapping[int, Sequence[pd.DataFrame]] | None = None,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    band_px: int = DEFAULT_BAND_PX,
    threshold: float = 0.5,
    near_empty_px: int = NEAR_EMPTY_PX,
) -> dict[str, Any]:
    """Build every uncertainty-audit table (see the module docstring)."""
    from melba.eval.report import spearman

    table = build_case_table(runs, averaged, band_px=band_px, threshold=threshold)
    table["failure_all"] = table["missed"] | (table["dice"] < 0.5)
    table["failure_non_missed"] = (~table["missed"]) & (table["dice"] < 0.5)
    table["clean"] = (~table["missed"]) & (table["dice"] >= 0.5)

    definitions = {
        "all_failures": (table["failure_all"], None),
        "non_missed_failures": (table["failure_non_missed"], ~table["missed"]),
        "missed_vs_clean": (table["missed"], table["missed"] | table["clean"]),
    }
    auroc_rows: list[dict[str, Any]] = []
    for key, description in PREDICTORS.items():
        if key not in table.columns:
            continue
        for definition, (labels, subset) in definitions.items():
            frame = table if subset is None else table.loc[subset]
            frame_labels = labels.loc[frame.index]
            result = auroc_ci(
                frame[key],
                frame_labels,
                frame["group_id"],
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
            )
            low_bound = high_bound = float("nan")
            if definition == "all_failures":
                low_bound, high_bound = auroc_nan_bounds(frame[key], frame_labels)
            auroc_rows.append(
                {
                    "predictor": key,
                    "description": description,
                    "failure_definition": definition,
                    **result,
                    "auroc_if_undefined_certain": low_bound,
                    "auroc_if_undefined_uncertain": high_bound,
                }
            )
    auroc_table = pd.DataFrame(auroc_rows)

    spearman_rows: list[dict[str, Any]] = []
    for key in PREDICTORS:
        if key not in table.columns:
            continue
        values = table[key].to_numpy(dtype=float)
        for target, target_values in (
            ("hd95", table["hd95"].to_numpy(dtype=float)),
            ("one_minus_dice", 1.0 - table["dice"].to_numpy(dtype=float)),
        ):
            rho, p_value = spearman(values, target_values)
            used = int(np.sum(np.isfinite(values) & np.isfinite(target_values)))
            spearman_rows.append(
                {
                    "predictor": key,
                    "target": target,
                    "n_used": used,
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )
    spearman_table = pd.DataFrame(spearman_rows)

    labels = table["failure_non_missed"]
    subset = table.loc[~table["missed"]]
    subset_labels = labels.loc[subset.index]
    test_rows: list[dict[str, Any]] = []
    for other in PREDICTORS:
        if other == "tta4_entropy_roi" or other not in table.columns:
            continue
        result = auroc_difference(
            subset["tta4_entropy_roi"],
            subset[other],
            subset_labels,
            subset["group_id"],
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
        test_rows.append(
            {
                "predictor": "tta4_entropy_roi",
                "compared_with": other,
                "failure_definition": "non_missed_failures",
                **result,
            }
        )
    test_table = pd.DataFrame(test_rows)

    notes: list[str] = []
    failures = table.loc[table["failure_all"]]
    notes.append(
        f"{len(failures)} failures (Dice < 0.5 or missed): "
        f"{int(failures['tta4_entropy_roi'].isna().sum())} have no ROI value (empty "
        f"prediction), {int((failures['pred_area_px'] <= near_empty_px).sum())} have a ROI "
        f"of <= {near_empty_px} px."
    )
    if by_seed is not None:
        notes.append(_validate_against_runner(runs, by_seed))
    return {
        "per_case": table,
        "auroc": auroc_table,
        "spearman": spearman_table,
        "tests": test_table,
        "roi_diagnostics": roi_diagnostics(table, near_empty_px=near_empty_px),
        "notes": notes,
    }


def _validate_against_runner(
    runs: Sequence[Any], by_seed: Mapping[int, Sequence[pd.DataFrame]], limit: int = 5
) -> str:
    """Recompute the tta4 ROI entropy from one seed's map and compare with the CSV."""
    run = sorted(runs, key=lambda item: (item.fold, item.seed))[0]
    frame = by_seed.get(int(run.seed))
    if not frame:
        return "validation skipped: no per-seed frame"
    fold_frame = next((item for item in frame if int(item["fold"].iloc[0]) == int(run.fold)), None)
    if fold_frame is None:
        return "validation skipped: no frame for the first fold"
    rows = fold_frame.loc[
        (fold_frame["method"] == "tta4") & (fold_frame["metric"] == "entropy_mean_roi")
    ].head(limit)
    differences: list[float] = []
    for _, row in rows.iterrows():
        path = Path(run.map_dir) / f"{row['case_id']}__tta4__mean.npy"
        if not path.is_file():
            continue
        stats = map_statistics(np.load(path).astype(np.float32))
        if np.isfinite(stats.entropy_roi) and np.isfinite(row["value"]):
            differences.append(abs(stats.entropy_roi - float(row["value"])))
    if not differences:
        return "validation skipped: no overlapping cases"
    return (
        f"validation: recomputed tta4 ROI entropy matches the runner's CSV on "
        f"{len(differences)} case(s), max |diff| = {max(differences):.2e}"
    )


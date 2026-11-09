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
* **AURC** (area under the risk-coverage curve, risk = 1 - Dice) next to the AUROC, on
  the same case set, together with the *optimal-ranking* AURC (oracle ordering) as the
  reference for how much selective prediction is left on the table;
* trivial baselines for the same task (predicted area, 1 - confidence, image entropy,
  single-view baseline entropy) and the seed-ensemble entropy/variance;
* **agreement detectors**: 1 - mean pairwise Dice between the four tta4 views, and
  between the three seeds' baseline maps (``ensemble3``) or tta4 maps
  (``ensemble3_tta4``), each scored against the failures of its own prediction;
* video-cluster bootstrap CIs on every AUROC/AURC and paired tests between
  descriptors (both statistics, shared resampling).

The view-level descriptor needs the individual view masks, which the runner does not
store: :func:`load_view_agreement` reads the ``view_pairwise_dice`` rows of a separate
CPU re-run (``configs/cvc_clinicdb/unet_resnet50_view_agreement.yaml``) and the audit
reports the NaNs and a note when that file is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from melba.metrics.uncertainty import (
    foreground_band,
    mean_pairwise_dice,
    predictive_entropy_map,
)

__all__ = [
    "DEFAULT_BAND_PX",
    "EXTRA_COMPARISONS",
    "NEAR_EMPTY_PX",
    "MapStats",
    "aurc",
    "aurc_ci",
    "aurc_optimal",
    "auroc_ci",
    "auroc_difference",
    "auroc_nan_bounds",
    "detector_difference",
    "load_view_agreement",
    "map_statistics",
    "roi_diagnostics",
    "run_uncertainty_audit",
    "selective_risk_curve",
    "view_agreement_dir",
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
    """Statistics of a mean probability map, using the runner's exact ROI definition.

    Works for 2-D (polyp) and 3-D (BraTS) maps: everything below is elementwise or a
    mean over the array, and ``foreground_band`` / ``predictive_entropy_map`` are
    dimension-agnostic.  ``area_px`` is therefore "foreground pixels" for 2-D data and
    "foreground voxels" for 3-D data.
    """
    probability = np.asarray(mean_map, dtype=np.float64)
    if probability.ndim < 2:
        raise ValueError(f"expected a probability map with >= 2 dims, got {probability.shape}")
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
    for _ in range(int(n_bootstrap) if unique.size else 0):
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



def _aurc_from_sorted(scores: np.ndarray, risks: np.ndarray) -> float:
    """AURC of an already-finite, already-paired (score, risk) pair of arrays.

    Cases are kept from the most confident (lowest score) to the least confident one,
    which is the risk-coverage curve of selective prediction; the AURC is the mean
    selective risk over all coverage levels.  Tied scores are averaged *within* the tie
    block, so the value does not depend on the order of equal scores (a deterministic
    tie-break would make e.g. the many 1.0 pairwise-Dice scores arbitrary).
    """
    if scores.size < 3:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    ordered_risks = risks[order]
    total = 0.0
    included = 0
    summed = 0.0
    start = 0
    while start < ordered_scores.size:
        stop = start + 1
        while stop < ordered_scores.size and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        block = ordered_risks[start:stop]
        block_mean = float(block.mean())
        for step in range(1, block.size + 1):
            total += (summed + step * block_mean) / (included + step)
        summed += block.size * block_mean
        included += block.size
        start = stop
    return float(total / ordered_risks.size)


def selective_risk_curve(
    scores: Iterable[float], risks: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective risk of the ranking given by ``scores`` (low = kept).

    Returns two arrays of the same length: the coverage ``k / n`` after keeping the
    ``k`` most confident cases and the mean risk (``1 - Dice``) of exactly those cases.
    Undefined (NaN) score/risk pairs are dropped first, so the curve refers to the same
    case set as :func:`aurc`.
    """
    values = np.asarray(list(scores), dtype=float)
    risk = np.asarray(list(risks), dtype=float)
    keep = np.isfinite(values) & np.isfinite(risk)
    values, risk = values[keep], risk[keep]
    if values.size == 0:
        return np.empty(0), np.empty(0)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(risk[order]) / np.arange(1, values.size + 1)
    coverage = np.arange(1, values.size + 1) / values.size
    return coverage, cumulative


def aurc(scores: Iterable[float], risks: Iterable[float]) -> float:
    """Area under the risk-coverage curve of a detector (lower = better detector).

    ``scores`` is an uncertainty score (higher = less confident) and ``risks`` the
    per-case ``1 - Dice`` of the prediction the score refers to.  A perfect detector
    (risk non-decreasing in the score) reaches the optimal-ranking AURC; a useless one
    sits at or above the risk of the whole cohort.
    """
    values = np.asarray(list(scores), dtype=float)
    risk = np.asarray(list(risks), dtype=float)
    keep = np.isfinite(values) & np.isfinite(risk)
    return _aurc_from_sorted(values[keep], risk[keep])


def aurc_optimal(risks: Iterable[float]) -> float:
    """AURC of the oracle ranking (cases ordered by their own risk): the lower bound.

    Reported next to every detector so a raw AURC can be read relative to what is
    achievable at all on that cohort.
    """
    risk = np.asarray(list(risks), dtype=float)
    risk = risk[np.isfinite(risk)]
    if risk.size < 3:
        return float("nan")
    return _aurc_from_sorted(risk, risk)


def _percentile_bounds(
    draws: Sequence[float], alpha: float, low_key: str, high_key: str, p_key: str
) -> dict[str, float]:
    """Percentile CI and two-sided bootstrap p-value of a paired difference."""
    if len(draws) < 10:
        return {}
    distribution = np.asarray(draws, dtype=float)
    low, high = np.percentile(distribution, [100 * alpha, 100 * (1 - alpha)])
    p_value = 2.0 * min(
        float(np.mean(distribution <= 0.0)), float(np.mean(distribution >= 0.0))
    )
    return {low_key: float(low), high_key: float(high), p_key: float(min(1.0, p_value))}


def aurc_ci(
    scores: Iterable[float],
    risks: Iterable[float],
    groups: Iterable[Any],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """AURC with a video-cluster bootstrap CI and the optimal-ranking reference.

    Videos are resampled as blocks (frames of one video are dependent) and every
    resample is re-ranked from scratch, so the CI covers the risk levels and the
    ordering uncertainty together.
    """
    values = np.asarray(list(scores), dtype=float)
    risk = np.asarray(list(risks), dtype=float)
    group_values = np.asarray(list(groups))
    keep = np.isfinite(values) & np.isfinite(risk)
    values, risk, group_values = values[keep], risk[keep], group_values[keep]
    point = _aurc_from_sorted(values, risk)
    optimal = aurc_optimal(risk)
    unique = np.unique(group_values)
    positions = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    for _ in range(int(n_bootstrap) if unique.size else 0):
        pick = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([positions[unique[index]] for index in pick])
        value = _aurc_from_sorted(values[rows], risk[rows])
        if np.isfinite(value):
            draws.append(float(value))
    result: dict[str, float] = {
        "aurc": point,
        "aurc_optimal": optimal,
        "aurc_excess": point - optimal if np.isfinite(optimal) else float("nan"),
        "aurc_ci_low": float("nan"),
        "aurc_ci_high": float("nan"),
        "n_used_aurc": int(values.size),
        "n_undefined_aurc": int((~keep).sum()),
    }
    if len(draws) >= 10:
        alpha = (1.0 - float(ci)) / 2.0
        low, high = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])
        result["aurc_ci_low"] = float(low)
        result["aurc_ci_high"] = float(high)
    return result


def detector_difference(
    scores_a: Iterable[float],
    scores_b: Iterable[float],
    labels: Iterable[Any],
    risks: Iterable[float],
    groups: Iterable[Any],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Paired video-cluster bootstrap of ``AUROC(a) - AUROC(b)`` and ``AURC(a) - AURC(b)``.

    One resampling is shared by both statistics, so a detector that wins on AUROC and
    loses on AURC cannot be explained away by two different bootstrap draws.  The AUROC
    delta is positive when ``a`` is the better *classifier*; the AURC delta is a
    difference of selective risks, i.e. **negative when ``a`` is the better detector**.
    """
    a = np.asarray(list(scores_a), dtype=float)
    b = np.asarray(list(scores_b), dtype=float)
    flags = np.asarray(list(labels)).astype(bool)
    risk = np.asarray(list(risks), dtype=float)
    group_values = np.asarray(list(groups))
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(risk)
    a, b, flags, risk, group_values = (
        a[keep],
        b[keep],
        flags[keep],
        risk[keep],
        group_values[keep],
    )
    delta_auroc = _auroc(a, flags) - _auroc(b, flags)
    delta_aurc = _aurc_from_sorted(a, risk) - _aurc_from_sorted(b, risk)
    unique = np.unique(group_values)
    positions = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    auroc_draws: list[float] = []
    aurc_draws: list[float] = []
    for _ in range(int(n_bootstrap) if unique.size else 0):
        pick = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([positions[unique[index]] for index in pick])
        auroc_value = _auroc(a[rows], flags[rows]) - _auroc(b[rows], flags[rows])
        if np.isfinite(auroc_value):
            auroc_draws.append(float(auroc_value))
        aurc_value = _aurc_from_sorted(a[rows], risk[rows]) - _aurc_from_sorted(
            b[rows], risk[rows]
        )
        if np.isfinite(aurc_value):
            aurc_draws.append(float(aurc_value))
    result: dict[str, float] = {
        "delta_auroc": float(delta_auroc),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": float("nan"),
        "delta_aurc": float(delta_aurc),
        "aurc_ci_low": float("nan"),
        "aurc_ci_high": float("nan"),
        "aurc_p_value": float("nan"),
        "n_used": int(a.size),
    }
    alpha = (1.0 - float(ci)) / 2.0
    result.update(_percentile_bounds(auroc_draws, alpha, "ci_low", "ci_high", "p_value"))
    result.update(
        _percentile_bounds(aurc_draws, alpha, "aurc_ci_low", "aurc_ci_high", "aurc_p_value")
    )
    return result


def _load_maps(
    runs: Sequence[Any], case_id: str, method: str, *, required: bool = True
) -> list[np.ndarray] | None:
    """Saved probability maps of one case, or ``None`` when they were not persisted.

    Evaluations that keep only a seeded subset of maps (``evaluation.probability_map_cases``,
    as the BraTS grid does) have maps for a handful of cases and none for the rest; a
    missing map is then a *normal* condition, not an error.  With ``required=False`` the
    caller gets ``None`` and falls back to the per-case uncertainty columns of the CSV.
    """
    maps = []
    for run in runs:
        path = Path(run.map_dir) / f"{case_id}__{method}__mean.npy"
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"missing probability map {path}")
            return None
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
    ensemble_frame: pd.DataFrame | None = None,
    view_agreement: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-case predictors and per-prediction failure labels.

    Every predictor is paired with the labels of **its own** prediction: ``baseline_*``
    with the single-view prediction, ``tta4_*`` with the tta4 prediction,
    ``ensemble3_*`` with the seed ensemble of the baseline maps and
    ``ensemble3_tta4``/``map_*_tta4``/``pred_area_px``/``one_minus_confidence`` with the
    seed-averaged tta4 map.  All label sets are kept as columns so any analysis can also
    be repeated against one shared set.

    ``view_agreement`` adds the mean pairwise Dice between the four tta4 *views*
    (one row per case, see :func:`load_view_agreement`); the two seed-map variants
    (``ensemble3_pairwise_dice`` over the three baseline maps,
    ``ensemble3_tta4_pairwise_dice`` over the three tta4 maps) are computed here from
    the saved probability maps.
    """
    case_metrics = (
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
        averaged.loc[averaged["method"] == "tta4", ["case_id", "group_id", "fold"]]
        .drop_duplicates("case_id")
        .assign(case_id=lambda frame: frame["case_id"].astype(str))
    )

    def labels_for(prefix: str, frame: pd.DataFrame, method: str) -> pd.DataFrame:
        """``dice``/``hd95``/``missed`` columns for one prediction, keyed by case id."""
        wide = _wide(frame, method, ("dice", "hd95"))
        missed = (
            frame.loc[frame["method"] == method, ["case_id", "missed_detection"]]
            .drop_duplicates("case_id")
            .rename(columns={"missed_detection": f"missed_{prefix}"})
        )
        wide = wide.merge(missed, on="case_id", how="outer")
        wide["case_id"] = wide["case_id"].astype(str)
        return wide.rename(columns={"dice": f"dice_{prefix}", "hd95": f"hd95_{prefix}"})

    table = labels_for("tta4", averaged, "tta4")
    table = table.merge(labels_for("baseline", averaged, "baseline"), on="case_id", how="outer")
    if ensemble_frame is not None and not ensemble_frame.empty:
        for prefix, method in (("ensemble3", "ensemble3"), ("ensemble3_tta4", "ensemble3_tta4")):
            if method in set(ensemble_frame["method"].unique()):
                table = table.merge(labels_for(prefix, ensemble_frame, method), on="case_id", how="outer")
            else:
                for column in (f"dice_{prefix}", f"hd95_{prefix}", f"missed_{prefix}"):
                    table[column] = np.nan
    else:
        for prefix in ("ensemble3", "ensemble3_tta4"):
            for column in (f"dice_{prefix}", f"hd95_{prefix}", f"missed_{prefix}"):
                table[column] = np.nan

    table = table.merge(meta, on="case_id", how="left")
    table = table.merge(tta4, on="case_id", how="left").merge(baseline, on="case_id", how="left")
    table["fold"] = table["fold"].astype(int)
    for prefix in ("tta4", "baseline", "ensemble3", "ensemble3_tta4"):
        column = f"missed_{prefix}"
        table[column] = table[column].fillna(0).astype(bool)
    if view_agreement is not None and not view_agreement.empty:
        agreed = view_agreement.copy()
        agreed["case_id"] = agreed["case_id"].astype(str)
        table = table.merge(agreed, on="case_id", how="left")
    else:
        table["tta4_view_pairwise_dice"] = np.nan
    for column in ("ensemble3_pairwise_dice", "ensemble3_tta4_pairwise_dice"):
        table[column] = np.nan

    runs_by_fold: dict[int, list[Any]] = {}
    for run in runs:
        runs_by_fold.setdefault(int(run.fold), []).append(run)
    maps_used = 0
    maps_missing = 0
    for fold, fold_runs in runs_by_fold.items():
        ordered = sorted(fold_runs, key=lambda item: item.seed)
        for index in table.index[table["fold"] == fold]:
            case_id = str(table.at[index, "case_id"])
            tta4_maps = _load_maps(ordered, case_id, "tta4", required=False)
            baseline_maps = _load_maps(ordered, case_id, "baseline", required=False)
            if tta4_maps is None or baseline_maps is None:
                # no saved map for this case (a capped map set, not an error): the map-only
                # descriptors stay NaN and the CSV-derived ones are filled in below
                maps_missing += 1
                continue
            maps_used += 1
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
            table.at[index, "ensemble3_pairwise_dice"] = _seed_pairwise_dice(
                baseline_maps, threshold
            )
            table.at[index, "ensemble3_tta4_pairwise_dice"] = _seed_pairwise_dice(
                tta4_maps, threshold
            )
    for source, target in (
        ("tta4_view_pairwise_dice", "tta4_view_dice_disagreement"),
        ("ensemble3_pairwise_dice", "ensemble3_dice_disagreement"),
        ("ensemble3_tta4_pairwise_dice", "ensemble3_tta4_dice_disagreement"),
    ):
        table[target] = 1.0 - table[source].astype(float)
    notes = _fill_from_csv(table, averaged, maps_used=maps_used, maps_missing=maps_missing)
    table.attrs["notes"] = tuple(notes)
    return table


def _csv_voxel_area(averaged: pd.DataFrame, method: str) -> pd.Series:
    """Predicted foreground in voxels, from ``pred_volume_mm3`` and the stored spacing."""
    wide = _wide(averaged, method, ("pred_volume_mm3",))
    if "pred_volume_mm3" not in wide.columns:
        return pd.Series(dtype=float)
    spacing = (
        averaged.loc[averaged["method"] == method, ["case_id", "spacing_mm"]]
        .drop_duplicates("case_id")
        .assign(case_id=lambda frame: frame["case_id"].astype(str))
    )
    wide["case_id"] = wide["case_id"].astype(str)
    merged = wide.merge(spacing, on="case_id", how="left")

    def voxel_volume(value: Any) -> float:
        parts = [item for item in str(value or "1").replace(";", ",").split(",") if item.strip()]
        try:
            values = [float(item) for item in parts] or [1.0]
        except ValueError:  # pragma: no cover - only a malformed spacing string
            return 1.0
        return float(np.prod(values))

    factor = merged["spacing_mm"].map(voxel_volume).astype(float)
    return pd.Series(
        merged["pred_volume_mm3"].astype(float).to_numpy() / factor.to_numpy(),
        index=merged["case_id"].to_numpy(),
    )


def _fill_from_csv(
    table: pd.DataFrame,
    averaged: pd.DataFrame,
    *,
    maps_used: int,
    maps_missing: int,
) -> list[str]:
    """Fill the map-only descriptors from the per-case CSV where an equivalent exists.

    The evaluation CSV already carries the descriptors that only need the *view stack*
    (``entropy_mean_roi``, ``variance_mean_roi``, ``view_pairwise_dice``, ``pred_volume_mm3``),
    so a run that saved no probability map can still be audited; only the genuinely
    map-only ones (single-map confidence, ambiguous fraction, ROI mask size, the seed
    ensembles) stay NaN, and a note says which.
    """
    notes: list[str] = []
    columns = ("entropy_mean_roi", "variance_mean_roi", "view_pairwise_dice", "pred_volume_mm3")
    tta4 = _wide(averaged, "tta4", columns)
    if not tta4.empty:
        tta4["case_id"] = tta4["case_id"].astype(str)
        tta4 = tta4.set_index("case_id").reindex(table["case_id"].astype(str))
        fallbacks = {
            "map_entropy_roi_tta4": "entropy_mean_roi",
            "map_variance_roi_tta4": "variance_mean_roi",
            "tta4_view_pairwise_dice": "view_pairwise_dice",
        }
        filled: list[str] = []
        for target, source in fallbacks.items():
            if source not in tta4.columns:
                continue
            values = tta4[source].astype(float).to_numpy()
            if target not in table.columns:
                table[target] = np.nan
            current = table[target].astype(float).to_numpy()
            updated = np.where(np.isfinite(current), current, values)
            table[target] = updated
            if np.isfinite(values).any():
                filled.append(f"{target} <- csv {source}")
        area = _csv_voxel_area(averaged, "tta4")
        if not area.empty:
            values = area.reindex(table["case_id"].astype(str)).to_numpy(dtype=float)
            if "pred_area_px" not in table.columns:
                table["pred_area_px"] = np.nan
            current = table["pred_area_px"].astype(float).to_numpy()
            table["pred_area_px"] = np.where(np.isfinite(current), current, values)
            if np.isfinite(values).any():
                filled.append("pred_area_px <- csv pred_volume_mm3 / voxel volume")
        if filled:
            notes.append(
                "map-derived descriptors replaced by the per-case CSV columns where the "
                "evaluation wrote an equivalent (" + "; ".join(filled) + ")."
            )
    if maps_missing:
        notes.append(
            f"probability maps present for {maps_used} of {maps_used + maps_missing} cases "
            "(evaluation.probability_map_cases caps the saved maps); the map-only "
            "descriptors `one_minus_confidence`, `map_ambiguous_fraction_tta4`, `pred_roi_px` "
            "and every ensemble column are NaN for the cases without a map, and their rows "
            "drop out of the predictor tables (`n_used` / `n_undefined` show it)."
        )
    for column in ("tta4_view_pairwise_dice", "ensemble3_pairwise_dice"):
        values = table[column].astype(float).to_numpy()
        if not np.isfinite(values).any():
            notes.append(
                f"`{column}` is undefined for every case: no view-agreement CSV and no map "
                "(ensembles need at least two seeds)."
            )
    return notes



def _seed_pairwise_dice(maps: Sequence[np.ndarray], threshold: float) -> float:
    """Mean pairwise Dice between the seeds' probability maps (NaN for a single seed)."""
    if len(maps) < 2:
        return float("nan")
    return mean_pairwise_dice(np.stack(maps, axis=0)[:, None], threshold)



def _nanmedian(values: np.ndarray) -> float:
    """Median of the finite values (NaN, without a warning, when none is finite)."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("nan")


def roi_diagnostics(table: pd.DataFrame, *, near_empty_px: int = NEAR_EMPTY_PX) -> pd.DataFrame:
    """How the ROI descriptors behave on the tta4 failures (empty, near-empty, larger)."""
    rows: list[dict[str, Any]] = []
    failure = table["missed_tta4"] | (table["dice_tta4"] < 0.5)
    failures = table.loc[failure]
    cohorts = {
        "all_cases": table,
        "failures_all": failures,
        "failures_non_missed": failures.loc[~failures["missed_tta4"]],
        "failures_missed": failures.loc[failures["missed_tta4"]],
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
    # agreement-based detectors: higher score = the members disagree = more uncertain
    "tta4_view_dice_disagreement": (
        "tta4: 1 - mean pairwise Dice between the 4 views (view agreement)"
    ),
    "ensemble3_dice_disagreement": (
        "ensemble3: 1 - mean pairwise Dice between the 3 seeds' baseline maps"
    ),
    "ensemble3_tta4_dice_disagreement": (
        "ensemble3_tta4: 1 - mean pairwise Dice between the 3 seeds' tta4 maps"
    ),
}

#: which prediction's failures each predictor is scored against (``own`` labels)
PREDICTOR_LABELS: dict[str, str] = {
    "tta4_entropy_roi": "tta4",
    "tta4_variance_roi": "tta4",
    "tta4_view_disagreement": "tta4",
    "tta4_entropy_image": "tta4",
    "map_entropy_roi_tta4": "ensemble3_tta4",
    "map_variance_roi_tta4": "ensemble3_tta4",
    "map_ambiguous_fraction_tta4": "ensemble3_tta4",
    "pred_area_px": "ensemble3_tta4",
    "one_minus_confidence": "ensemble3_tta4",
    "baseline_entropy_roi": "baseline",
    "baseline_entropy_image": "baseline",
    "ensemble3_entropy_roi": "ensemble3",
    "ensemble3_variance_roi": "ensemble3",
    "ensemble3_entropy_image": "ensemble3",
    "ensemble3_variance_image": "ensemble3",
    "tta4_view_dice_disagreement": "tta4",
    "ensemble3_dice_disagreement": "ensemble3",
    "ensemble3_tta4_dice_disagreement": "ensemble3_tta4",
}

#: paired comparisons demanded by the view-agreement analysis, beyond the
#: ``tta4_entropy_roi`` anchor that :func:`run_uncertainty_audit` always uses.
#: ``(predictor, compared_with)``; the *first* entry of each pair supplies the labels.
EXTRA_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("tta4_view_dice_disagreement", "tta4_entropy_roi"),
    ("tta4_view_dice_disagreement", "one_minus_confidence"),
    ("ensemble3_dice_disagreement", "ensemble3_entropy_roi"),
    ("ensemble3_tta4_dice_disagreement", "map_entropy_roi_tta4"),
)



def run_uncertainty_audit(
    runs: Sequence[Any],
    averaged: pd.DataFrame,
    *,
    by_seed: Mapping[int, Sequence[pd.DataFrame]] | None = None,
    ensemble_frame: pd.DataFrame | None = None,
    view_agreement: pd.DataFrame | None = None,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    band_px: int = DEFAULT_BAND_PX,
    threshold: float = 0.5,
    near_empty_px: int = NEAR_EMPTY_PX,
) -> dict[str, Any]:
    """Build every uncertainty-audit table (see the module docstring).

    ``view_agreement`` is the per-case output of :func:`load_view_agreement`; without
    it the view-pairwise-Dice detector is NaN and a note says so.
    """
    from melba.eval.report import spearman

    table = build_case_table(
        runs,
        averaged,
        band_px=band_px,
        threshold=threshold,
        ensemble_frame=ensemble_frame,
        view_agreement=view_agreement,
    )
    definitions_by_source: dict[str, dict[str, tuple[pd.Series, pd.Series | None]]] = {}
    for prefix in ("tta4", "baseline", "ensemble3", "ensemble3_tta4"):
        dice = table[f"dice_{prefix}"].astype(float)
        missed = table[f"missed_{prefix}"].astype(bool)
        definitions_by_source[prefix] = {
            "all_failures": (missed | (dice < 0.5), None),
            "non_missed_failures": ((~missed) & (dice < 0.5), ~missed),
            "missed_vs_clean": (missed, missed | ((~missed) & (dice >= 0.5))),
        }
    auroc_rows: list[dict[str, Any]] = []
    for key, description in PREDICTORS.items():
        if key not in table.columns:
            continue
        own = PREDICTOR_LABELS.get(key, "tta4")
        if not np.isfinite(table[f"dice_{own}"].to_numpy(dtype=float)).any():
            own = "tta4"  # no ensemble labels available (e.g. ensembles skipped)
        for source, label_source in ((own, "own"), ("tta4", "tta4")):
            if label_source == "tta4" and own == "tta4":
                continue  # own == shared, do not duplicate
            definitions = definitions_by_source[source]
            risks = 1.0 - table[f"dice_{source}"].astype(float)
            missed_flags = table[f"missed_{source}"].astype(bool)
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
                # AURC on the same case set: risk = 1 - Dice of the labelled prediction
                aurc_result = aurc_ci(
                    frame[key],
                    risks.loc[frame.index],
                    frame["group_id"],
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                    seed=seed,
                )
                worst_case_risk = np.where(
                    missed_flags.loc[frame.index].to_numpy(),
                    1.0,
                    risks.loc[frame.index].to_numpy(dtype=float),
                )
                low_bound = high_bound = float("nan")
                if definition == "all_failures":
                    low_bound, high_bound = auroc_nan_bounds(frame[key], frame_labels)
                auroc_rows.append(
                    {
                        "predictor": key,
                        "description": description,
                        "label_source": label_source,
                        "label_prediction": source,
                        "failure_definition": definition,
                        **result,
                        **aurc_result,
                        "aurc_if_missed_is_worst": aurc(frame[key], worst_case_risk),
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
        own = PREDICTOR_LABELS.get(key, "tta4")
        if not np.isfinite(table[f"dice_{own}"].to_numpy(dtype=float)).any():
            own = "tta4"
        for source, label_source in ((own, "own"), ("tta4", "tta4")):
            if label_source == "tta4" and own == "tta4":
                continue
            for target, target_values in (
                ("hd95", table[f"hd95_{source}"].to_numpy(dtype=float)),
                ("one_minus_dice", 1.0 - table[f"dice_{source}"].to_numpy(dtype=float)),
            ):
                rho, p_value = spearman(values, target_values)
                used = int(np.sum(np.isfinite(values) & np.isfinite(target_values)))
                spearman_rows.append(
                    {
                        "predictor": key,
                        "label_source": label_source,
                        "label_prediction": source,
                        "target": target,
                        "n_used": used,
                        "spearman_rho": rho,
                        "p_value": p_value,
                    }
                )
    spearman_table = pd.DataFrame(spearman_rows)

    test_rows: list[dict[str, Any]] = []

    def _add_comparison(first: str, second: str, *, label_from: str | None = None) -> None:
        """Paired AUROC/AURC bootstrap of ``first`` vs ``second``.

        ``label_from`` names the prediction whose failures are used; it defaults to the
        first predictor's own prediction.
        """
        if first not in table.columns or second not in table.columns:
            return
        if not np.isfinite(table[first].to_numpy(dtype=float)).any():
            return
        if not np.isfinite(table[second].to_numpy(dtype=float)).any():
            return
        source = label_from or PREDICTOR_LABELS.get(first, "tta4")
        if not np.isfinite(table[f"dice_{source}"].to_numpy(dtype=float)).any():
            source = "tta4"
        labels, subset = definitions_by_source[source]["non_missed_failures"]
        frame = table.loc[subset]
        risks = (1.0 - table[f"dice_{source}"].astype(float)).loc[frame.index]
        result = detector_difference(
            frame[first],
            frame[second],
            labels.loc[frame.index],
            risks,
            frame["group_id"],
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
        test_rows.append(
            {
                "predictor": first,
                "compared_with": second,
                "label_prediction": source,
                "failure_definition": "non_missed_failures",
                **result,
            }
        )

    for other in PREDICTORS:
        if other == "tta4_entropy_roi":
            continue
        # under the tta4 labels, and under the alternative's own labels
        other_source = PREDICTOR_LABELS.get(other, "tta4")
        for label_from in dict.fromkeys(("tta4", other_source)):
            _add_comparison("tta4_entropy_roi", other, label_from=label_from)
    for first, second in EXTRA_COMPARISONS:
        _add_comparison(first, second)
    test_table = pd.DataFrame(test_rows)

    notes: list[str] = list(table.attrs.get("notes", ()))
    tta4_failure = table["missed_tta4"] | (table["dice_tta4"] < 0.5)
    failures = table.loc[tta4_failure]
    counts = {
        prefix: int(
            (table[f"missed_{prefix}"] | (table[f"dice_{prefix}"] < 0.5)).sum()
        )
        for prefix in ("tta4", "baseline", "ensemble3", "ensemble3_tta4")
    }
    notes.append(
        "failures per prediction (Dice < 0.5 or missed): "
        + ", ".join(f"{prefix} {count}" for prefix, count in counts.items())
    )
    notes.append(
        f"tta4 shares {len(failures)} failures; "
        f"{int(failures['tta4_entropy_roi'].isna().sum())} of them have no ROI value "
        f"(empty prediction), {int((failures['pred_area_px'] <= near_empty_px).sum())} "
        f"have a ROI of <= {near_empty_px} px."
    )
    notes.append(
        "AURC uses risk = 1 - Dice of the prediction a descriptor is scored against "
        "(lower is better); `aurc_optimal` orders the same cases by their own risk, so "
        "`aurc_excess` is what the ranking loses against the oracle. Missed detections "
        "have no Dice and are dropped (`n_undefined_aurc`); "
        "`aurc_if_missed_is_worst` repeats the point estimate with their risk set to 1."
    )
    missing_view = int(table["tta4_view_pairwise_dice"].isna().sum())
    if missing_view:
        notes.append(
            f"view pairwise Dice unavailable for {missing_view} of {len(table)} cases: "
            "the runner does not store per-view masks. Re-run tta4 on CPU with "
            "configs/cvc_clinicdb/unet_resnet50_view_agreement.yaml so the "
            "`view_pairwise_dice` metric exists, then pass it as `view_agreement`."
        )
    else:
        notes.append(
            "view pairwise Dice: mean over the 4 tta4 views per case, averaged over the "
            "seeds (1.0 means the four views produced the same binary mask)."
        )
    seed_dice = table["ensemble3_pairwise_dice"]
    notes.append(
        f"seed agreement: {int(np.isfinite(seed_dice).sum())} cases have a pairwise Dice "
        "between the three seeds' baseline maps (NaN when an ensemble has one seed); "
        "`ensemble3_tta4_pairwise_dice` is the same over the three tta4 maps."
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


def view_agreement_dir(config: Any) -> Path:
    """Default directory of the tta4 view-agreement re-run, one per experiment."""
    experiment = str(config.get("experiment.name", "experiment")).strip() or "experiment"
    return Path("results") / experiment / "view_agreement" / "evaluation"


def load_view_agreement(
    paths: Iterable[str | Path],
    *,
    method: str = "tta4",
    metric: str = "view_pairwise_dice",
) -> pd.DataFrame:
    """Mean pairwise Dice between a method's views, averaged over seeds, per case.

    ``paths`` are the per-case CSVs of the CPU view-agreement re-run
    (``scripts/evaluate.py --config .../unet_resnet50_view_agreement.yaml``); a
    directory is expanded to its ``*_per_case.csv`` files.  The result has the columns
    ``case_id`` (``str``), ``tta4_view_pairwise_dice`` and ``n_seeds``; a case missing
    from some seeds is averaged over the seeds that do have it.
    """
    resolved: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*_per_case.csv")))
        elif path.is_file():
            resolved.append(path)
    frames: list[pd.DataFrame] = []
    for path in resolved:
        frame = pd.read_csv(path)
        if not {"case_id", "method", "metric", "value"} <= set(frame.columns):
            continue
        subset = frame.loc[(frame["method"] == method) & (frame["metric"] == metric)]
        if subset.empty:
            continue
        subset = subset[["case_id", "seed", "value"]].copy()
        subset["case_id"] = subset["case_id"].astype(str)
        frames.append(subset)
    if not frames:
        return pd.DataFrame(columns=["case_id", "tta4_view_pairwise_dice", "n_seeds"])
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby("case_id", as_index=False).agg(
        tta4_view_pairwise_dice=("value", "mean"),
        n_seeds=("seed", "nunique"),
    )
    return grouped


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


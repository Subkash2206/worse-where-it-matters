"""MC-dropout summary: how ``mcdropout_T`` compares with geometric TTA, and whether its
entropy/variance are better failure detectors than the view descriptors.

The runner keeps one *per-sample* stack for an MC-dropout method, so exactly the same
descriptors are available as for ``tta4`` (entropy of the mean sample, across-sample
variance, disagreement, per-sample agreement).  This module scores them against the
failures of the MC prediction itself and compares them, on the same cases, with the
``tta4`` view descriptors — the question "does sampling the weights buy a better
uncertainty estimate than transforming the input?".

Everything is CPU-only and read-only: it consumes the per-case CSVs of an existing grid.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from melba.eval.report import CORE_METRICS
from melba.eval.uncertainty_audit import (
    aurc_ci,
    auroc_ci,
    auroc_nan_bounds,
    detector_difference,
)
from melba.stats import analyze_long

__all__ = [
    "MC_DESCRIPTORS",
    "build_mc_case_table",
    "mc_dropout_audit",
    "mc_dropout_deltas",
    "mc_dropout_method_table",
]

#: descriptors the runner derives from the MC samples (``melba.metrics.uncertainty``)
MC_DESCRIPTORS: dict[str, str] = {
    "entropy_mean": "mcdropout: entropy of the mean sample (whole image)",
    "entropy_mean_roi": "mcdropout: entropy of the mean sample in its ROI",
    "variance_mean": "mcdropout: across-sample variance (whole image)",
    "variance_mean_roi": "mcdropout: across-sample variance in its ROI",
    "view_disagreement": "mcdropout: fraction of voxels where the samples disagree",
    "per_view_dice_mean": "mcdropout: mean agreement of each sample with the consensus",
}


def _wide(frame: pd.DataFrame, method: str, metrics: Sequence[str]) -> pd.DataFrame:
    """One row per case with the named metrics of ``method`` as columns."""
    subset = frame.loc[frame["method"] == method, ["case_id", "metric", "value"]]
    subset = subset.loc[subset["metric"].isin(list(metrics))]
    if subset.empty:
        return pd.DataFrame()
    return subset.pivot(index="case_id", columns="metric", values="value").reset_index()


def build_mc_case_table(
    averaged: pd.DataFrame,
    *,
    method: str = "mcdropout_8",
    reference_method: str = "tta4",
) -> pd.DataFrame:
    """Per-case MC descriptors plus the labels and the reference descriptor they need.

    ``dice_<method>``/``hd95_<method>``/``missed_<method>`` are the MC prediction's own
    failures; ``reference_entropy_roi`` is ``tta4``'s ROI entropy, used for the paired
    "sampling vs input transforms" comparison on exactly the same cases.
    """
    if averaged.empty:
        return pd.DataFrame()
    available = set(averaged["method"].unique())
    if method not in available:
        raise ValueError(
            f"method '{method}' is not in the per-case results; available: {sorted(available)}"
        )
    descriptors = [
        name for name in MC_DESCRIPTORS if name in set(averaged["metric"].unique())
    ]
    table = _wide(averaged, method, (*descriptors, "dice", "hd95"))
    if table.empty:
        raise ValueError(f"method '{method}' has no descriptor rows")
    table = table.rename(columns={name: f"mc_{name}" for name in descriptors})
    table = table.rename(columns={"dice": f"dice_{method}", "hd95": f"hd95_{method}"})
    missed = (
        averaged.loc[averaged["method"] == method, ["case_id", "missed_detection"]]
        .drop_duplicates("case_id")
        .rename(columns={"missed_detection": f"missed_{method}"})
    )
    meta = (
        averaged.loc[averaged["method"] == method, ["case_id", "group_id", "fold"]]
        .drop_duplicates("case_id")
        .copy()
    )
    for frame in (table, missed, meta):
        frame["case_id"] = frame["case_id"].astype(str)
    table = table.merge(missed, on="case_id", how="outer")
    table = table.merge(meta, on="case_id", how="left")
    table[f"missed_{method}"] = table[f"missed_{method}"].fillna(0).astype(bool)

    if reference_method in available:
        reference = _wide(
            averaged, reference_method, ("entropy_mean_roi", "dice", "hd95")
        ).rename(
            columns={
                "entropy_mean_roi": "reference_entropy_roi",
                "dice": f"dice_{reference_method}",
                "hd95": f"hd95_{reference_method}",
            }
        )
        reference["case_id"] = reference["case_id"].astype(str)
        table = table.merge(reference, on="case_id", how="left")
    else:  # pragma: no cover - defensive: tta4 missing from a dropout config
        table["reference_entropy_roi"] = np.nan
    table["fold"] = table["fold"].astype(int)
    return table




def mc_dropout_audit(
    averaged: pd.DataFrame,
    *,
    method: str = "mcdropout_8",
    reference_method: str = "tta4",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """AUROC/AURC of every MC descriptor against the MC prediction's own failures."""
    table = build_mc_case_table(
        averaged, method=method, reference_method=reference_method
    )
    missed = table[f"missed_{method}"].to_numpy(dtype=bool)
    dice = table[f"dice_{method}"].astype(float)
    labels_all = missed | (dice < 0.5)
    subsets: dict[str, tuple[pd.Series, pd.Series | None]] = {
        "all_failures": (labels_all, None),
        "non_missed_failures": ((~missed) & (dice < 0.5), ~missed),
    }
    rows: list[dict[str, Any]] = []
    for key, description in MC_DESCRIPTORS.items():
        column = f"mc_{key}"
        if column not in table.columns:
            continue
        for definition, (labels, subset) in subsets.items():
            frame = table if subset is None else table.loc[subset]
            frame_labels = labels.loc[frame.index]
            risks = 1.0 - frame[f"dice_{method}"].astype(float)
            result = auroc_ci(
                frame[column],
                frame_labels,
                frame["group_id"],
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
            )
            aurc_result = aurc_ci(
                frame[column],
                risks,
                frame["group_id"],
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
            )
            low = high = float("nan")
            if definition == "all_failures":
                low, high = auroc_nan_bounds(frame[column], frame_labels)
            rows.append(
                {
                    "predictor": column,
                    "description": description,
                    "failure_definition": definition,
                    **result,
                    **aurc_result,
                    "auroc_if_undefined_certain": low,
                    "auroc_if_undefined_uncertain": high,
                }
            )
    auroc_table = pd.DataFrame(rows)

    tests: list[dict[str, Any]] = []
    if "reference_entropy_roi" in table.columns:
        labels, subset = subsets["non_missed_failures"]
        frame = table.loc[subset]
        frame_labels = labels.loc[frame.index]
        risks = 1.0 - frame[f"dice_{method}"].astype(float)
        for key in MC_DESCRIPTORS:
            column = f"mc_{key}"
            if column not in frame.columns:
                continue
            result = detector_difference(
                frame[column],
                frame["reference_entropy_roi"],
                frame_labels,
                risks,
                frame["group_id"],
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
            )
            tests.append(
                {
                    "predictor": column,
                    "compared_with": "reference_entropy_roi",
                    "label_prediction": method,
                    "failure_definition": "non_missed_failures",
                    **result,
                }
            )
    return {
        "per_case": table,
        "auroc": auroc_table,
        "tests": pd.DataFrame(tests),
        "notes": [
            f"{int(labels_all.sum())} of {len(table)} cases fail under {method} "
            f"(Dice < 0.5 or missed); {int(missed.sum())} of them are missed detections, "
            "for which the ROI descriptors are undefined.",
            "AUROC/AURC are computed against the MC prediction's own failures; the paired "
            "tests compare each MC descriptor with the tta4 ROI entropy on the same cases "
            "(delta_aurc < 0 means the MC descriptor ranks failures better).",
        ],
    }


def mc_dropout_method_table(
    averaged: pd.DataFrame,
    *,
    methods: Sequence[str] = ("baseline", "tta4", "tta8", "ensemble3", "mcdropout_8"),
    metrics: Sequence[str] = CORE_METRICS,
    unit: str = "group_id",
) -> pd.DataFrame:
    """Video-level mean ± std per method/metric, so the MC method sits next to TTA."""
    rows: list[dict[str, Any]] = []
    for method in methods:
        for metric in metrics:
            subset = averaged.loc[
                (averaged["method"] == method) & (averaged["metric"] == metric)
            ]
            if subset.empty:
                continue
            per_unit = subset.groupby(unit)["value"].mean()
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_cases": int(subset["case_id"].nunique()),
                    "n_units": int(per_unit.size),
                    "mean": float(per_unit.mean()),
                    "std": float(per_unit.std(ddof=1))
                    if per_unit.size > 1
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def mc_dropout_deltas(
    averaged: pd.DataFrame,
    *,
    method: str = "mcdropout_8",
    controls: Sequence[str] = ("baseline", "tta4"),
    metrics: Sequence[str] = CORE_METRICS,
    unit: str = "group",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
) -> pd.DataFrame:
    """Paired tests of the MC method against ``baseline`` and against ``tta4``."""
    available = set(averaged["method"].unique())
    if method not in available:
        return pd.DataFrame()
    available_metrics = [
        metric for metric in metrics if metric in set(averaged["metric"].unique())
    ]
    parts: list[pd.DataFrame] = []
    for control in controls:
        if control not in available:
            continue
        table = analyze_long(
            averaged,
            treatment=method,
            control=control,
            metrics=available_metrics,
            unit=unit,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            correction=correction,
        )
        parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


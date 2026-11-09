"""Cross-model pooling for external validation.

Internal CVC-ClinicDB runs are pooled with the fold/seed machinery in
:mod:`melba.eval.aggregate`: every case is a test unit exactly once.  External evaluation is
different - *the same* external cases are scored by every checkpoint of the grid (5 folds x
3 seeds per architecture), so the models are not extra cases and must not inflate the
sample size:

* :func:`case_averaged_frame` averages the metric values **per external case across the
  models**, which keeps one independent unit per image and is what the paired tests and the
  detection/uncertainty tables are computed on;
* :func:`external_model_spread` reports the **per-model spread** (mean / std / min / max
  across the checkpoints), i.e. how much the answer depends on which checkpoint of the
  internal grid one happens to draw;
* :func:`external_tables` bundles both with the standard paired, detection, false-positive
  and challenge-scoring tables, each carrying a ``scope: external`` column so no external
  number can be mistaken for an internal one.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from melba.stats import analyze_long

from .aggregate import (
    _majority,
    _quality_subset,
    challenge_scoring_table,
    detection_table,
    false_positive_table,
)

__all__ = [
    "EXTERNAL_SCOPE",
    "INTERNAL_SCOPE",
    "case_averaged_frame",
    "external_model_spread",
    "external_tables",
    "label_scope",
]

#: scope labels written next to every external table
EXTERNAL_SCOPE = "external"
INTERNAL_SCOPE = "internal"


def label_scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Add a ``scope`` column (``internal`` / ``external``) to a long-format frame."""
    labelled = frame.copy()
    labelled["scope"] = str(scope)
    return labelled


def case_averaged_frame(frames_by_model: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-case mean over models, one row per (case, method, metric).

    ``n_models`` records how many checkpoints contributed; the empty flag is the majority
    vote, so a case counts as a missed detection when most models predict nothing (the
    reference never changes between models).
    """
    if not frames_by_model:
        raise ValueError("case_averaged_frame needs at least one model")
    parts: list[pd.DataFrame] = []
    for label, frame in sorted(frames_by_model.items()):
        part = _quality_subset(frame, None).copy()
        part["model"] = str(label)
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)

    keys = ["case_id", "group_id", "method", "metric"]
    values = combined.groupby(keys, as_index=False).agg(
        value=("value", "mean"), n_models=("model", "nunique")
    )
    flags = (
        combined.drop_duplicates(["case_id", "method", "model"])
        .groupby(["case_id", "method"], as_index=False)
        .agg(
            empty_flag=("empty_flag", _majority),
            missed_fraction=("missed_detection", "mean"),
        )
    )
    merged = values.merge(flags, on=["case_id", "method"], how="left")
    merged["missed_detection"] = (merged["missed_fraction"] >= 0.5).astype(int)
    merged = merged.drop(columns=["missed_fraction"])
    merged["split"] = str(combined["split"].iloc[0]) if "split" in combined else "test"
    merged["fold"] = 0
    merged["seed"] = -1
    merged["run_id"] = "model_averaged"
    merged["scope"] = EXTERNAL_SCOPE
    return merged


def external_model_spread(
    frames_by_model: Mapping[str, pd.DataFrame],
    *,
    metrics: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per method/metric: the distribution of the result across the internal checkpoints.

    For every model the metric is averaged over the external cases (the same cases for
    every model, so the differences are checkpoint differences) and the table reports the
    mean, standard deviation, minimum and maximum of those per-model means.
    """
    if not frames_by_model:
        raise ValueError("external_model_spread needs at least one model")
    rows: list[dict[str, Any]] = []
    for label, frame in sorted(frames_by_model.items()):
        subset = _quality_subset(frame, metrics)
        if subset.empty:
            continue
        grouped = subset.groupby(["method", "metric"])["value"].agg(["mean", "count"])
        for (method, metric), values in grouped.iterrows():
            rows.append(
                {
                    "model": str(label),
                    "method": str(method),
                    "metric": str(metric),
                    "n_cases": int(values["count"]),
                    "mean": float(values["mean"]),
                }
            )
    per_model = pd.DataFrame(rows)
    if per_model.empty:
        return per_model
    if methods is not None:
        per_model = per_model.loc[per_model["method"].isin(list(methods))]
    spread = (
        per_model.groupby(["method", "metric"], as_index=False)
        .agg(
            n_models=("model", "nunique"),
            n_cases=("n_cases", "max"),
            mean=("mean", "mean"),
            std=("mean", "std"),
            min=("mean", "min"),
            max=("mean", "max"),
        )
        .sort_values(["method", "metric"])
        .reset_index(drop=True)
    )
    spread["range"] = spread["max"] - spread["min"]
    spread["scope"] = EXTERNAL_SCOPE
    return spread


def _paired_block(
    frame: pd.DataFrame,
    *,
    treatments: Sequence[str],
    control: str,
    metrics: Sequence[str],
    unit: str,
    n_bootstrap: int,
    ci: float,
    seed: int,
    correction: str,
    primary_test: str,
    alternative: str,
) -> pd.DataFrame:
    """Paired TTA-vs-control comparisons, one call per treatment, concatenated."""
    tables = [
        analyze_long(
            frame,
            treatment=treatment,
            control=control,
            metrics=list(metrics),
            unit=unit,
            primary_test=primary_test,
            alternative=alternative,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            correction=correction,
        )
        for treatment in treatments
    ]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def external_tables(
    frames_by_model: Mapping[str, pd.DataFrame],
    *,
    metrics: Sequence[str],
    control: str = "baseline",
    treatments: Sequence[str] = ("tta2", "tta4", "tta8"),
    unit: str = "group",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
    primary_test: str = "wilcoxon",
    alternative: str = "two-sided",
) -> dict[str, pd.DataFrame]:
    """All external tables: per-model spread plus case-averaged paired/detection results."""
    methods = sorted(
        {
            str(method)
            for frame in frames_by_model.values()
            for method in frame["method"].unique()
        }
    )
    detected = [control] + [name for name in treatments if name in methods]
    present = [name for name in treatments if name in methods]
    averaged = case_averaged_frame(frames_by_model)

    tables: dict[str, pd.DataFrame] = {
        "external_model_spread": external_model_spread(
            frames_by_model, metrics=metrics, methods=methods
        ),
    }
    if present:
        tables["external_paired"] = _paired_block(
            averaged,
            treatments=present,
            control=control,
            metrics=metrics,
            unit=unit,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            correction=correction,
            primary_test=primary_test,
            alternative=alternative,
        )
    tables["external_detection"] = detection_table(
        averaged,
        methods=detected,
        control=control,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        correction=correction,
    )
    tables["external_false_positive"] = false_positive_table(
        averaged,
        methods=detected,
        control=control,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        correction=correction,
    )
    tables["external_challenge_scoring"] = challenge_scoring_table(
        averaged, n_bootstrap=n_bootstrap, ci=ci, seed=seed
    )
    for name, table in tables.items():
        if table is not None and not table.empty and "scope" not in table.columns:
            table.insert(0, "scope", EXTERNAL_SCOPE)
    return tables



"""Detection-error analysis at both ends of the empty-mask spectrum.

Two failures are counted, and they are *not* symmetric:

* **missed detection** - empty prediction on a non-empty reference.  Under
  ``empty_policy: nan`` those cases silently drop out of distance-metric means, which
  biases the reported HD95/ASSD downwards (survivorship bias);
* **false positive on an empty reference** - a prediction with foreground where the
  reference has none.  BraTS 2021 ET has such patients (~3%), and under
  ``score_empty_reference: false`` they also leave the Dice mean, so the second failure
  mode is invisible unless it is reported explicitly.

For each direction the module provides:

* a per-method rate with a **cluster bootstrap** CI over groups (frames of one video, or
  repeat slices of one patient, are not independent);
* a transition table between the control and each treatment;
* an **exact McNemar test** on the discordant pairs.

Transition tables default to ``unit="case"`` (frames), matching the McNemar question;
``unit="group"`` (a video/patient counts as an event when any of its cases is one) is
available as a clustering-aware sensitivity check.  The false-positive analysis also
summarises the **predicted volume** of the false-positive cases when the per-case
``pred_volume_mm3`` metric is present in the long frame.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "analyze_detection",
    "analyze_false_positives",
    "empty_reference_transition_table",
    "false_positive_rate",
    "mcnemar_exact",
    "missed_detection_rate",
    "transition_table",
]

_REQUIRED_COLUMNS = ("case_id", "group_id", "method", "empty_flag")

#: per-case metric summarised for the false-positive cases
_VOLUME_METRIC = "pred_volume_mm3"

#: the two directions: which reference cases count, which flag is the event, and how the
#: rate columns and transition columns are named
_DIRECTION_SPECS: dict[str, dict[str, Any]] = {
    "missed": {
        "flags": ("both_nonempty", "pred_empty"),
        "event": "pred_empty",
        "prefix": "missed",
        "transitions": ("both_nonempty", "nonempty_to_empty", "empty_to_nonempty", "both_empty"),
    },
    "false_positive": {
        "flags": ("gt_empty", "both_empty"),
        "event": "gt_empty",
        "prefix": "false_positive",
        "transitions": ("both_correct", "correct_to_fp", "fp_to_correct", "both_fp"),
    },
}


def _direction(direction: str) -> dict[str, Any]:
    try:
        return _DIRECTION_SPECS[direction]
    except KeyError:  # pragma: no cover - guarded by the public wrappers
        raise ValueError(
            f"direction must be one of {sorted(_DIRECTION_SPECS)}, got {direction!r}"
        ) from None


def _case_flags(
    frame: pd.DataFrame,
    method: str,
    *,
    direction: str = "missed",
    with_volume: bool = False,
) -> pd.DataFrame:
    """One row per case for ``method``: its empty flag and the error bit.

    ``direction="missed"`` keeps the cases whose *reference* is non-empty and sets the
    event bit for empty predictions; ``direction="false_positive"`` keeps the cases whose
    reference is empty and sets the bit for predictions that found something.
    """
    spec = _direction(direction)
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Long-format frame is missing column(s) {missing}")
    rows = frame.loc[frame["method"] == method, ["case_id", "group_id", "empty_flag"]]
    if rows.empty:
        raise ValueError(f"Method {method!r} not found in the frame")
    unique = rows.drop_duplicates("case_id").copy()
    unique["event"] = (unique["empty_flag"] == spec["event"]).astype(int)
    # only cases whose reference has (non-)foreground can be an event in this direction
    valid = unique["empty_flag"].isin(spec["flags"])
    unique = unique.loc[valid].reset_index(drop=True)
    if with_volume:
        unique[_VOLUME_METRIC] = _case_volume(frame, method, unique["case_id"])
    return unique


def _case_volume(frame: pd.DataFrame, method: str, case_ids: pd.Series) -> np.ndarray:
    """Per-case ``pred_volume_mm3`` aligned with ``case_ids`` (``nan`` when absent)."""
    empty = np.full(len(case_ids), np.nan, dtype=float)
    if "metric" not in frame.columns or "value" not in frame.columns:
        return empty
    subset = frame.loc[
        (frame["method"] == method) & (frame["metric"] == _VOLUME_METRIC),
        ["case_id", "value"],
    ]
    if subset.empty:
        return empty
    mapping = dict(
        zip(subset["case_id"].astype(str), subset["value"].astype(float))
    )
    return np.array([mapping.get(str(case), np.nan) for case in case_ids], dtype=float)



def _event_rate(
    frame: pd.DataFrame,
    method: str,
    *,
    direction: str,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    unit_label: str = "cluster_bootstrap_video",
) -> dict[str, Any]:
    """Rate of one detection error with a cluster (group-level) bootstrap CI."""
    spec = _direction(direction)
    prefix = str(spec["prefix"])
    rows = _case_flags(
        frame, method, direction=direction, with_volume=direction == "false_positive"
    )
    n_cases = int(len(rows))
    n_events = int(rows["event"].sum())
    rate = n_events / n_cases if n_cases else float("nan")

    result: dict[str, Any] = {
        "method": method,
        "n_cases": n_cases,
        "n_videos": int(rows["group_id"].nunique()) if n_cases else 0,
        f"n_{prefix}": n_events,
        f"{prefix}_rate": float(rate),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "unit": unit_label,
    }
    if direction == "false_positive":
        result.update(_volume_summary(rows))
    if n_cases == 0 or n_events == 0:
        # No failures: the percentile bootstrap is degenerate at zero.
        result["ci_low"] = 0.0
        result["ci_high"] = 0.0
        return result

    grouped = rows.groupby("group_id")["event"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, len(sums), size=(int(n_bootstrap), len(sums)))
    sampled_sums = sums[draw].sum(axis=1)
    sampled_counts = counts[draw].sum(axis=1)
    rates = sampled_sums / sampled_counts
    alpha = (1.0 - float(ci)) / 2.0
    result["ci_low"] = float(np.percentile(rates, 100 * alpha))
    result["ci_high"] = float(np.percentile(rates, 100 * (1 - alpha)))
    return result


def _volume_summary(rows: pd.DataFrame) -> dict[str, float]:
    """Predicted-volume statistics of the false-positive cases (``nan`` when unknown)."""
    if _VOLUME_METRIC not in rows.columns:
        return {}
    events = rows.loc[rows["event"] == 1, _VOLUME_METRIC].to_numpy(dtype=float)
    events = events[np.isfinite(events)]
    if events.size == 0:
        return {
            f"{_VOLUME_METRIC}_mean": float("nan"),
            f"{_VOLUME_METRIC}_median": float("nan"),
            f"{_VOLUME_METRIC}_max": float("nan"),
            f"{_VOLUME_METRIC}_total": float("nan"),
        }
    return {
        f"{_VOLUME_METRIC}_mean": float(np.mean(events)),
        f"{_VOLUME_METRIC}_median": float(np.median(events)),
        f"{_VOLUME_METRIC}_max": float(np.max(events)),
        f"{_VOLUME_METRIC}_total": float(np.sum(events)),
    }


def missed_detection_rate(
    frame: pd.DataFrame,
    method: str,
    *,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Missed-detection rate with a cluster (video-level) bootstrap CI."""
    return _event_rate(
        frame,
        method,
        direction="missed",
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
    )


def false_positive_rate(
    frame: pd.DataFrame,
    method: str,
    *,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """False-positive rate **on empty references**, with a cluster bootstrap CI.

    The denominator is the number of cases whose reference mask is empty (BraTS: patients
    without enhancing tumour), not the whole fold: a model cannot "false-positive" on a
    patient that does have a lesion, it can only miss it.  ``pred_volume_mm3_mean`` /
    ``_median`` / ``_max`` / ``_total`` summarise how much tumour the false positives
    invented, when the per-case volume metric was computed.
    """
    return _event_rate(
        frame,
        method,
        direction="false_positive",
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        unit_label="cluster_bootstrap_group",
    )



def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for ``b`` and ``c`` discordant pairs."""
    b, c = int(b), int(c)
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be non-negative, got {b} and {c}")
    if b + c == 0:
        return 1.0
    return float(stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)


def _transition_counts(
    frame: pd.DataFrame,
    treatment: str,
    *,
    control: str = "baseline",
    unit: str = "case",
    direction: str = "missed",
) -> tuple[dict[str, int], int]:
    """Concordant/discordant counts between ``control`` and ``treatment``.

    The four cells are returned in the order of the direction's ``transitions`` spec
    (both clear, control -> treatment event, treatment -> control event, both events),
    together with the number of units.
    """
    if unit not in {"case", "group"}:
        raise ValueError(f"unit must be 'case' or 'group', got {unit!r}")
    control_rows = _case_flags(frame, control, direction=direction)
    treatment_rows = _case_flags(frame, treatment, direction=direction)

    merged = control_rows.merge(
        treatment_rows, on=["case_id", "group_id"], suffixes=("_control", "_treatment")
    )
    merged["control_event"] = merged["event_control"].astype(bool)
    merged["treatment_event"] = merged["event_treatment"].astype(bool)

    if unit == "group":
        merged = (
            merged.groupby("group_id")[["control_event", "treatment_event"]]
            .any()
            .reset_index()
        )

    cells = (
        int((~merged["control_event"] & ~merged["treatment_event"]).sum()),
        int((~merged["control_event"] & merged["treatment_event"]).sum()),
        int((merged["control_event"] & ~merged["treatment_event"]).sum()),
        int((merged["control_event"] & merged["treatment_event"]).sum()),
    )
    names = tuple(str(name) for name in _direction(direction)["transitions"])
    return dict(zip(names, cells)), int(len(merged))


def transition_table(
    frame: pd.DataFrame,
    treatment: str,
    *,
    control: str = "baseline",
    unit: str = "case",
) -> dict[str, Any]:
    """Detection transitions between ``control`` and ``treatment``."""
    cells, n_units = _transition_counts(
        frame, treatment, control=control, unit=unit, direction="missed"
    )
    both_ok, control_only, treatment_only, both_failed = tuple(
        str(name) for name in _DIRECTION_SPECS["missed"]["transitions"]
    )
    return {
        "method": treatment,
        "control": control,
        "unit": unit,
        "n": n_units,
        both_ok: cells[both_ok],
        control_only: cells[control_only],
        treatment_only: cells[treatment_only],
        both_failed: cells[both_failed],
        "mcnemar_p": mcnemar_exact(cells[control_only], cells[treatment_only]),
    }


def empty_reference_transition_table(
    frame: pd.DataFrame,
    treatment: str,
    *,
    control: str = "baseline",
    unit: str = "case",
) -> dict[str, Any]:
    """False-positive transitions on empty references (control vs treatment).

    ``correct_to_fp`` counts empty-reference cases that the control left empty and the
    treatment filled (a *new* false positive), ``fp_to_correct`` the opposite direction;
    the exact McNemar test is computed on those two discordant cells.
    """
    cells, n_units = _transition_counts(
        frame, treatment, control=control, unit=unit, direction="false_positive"
    )
    both_ok, control_only, treatment_only, both_failed = tuple(
        str(name) for name in _DIRECTION_SPECS["false_positive"]["transitions"]
    )
    return {
        "method": treatment,
        "control": control,
        "unit": unit,
        "n": n_units,
        both_ok: cells[both_ok],
        control_only: cells[control_only],
        treatment_only: cells[treatment_only],
        both_failed: cells[both_failed],
        "mcnemar_p": mcnemar_exact(cells[control_only], cells[treatment_only]),
    }



def _analyze(
    frame: pd.DataFrame,
    *,
    direction: str,
    control: str,
    methods: Sequence[str] | None,
    unit: str,
    n_bootstrap: int,
    ci: float,
    seed: int,
    correction: str,
) -> pd.DataFrame:
    """Rates for every method plus transitions/McNemar against the control."""
    spec = _direction(direction)
    prefix = str(spec["prefix"])
    names = tuple(str(name) for name in spec["transitions"])
    available = [str(method) for method in frame["method"].dropna().unique()]
    if methods is None:
        methods = [control] + [method for method in sorted(available) if method != control]
    else:
        methods = list(methods)
        if control not in methods:
            methods = [control] + methods

    rows: list[dict[str, Any]] = []
    for method in methods:
        row = _event_rate(
            frame,
            method,
            direction=direction,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            unit_label=(
                "cluster_bootstrap_group"
                if direction == "false_positive"
                else "cluster_bootstrap_video"
            ),
        )
        if method != control:
            transitions = (
                empty_reference_transition_table(frame, method, control=control, unit=unit)
                if direction == "false_positive"
                else transition_table(frame, method, control=control, unit=unit)
            )
            row.update(
                {
                    key: value
                    for key, value in transitions.items()
                    if key not in {"method", "control", "unit"}
                }
            )
        else:
            control_rows = _case_flags(frame, control, direction=direction)
            unit_size = (
                int(control_rows["group_id"].nunique())
                if unit == "group"
                else int(len(control_rows))
            )
            row.update(
                {
                    "n": unit_size,
                    names[0]: float("nan"),
                    names[1]: float("nan"),
                    names[2]: float("nan"),
                    names[3]: float("nan"),
                    "mcnemar_p": float("nan"),
                }
            )
        row["control"] = control
        row["transition_unit"] = unit
        rows.append(row)

    table = pd.DataFrame(rows)
    if correction == "holm":
        from .paired import holm_adjust

        table["mcnemar_p_adjusted"] = holm_adjust(table["mcnemar_p"].to_numpy(dtype=float))
    elif correction in (None, "none"):
        table["mcnemar_p_adjusted"] = table["mcnemar_p"]
    else:
        raise ValueError(f"Unknown correction {correction!r}; expected 'holm' or 'none'")

    columns = [
        "method",
        "control",
        "n_cases",
        "n_videos",
        f"n_{prefix}",
        f"{prefix}_rate",
        "ci_low",
        "ci_high",
        "unit",
        "n",
        *names,
        "mcnemar_p",
        "mcnemar_p_adjusted",
        "transition_unit",
    ]
    if direction == "false_positive":
        columns.extend(
            f"{_VOLUME_METRIC}_{statistic}"
            for statistic in ("mean", "median", "max", "total")
        )
    return table[[column for column in columns if column in table.columns]]


def analyze_detection(
    frame: pd.DataFrame,
    *,
    control: str = "baseline",
    methods: Sequence[str] | None = None,
    unit: str = "case",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
) -> pd.DataFrame:
    """Missed-detection rates for every method plus transitions/McNemar vs the control."""
    return _analyze(
        frame,
        direction="missed",
        control=control,
        methods=methods,
        unit=unit,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        correction=correction,
    )


def analyze_false_positives(
    frame: pd.DataFrame,
    *,
    control: str = "baseline",
    methods: Sequence[str] | None = None,
    unit: str = "case",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
) -> pd.DataFrame:
    """False-positive rates on **empty references**, transitions and McNemar vs control.

    The mirror image of :func:`analyze_detection`: the cohort is the set of cases whose
    reference mask is empty, the event is a prediction with foreground on such a case, and
    the transition columns count how many empty-reference cases each method fills in
    (``correct_to_fp``) or leaves empty (``fp_to_correct``) relative to the control.
    ``pred_volume_mm3_*`` columns summarise the invented volume (they are absent from the
    table when the per-case volume metric was not computed).
    """
    return _analyze(
        frame,
        direction="false_positive",
        control=control,
        methods=methods,
        unit=unit,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        correction=correction,
    )



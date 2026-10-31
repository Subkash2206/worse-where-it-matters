"""Missed-detection analysis: empty predictions on non-empty references.

An empty prediction on a frame that contains a lesion is a *failure to detect*, not
a small segmentation error.  Under ``empty_policy: nan`` those cases silently drop
out of distance-metric means, which biases the reported HD95/ASSD downwards
(survivorship bias).  This module makes the failure explicit:

* per-method missed-detection rate with a **cluster bootstrap** CI over videos
  (frames of one video are not independent);
* a transition table between the control and each treatment
  (``nonempty -> empty``, ``empty -> nonempty``, both nonempty, both empty);
* an **exact McNemar test** on the discordant pairs.

Transition tables default to ``unit="case"`` (frames), matching the McNemar
question; ``unit="group"`` (a video counts as missed when any of its frames is
missed) is available as a clustering-aware sensitivity check.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "analyze_detection",
    "mcnemar_exact",
    "missed_detection_rate",
    "transition_table",
]

_REQUIRED_COLUMNS = ("case_id", "group_id", "method", "empty_flag")


def _case_flags(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    """One row per case for ``method`` with its empty flag and missed-detection bit."""
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Long-format frame is missing column(s) {missing}")
    rows = frame.loc[frame["method"] == method, ["case_id", "group_id", "empty_flag"]]
    if rows.empty:
        raise ValueError(f"Method {method!r} not found in the frame")
    unique = rows.drop_duplicates("case_id").copy()
    unique["missed"] = (unique["empty_flag"] == "pred_empty").astype(int)
    # cases without a reference lesion cannot be missed detections
    valid = unique["empty_flag"].isin(["both_nonempty", "pred_empty"])
    return unique.loc[valid].reset_index(drop=True)


def missed_detection_rate(
    frame: pd.DataFrame,
    method: str,
    *,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Missed-detection rate with a cluster (video-level) bootstrap CI."""
    rows = _case_flags(frame, method)
    n_cases = int(len(rows))
    n_missed = int(rows["missed"].sum())
    rate = n_missed / n_cases if n_cases else float("nan")

    result: dict[str, Any] = {
        "method": method,
        "n_cases": n_cases,
        "n_videos": int(rows["group_id"].nunique()),
        "n_missed": n_missed,
        "missed_rate": float(rate),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "unit": "cluster_bootstrap_video",
    }
    if n_cases == 0 or n_missed == 0:
        # No failures: the percentile bootstrap is degenerate at zero.
        result["ci_low"] = 0.0
        result["ci_high"] = 0.0
        return result

    grouped = rows.groupby("group_id")["missed"].agg(["sum", "count"])
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


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for ``b`` and ``c`` discordant pairs."""
    b, c = int(b), int(c)
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be non-negative, got {b} and {c}")
    if b + c == 0:
        return 1.0
    return float(stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)


def transition_table(
    frame: pd.DataFrame,
    treatment: str,
    *,
    control: str = "baseline",
    unit: str = "case",
) -> dict[str, Any]:
    """Detection transitions between ``control`` and ``treatment``."""
    if unit not in {"case", "group"}:
        raise ValueError(f"unit must be 'case' or 'group', got {unit!r}")
    control_rows = _case_flags(frame, control)
    treatment_rows = _case_flags(frame, treatment)

    merged = control_rows.merge(
        treatment_rows, on=["case_id", "group_id"], suffixes=("_control", "_treatment")
    )
    merged["control_empty"] = merged["empty_flag_control"] == "pred_empty"
    merged["treatment_empty"] = merged["empty_flag_treatment"] == "pred_empty"

    if unit == "group":
        merged = (
            merged.groupby("group_id")[["control_empty", "treatment_empty"]]
            .any()
            .reset_index()
        )

    both_nonempty = int((~merged["control_empty"] & ~merged["treatment_empty"]).sum())
    nonempty_to_empty = int((~merged["control_empty"] & merged["treatment_empty"]).sum())
    empty_to_nonempty = int((merged["control_empty"] & ~merged["treatment_empty"]).sum())
    both_empty = int((merged["control_empty"] & merged["treatment_empty"]).sum())
    return {
        "method": treatment,
        "control": control,
        "unit": unit,
        "n": int(len(merged)),
        "both_nonempty": both_nonempty,
        "nonempty_to_empty": nonempty_to_empty,
        "empty_to_nonempty": empty_to_nonempty,
        "both_empty": both_empty,
        "mcnemar_p": mcnemar_exact(nonempty_to_empty, empty_to_nonempty),
    }


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
    """Rates for every method plus transitions/McNemar against the control."""
    available = [str(method) for method in frame["method"].dropna().unique()]
    if methods is None:
        methods = [control] + [method for method in sorted(available) if method != control]
    else:
        methods = list(methods)
        if control not in methods:
            methods = [control] + methods

    rows: list[dict[str, Any]] = []
    for method in methods:
        row = missed_detection_rate(
            frame, method, n_bootstrap=n_bootstrap, ci=ci, seed=seed
        )
        if method != control:
            transitions = transition_table(frame, method, control=control, unit=unit)
            row.update(
                {
                    key: value
                    for key, value in transitions.items()
                    if key not in {"method", "control", "unit"}
                }
            )
        else:
            control_rows = _case_flags(frame, control)
            unit_size = (
                int(control_rows["group_id"].nunique())
                if unit == "group"
                else int(len(control_rows))
            )
            row.update(
                {
                    "n": unit_size,
                    "both_nonempty": float("nan"),
                    "nonempty_to_empty": float("nan"),
                    "empty_to_nonempty": float("nan"),
                    "both_empty": float("nan"),
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
        "n_missed",
        "missed_rate",
        "ci_low",
        "ci_high",
        "unit",
        "n",
        "both_nonempty",
        "nonempty_to_empty",
        "empty_to_nonempty",
        "both_empty",
        "mcnemar_p",
        "mcnemar_p_adjusted",
        "transition_unit",
    ]
    return table[columns]


"""Missed-detection rate, transition table and exact McNemar test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from melba.stats import (
    analyze_detection,
    mcnemar_exact,
    missed_detection_rate,
    transition_table,
)


def _frame() -> pd.DataFrame:
    """Two videos, six frames; baseline misses c3; tta8 misses c2 and c3."""
    flags = {
        "c1": ("both_nonempty", "both_nonempty"),
        "c2": ("both_nonempty", "pred_empty"),
        "c3": ("pred_empty", "pred_empty"),
        "c4": ("both_nonempty", "both_nonempty"),
        "c5": ("pred_empty", "both_nonempty"),
        "c6": ("both_nonempty", "both_nonempty"),
    }
    groups = {"c1": "v1", "c2": "v1", "c3": "v1", "c4": "v2", "c5": "v2", "c6": "v2"}
    rows = []
    for case, (baseline_flag, tta_flag) in flags.items():
        for method, flag in (("baseline", baseline_flag), ("tta8", tta_flag)):
            rows.append(
                {
                    "case_id": case,
                    "group_id": groups[case],
                    "method": method,
                    "metric": "dice",
                    "value": 0.0 if flag == "pred_empty" else 0.8,
                    "empty_flag": flag,
                    "missed_detection": int(flag == "pred_empty"),
                }
            )
    return pd.DataFrame(rows)


def test_mcnemar_exact_matches_binomial_test() -> None:
    assert mcnemar_exact(5, 1) == pytest.approx(
        stats.binomtest(5, 6, 0.5, alternative="two-sided").pvalue
    )
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_missed_detection_rate_and_ci() -> None:
    frame = _frame()
    baseline = missed_detection_rate(frame, "baseline", n_bootstrap=500, seed=0)
    assert baseline["n_cases"] == 6
    assert baseline["n_videos"] == 2
    assert baseline["n_missed"] == 2  # c3, c5
    assert baseline["missed_rate"] == pytest.approx(2 / 6)
    assert 0.0 <= baseline["ci_low"] <= baseline["missed_rate"] <= baseline["ci_high"] <= 1.0

    tta = missed_detection_rate(frame, "tta8", n_bootstrap=500, seed=0)
    assert tta["n_missed"] == 2  # c2, c3
    assert tta["missed_rate"] == pytest.approx(2 / 6)

    # deterministic given the seed
    assert missed_detection_rate(frame, "baseline", n_bootstrap=500, seed=0) == baseline

    # no misses -> degenerate CI at zero
    perfect = frame.copy()
    perfect["empty_flag"] = "both_nonempty"
    zero = missed_detection_rate(perfect, "baseline", n_bootstrap=100)
    assert zero["n_missed"] == 0 and zero["ci_low"] == zero["ci_high"] == 0.0


def test_transition_table_counts_and_unit() -> None:
    frame = _frame()
    case_table = transition_table(frame, "tta8", control="baseline", unit="case")
    assert case_table["n"] == 6
    assert case_table["both_nonempty"] == 3  # c1, c4, c6
    assert case_table["nonempty_to_empty"] == 1  # c2: baseline detected, TTA missed
    assert case_table["empty_to_nonempty"] == 1  # c5: baseline missed, TTA detected
    assert case_table["both_empty"] == 1  # c3
    assert case_table["mcnemar_p"] == pytest.approx(1.0)

    video_table = transition_table(frame, "tta8", control="baseline", unit="group")
    assert video_table["n"] == 2
    # v1 is missed by both (any-frame rule), v2 is missed by the baseline only
    assert video_table["both_empty"] == 1
    assert video_table["empty_to_nonempty"] == 1
    assert video_table["nonempty_to_empty"] == 0
    assert video_table["mcnemar_p"] == pytest.approx(1.0)

    # the control row of analyze_detection reports the number of units, not cases
    video_analysis = analyze_detection(frame, control="baseline", unit="group", n_bootstrap=100)
    assert video_analysis.set_index("method").loc["baseline", "n"] == 2
    case_analysis = analyze_detection(frame, control="baseline", unit="case", n_bootstrap=100)
    assert case_analysis.set_index("method").loc["baseline", "n"] == 6

    with pytest.raises(ValueError, match="unit"):
        transition_table(frame, "tta8", unit="patient")


def test_analyze_detection_table() -> None:
    frame = _frame()
    table = analyze_detection(frame, control="baseline", unit="case", n_bootstrap=300, seed=0)
    assert table["method"].tolist() == ["baseline", "tta8"]
    control_row = table.set_index("method").loc["baseline"]
    assert np.isnan(control_row["mcnemar_p"])
    treatment_row = table.set_index("method").loc["tta8"]
    assert treatment_row["nonempty_to_empty"] == 1
    assert treatment_row["empty_to_nonempty"] == 1
    assert 0.0 <= treatment_row["mcnemar_p_adjusted"] <= 1.0
    assert table["transition_unit"].unique().tolist() == ["case"]

    with pytest.raises(ValueError, match="Unknown correction"):
        analyze_detection(frame, correction="bonferroni")
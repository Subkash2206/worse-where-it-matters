"""The rule-selection and rule-report helpers, on synthetic long-format frames.

Both scripts run *after* hours of GPU work, so their logic is pinned here: the selection
rule must be mechanical and reproducible, and the miss/false-positive tables must count
cases (not rows) and test the discordant pairs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import detection_rules_report as report  # noqa: E402
import select_detection_rule as selection  # noqa: E402


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build a long-format frame with the columns the scripts read."""
    return pd.DataFrame(rows)


def _row(case, method, metric, value, *, missed=0, flag="both_nonempty", run="r1"):
    return {
        "run_id": run,
        "case_id": case,
        "group_id": case,
        "method": method,
        "metric": metric,
        "value": value,
        "missed_detection": missed,
        "empty_flag": flag,
        "fold": 0,
        "seed": 0,
        "dataset": "cvc_clinicdb",
        "split": "test",
    }


# ------------------------------------------------------------------ selection rule


def _selection_table(rows: list[tuple[str, float, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": method,
                "n_checkpoints": 15,
                "n_case_evaluations": 1836,
                "n_cases": 612,
                "seeds_per_case": 3.0,
                "dice_mean": dice,
                "dice_median": dice,
                "missed_checkpoints": missed,
                "missed_checkpoint_rate_pct": 100.0 * missed / 1836,
                "cases_missed_any": missed,
                "cases_majority_missed": majority,
            }
            for method, dice, missed, majority in rows
        ]
    )


def test_choose_prefers_the_fewest_misses_within_the_dice_budget() -> None:
    table = _selection_table(
        [
            ("tta8", 0.8046, 69, 19),
            ("tta8_lowdet_20", 0.8124, 21, 8),
            ("tta8_lowdet_30", 0.8085, 39, 14),
            ("tta8_lowdet_40", 0.8057, 57, 17),
        ]
    )
    decision = selection.choose(
        table, "tta8", ["tta8_lowdet_20", "tta8_lowdet_30", "tta8_lowdet_40"], 0.002
    )
    assert decision["chosen_method"] == "tta8_lowdet_20"
    assert decision["chosen_threshold"] == pytest.approx(0.20)
    assert decision["rejected"] == []
    assert [entry["method"] for entry in decision["eligible"]] == [
        "tta8_lowdet_20",
        "tta8_lowdet_30",
        "tta8_lowdet_40",
    ]


def test_choose_rejects_a_candidate_that_costs_too_much_dice() -> None:
    table = _selection_table(
        [
            ("tta8", 0.80, 69, 19),
            ("tta8_lowdet_20", 0.79, 5, 2),  # most misses saved, but -0.01 Dice
            ("tta8_lowdet_30", 0.7995, 30, 11),
        ]
    )
    decision = selection.choose(table, "tta8", ["tta8_lowdet_20", "tta8_lowdet_30"], 0.002)
    assert decision["chosen_method"] == "tta8_lowdet_30"
    assert [entry["method"] for entry in decision["rejected"]] == ["tta8_lowdet_20"]


def test_choose_refuses_when_every_candidate_breaks_the_budget() -> None:
    table = _selection_table([("tta8", 0.80, 69, 19), ("tta8_lowdet_20", 0.78, 5, 2)])
    with pytest.raises(SystemExit, match="not admissible"):
        selection.choose(table, "tta8", ["tta8_lowdet_20"], 0.002)


def test_choose_needs_the_reference_method() -> None:
    table = _selection_table([("tta8_lowdet_20", 0.81, 21, 8)])
    with pytest.raises(SystemExit, match="reference method"):
        selection.choose(table, "tta8", ["tta8_lowdet_20"], 0.002)


def test_method_summary_counts_cases_not_rows() -> None:
    rows = []
    # one case, three seeds: missed by two of them -> a majority miss, three checkpoint misses
    for seed, missed in enumerate([1, 1, 0]):
        rows.append(_row("c1", "tta8", "dice", 0.5, missed=missed, run=f"r{seed}"))
        rows.append(_row("c2", "tta8", "dice", 0.9, missed=0, run=f"r{seed}"))
    table = selection.method_summary(_frame(rows), ["tta8"])
    row = table.iloc[0]
    assert row["n_cases"] == 2
    assert row["missed_checkpoints"] == 2
    assert row["cases_missed_any"] == 1
    assert row["cases_majority_missed"] == 1  # 2 of 3 seeds
    assert row["dice_mean"] == pytest.approx((0.5 + 0.9) / 2)


# ---------------------------------------------------------------- report helpers


def test_unit_averaged_pools_checkpoints_and_keeps_the_empty_flag() -> None:
    rows = [
        _row("c1", "tta8", "dice", 0.4, missed=1, flag="pred_empty", run="r1"),
        _row("c1", "tta8", "dice", 0.6, missed=0, flag="both_nonempty", run="r2"),
        _row("c1", "tta8", "dice", 0.7, missed=0, flag="both_nonempty", run="r3"),
        _row("c1", "tta8", "hd95", 10.0, missed=1, flag="pred_empty", run="r1"),
        _row("c1", "tta8", "hd95", 20.0, missed=0, flag="both_nonempty", run="r2"),
        _row("c1", "tta8", "hd95", 30.0, missed=0, flag="both_nonempty", run="r3"),
        _row("c1", "tta8", "jaccard", 0.3, missed=0, flag="both_nonempty", run="r1"),
    ]
    frame = report.unit_averaged(_frame(rows), "case")
    dice = frame.loc[frame["metric"] == "dice"].iloc[0]
    assert dice["value"] == pytest.approx((0.4 + 0.6 + 0.7) / 3)
    assert dice["empty_flag"] == "both_nonempty"  # the majority state
    # only the reported metrics survive
    assert set(frame["metric"]) == {"dice", "hd95"}


def test_unit_averaged_tie_breaks_towards_a_miss() -> None:
    rows = [
        _row("c1", "tta8", "dice", 0.4, missed=1, flag="pred_empty", run="r1"),
        _row("c1", "tta8", "dice", 0.6, missed=0, flag="both_nonempty", run="r2"),
    ]
    frame = report.unit_averaged(_frame(rows), "case")
    assert frame.iloc[0]["empty_flag"] == "pred_empty"


def test_miss_table_counts_majority_misses_and_tests_the_discordant_pairs() -> None:
    rows = []
    # c1: tta8 misses in 8/15 checkpoints, tta8_anyview never -> one case saved
    for index in range(15):
        rows.append(_row("c1", "tta8", "dice", 0.0, missed=int(index < 8), flag="pred_empty", run=f"r{index}"))
        rows.append(_row("c1", "tta8_anyview", "dice", 0.5, missed=0, run=f"r{index}"))
    # c2: both miss everywhere
    for index in range(15):
        rows.append(_row("c2", "tta8", "dice", 0.0, missed=1, flag="pred_empty", run=f"r{index}"))
        rows.append(_row("c2", "tta8_anyview", "dice", 0.0, missed=1, flag="pred_empty", run=f"r{index}"))
    table = report.miss_table(_frame(rows), ["tta8", "tta8_anyview"], reference="tta8").set_index("method")
    assert table.loc["tta8", "units_majority_missed"] == 2
    assert table.loc["tta8_anyview", "units_majority_missed"] == 1
    assert table.loc["tta8_anyview", "new_vs_reference"] == 0
    assert table.loc["tta8_anyview", "left_vs_reference"] == 1
    assert table.loc["tta8_anyview", "mcnemar_p_vs_reference"] == pytest.approx(1.0)
    assert table.loc["tta8_anyview", "missed_checkpoints"] == 15


def test_false_positive_table_separates_lesion_free_from_missed_et() -> None:
    rows = []
    for run in ("r1", "r2", "r3"):
        # patient A has no ET reference: tta8 stays (correctly) silent, the rule predicts
        rows.append(_row("A", "tta8", "dice", float("nan"), flag="both_empty", run=run))
        rows.append(_row("A", "tta8_anyview", "dice", 0.0, flag="gt_empty", run=run))
        rows.append(_row("A", "tta8_anyview", "pred_volume_mm3", 120.0, flag="gt_empty", run=run))
        # patient B has an ET: tta8 misses it, the rule finds it
        rows.append(_row("B", "tta8", "dice", 0.0, flag="pred_empty", missed=1, run=run))
        rows.append(_row("B", "tta8_anyview", "dice", 0.4, flag="both_nonempty", run=run))
        rows.append(_row("B", "tta8_anyview", "pred_volume_mm3", 900.0, flag="both_nonempty", run=run))
    table = report.false_positive_table(_frame(rows), ["tta8", "tta8_anyview"], reference="tta8")
    table = table.set_index("method")
    assert table.loc["tta8", "lesion_free_patients"] == 1
    assert table.loc["tta8", "false_positive_patients"] == 0
    assert table.loc["tta8", "missed_et_patients"] == 1
    assert table.loc["tta8_anyview", "false_positive_patients"] == 1
    assert table.loc["tta8_anyview", "false_positive_rate_pct"] == pytest.approx(100.0)
    assert table.loc["tta8_anyview", "median_fp_volume_mm3"] == pytest.approx(120.0)
    assert table.loc["tta8_anyview", "missed_et_patients"] == 0
    assert table.loc["tta8_anyview", "fp_new_vs_reference"] == 1
    assert table.loc["tta8_anyview", "missed_et_left_vs_reference"] == 1


def test_false_positive_table_is_empty_without_the_reference_method() -> None:
    rows = [_row("A", "tta8_anyview", "dice", 0.0, flag="gt_empty")]
    assert report.false_positive_table(_frame(rows), ["tta8_anyview"], reference="tta8").empty


def test_load_dir_refuses_an_empty_directory(tmp_path) -> None:
    with pytest.raises(SystemExit, match="no per-case CSVs"):
        report.load_dir(tmp_path)


"""False positives on empty references, and BraTS challenge-style scoring."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from melba.eval import (
    CHALLENGE_METRIC,
    challenge_scoring_table,
    challenge_style_dice,
    false_positive_table,
)
from melba.stats import (
    analyze_false_positives,
    empty_reference_transition_table,
    false_positive_rate,
)


def _frame() -> pd.DataFrame:
    """Six BraTS-like cases: two without enhancing tumour, four with.

    Every case also carries a ``pred_volume_mm3`` row, as the real per-case CSV does.
    ``empty_flag`` is whatever the evaluation wrote: under our protocol the Dice of an
    empty-reference case is NaN (undefined), not 1.
    """
    flags = {
        "c1": ({"baseline": "both_nonempty", "tta4": "both_nonempty"}, 900.0),
        "c2": ({"baseline": "pred_empty", "tta4": "both_nonempty"}, 120.0),
        "c3": ({"baseline": "both_empty", "tta4": "both_empty"}, 0.0),
        "c4": ({"baseline": "gt_empty", "tta4": "both_empty"}, 480.0),
        "c5": ({"baseline": "both_nonempty", "tta4": "both_nonempty"}, 700.0),
        "c6": ({"baseline": "both_nonempty", "tta4": "pred_empty"}, 60.0),
    }
    dice = {
        "both_nonempty": 0.8,
        "pred_empty": 0.0,
        "gt_empty": float("nan"),
        "both_empty": float("nan"),
    }
    rows = []
    for case, (method_flags, volume) in flags.items():
        for method, flag in method_flags.items():
            base = {
                # the provenance columns of a real per-case CSV (pool_folds needs `split`)
                "run_id": "run-fold0-seed0",
                "created_utc": "2026-01-01T00:00:00Z",
                "config_path": "configs/brats2021/segresnet.yaml",
                "config_hash": "deadbeef",
                "seed": 0,
                "dataset": "brats2021",
                "model": "segresnet_3d",
                "split": "test",
                "fold": 0,
                "method": method,
                "aggregator": "mean",
                "n_views": 1,
                "view_names": "identity",
                "threshold": 0.5,
                "spacing_mm": "1,1,1",
                "case_id": case,
                "group_id": case,
                "empty_flag": flag,
                "missed_detection": int(flag == "pred_empty"),
            }
            rows.append({**base, "metric": "dice", "value": dice[flag]})
            rows.append({**base, "metric": "pred_volume_mm3", "value": volume})
    return pd.DataFrame(rows)


def test_false_positive_rate_counts_only_empty_references() -> None:
    frame = _frame()
    baseline = false_positive_rate(frame, "baseline", n_bootstrap=500, seed=0)

    # c3 and c4 have no reference ET; only c4 is a false positive for the baseline
    assert baseline["n_cases"] == 2
    assert baseline["n_false_positive"] == 1
    assert baseline["false_positive_rate"] == pytest.approx(0.5)
    assert (
        0.0
        <= baseline["ci_low"]
        <= baseline["false_positive_rate"]
        <= baseline["ci_high"]
        <= 1.0
    )
    # the invented volume is reported in mm^3
    assert baseline["pred_volume_mm3_mean"] == pytest.approx(480.0)
    assert baseline["pred_volume_mm3_max"] == pytest.approx(480.0)
    assert baseline["pred_volume_mm3_total"] == pytest.approx(480.0)

    tta = false_positive_rate(frame, "tta4", n_bootstrap=500, seed=0)
    assert tta["n_false_positive"] == 0
    assert tta["false_positive_rate"] == 0.0
    assert tta["ci_low"] == tta["ci_high"] == 0.0
    assert math.isnan(tta["pred_volume_mm3_mean"])

    # deterministic for a given seed
    assert false_positive_rate(frame, "baseline", n_bootstrap=500, seed=0) == baseline


def test_false_positive_transitions_and_mcnemar() -> None:
    frame = _frame()
    table = empty_reference_transition_table(frame, "tta4", control="baseline", unit="case")
    assert table["n"] == 2  # c3, c4
    assert table["both_correct"] == 1  # c3: both predict nothing
    assert table["fp_to_correct"] == 1  # c4: baseline false-positived, tta4 did not
    assert table["correct_to_fp"] == 0
    assert table["both_fp"] == 0
    assert table["mcnemar_p"] == pytest.approx(1.0)

    analysis = analyze_false_positives(
        frame, control="baseline", unit="case", n_bootstrap=300, seed=0
    )
    assert analysis["method"].tolist() == ["baseline", "tta4"]
    control_row = analysis.set_index("method").loc["baseline"]
    assert control_row["n_cases"] == 2
    assert control_row["n_false_positive"] == 1
    assert math.isnan(control_row["mcnemar_p"])  # the control has no comparison
    treatment_row = analysis.set_index("method").loc["tta4"]
    assert treatment_row["n_false_positive"] == 0
    assert treatment_row["fp_to_correct"] == 1
    assert 0.0 <= treatment_row["mcnemar_p_adjusted"] <= 1.0
    for column in ("pred_volume_mm3_mean", "pred_volume_mm3_median", "pred_volume_mm3_max"):
        assert column in analysis.columns


def test_false_positive_analysis_without_empty_references_is_explicit() -> None:
    frame = _frame()
    frame = frame.loc[frame["case_id"].isin(["c1", "c2", "c5", "c6"])]  # no empty reference
    rate = false_positive_rate(frame, "baseline", n_bootstrap=100)
    assert rate["n_cases"] == 0
    assert math.isnan(rate["false_positive_rate"])
    assert rate["ci_low"] == rate["ci_high"] == 0.0  # nothing could be counted


def test_challenge_style_dice_counts_empty_references() -> None:
    frame = _frame()
    challenge = challenge_style_dice(frame)
    assert set(challenge["metric"]) == {CHALLENGE_METRIC}
    indexed = challenge.set_index(["method", "case_id"])
    values = indexed["value"]
    assert values[("baseline", "c3")] == 1.0  # both empty -> challenge Dice 1
    assert values[("baseline", "c4")] == 0.0  # false positive on empty -> challenge Dice 0
    assert values[("baseline", "c1")] == pytest.approx(0.8)  # unchanged
    assert values[("baseline", "c2")] == 0.0  # missed detection stays 0
    # the protocol value is kept beside the re-scored one, and the input is untouched
    assert math.isnan(indexed["strict_value"].loc[("baseline", "c3")])
    assert indexed["strict_value"].loc[("baseline", "c1")] == pytest.approx(0.8)
    assert frame.loc[frame["metric"] == "dice", "value"].isna().sum() == 4


def test_challenge_scoring_table_reports_strict_next_to_challenge() -> None:
    frame = _frame()
    table = challenge_scoring_table(frame, n_bootstrap=300, seed=0)
    baseline = table.set_index("method").loc["baseline"]

    assert baseline["n_cases"] == 6
    assert baseline["n_empty_reference"] == 2
    assert baseline["n_false_positive"] == 1
    assert baseline["n_missed"] == 1
    assert baseline["n_scored_strict"] == 4  # c1, c2, c5, c6 (the missed case scores 0)
    assert baseline["strict_dice_mean"] == pytest.approx((0.8 + 0.0 + 0.8 + 0.8) / 4)
    # challenge for the baseline: c1 .8, c2 0 (missed), c3 1 (both empty),
    # c4 0 (false positive), c5 .8, c6 .8
    assert baseline["challenge_dice_mean"] == pytest.approx(
        (0.8 + 0.0 + 1.0 + 0.0 + 0.8 + 0.8) / 6
    )
    assert baseline["delta_mean"] == pytest.approx(
        baseline["challenge_dice_mean"] - baseline["strict_dice_mean"]
    )
    assert baseline["delta_ci_low"] <= baseline["delta_mean"] <= baseline["delta_ci_high"]

    tta = table.set_index("method").loc["tta4"]
    # same empty references, but no false positive: the challenge convention helps more
    assert tta["n_false_positive"] == 0
    assert tta["delta_mean"] > baseline["delta_mean"]

    by_fold = challenge_scoring_table(frame, by_fold=True, n_bootstrap=100, seed=0)
    assert set(by_fold["fold"]) == {0}
    assert set(by_fold["method"]) == {"baseline", "tta4"}


def test_challenge_scoring_is_the_identity_without_empty_references() -> None:
    frame = _frame()
    subset = frame.loc[frame["case_id"].isin(["c1", "c2", "c5", "c6"])]
    table = challenge_scoring_table(subset, n_bootstrap=200, seed=0).set_index("method")
    for method in ("baseline", "tta4"):
        assert table.loc[method, "delta_mean"] == pytest.approx(0.0)
        assert table.loc[method, "n_empty_reference"] == 0


def test_false_positive_table_pools_folds() -> None:
    first = _frame()
    second = _frame()
    second["case_id"] = second["case_id"] + "b"
    second["group_id"] = second["case_id"]
    second["fold"] = 1
    combined = pd.concat([first, second], ignore_index=True)
    table = false_positive_table(
        combined,
        methods=["baseline", "tta4"],
        control="baseline",
        n_bootstrap=300,
        seed=0,
    )
    baseline = table.set_index("method").loc["baseline"]
    assert baseline["n_cases"] == 4  # two empty-reference cases in each fold
    assert baseline["n_false_positive"] == 2
    assert baseline["false_positive_rate"] == pytest.approx(0.5)
    assert baseline["n"] == 4  # units in the transition table (case level)
    assert baseline["pred_volume_mm3_total"] == pytest.approx(960.0)


def test_challenge_style_dice_needs_the_empty_flag() -> None:
    frame = _frame().drop(columns=["empty_flag"])
    with pytest.raises(ValueError, match="empty_flag"):
        challenge_style_dice(frame)


def test_aggregate_all_emits_the_new_tables(toy_dataset_config, tmp_path) -> None:
    """End-to-end wiring: the aggregate writes the FP and challenge tables as CSVs."""
    from melba.eval import RunFiles, aggregate_all, per_case_path

    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    runs = []
    for fold in (0, 1):
        frame = _frame()  # two empty-reference cases per fold
        frame["case_id"] = frame["case_id"] + f"f{fold}"
        frame["group_id"] = frame["case_id"]
        frame["fold"] = fold
        path = per_case_path(toy_dataset_config, fold, 0, output_dir=output_dir)
        frame.to_csv(path, index=False)
        runs.append(RunFiles(fold=fold, seed=0, per_case=path, map_dir=tmp_path / "maps"))

    result = aggregate_all(
        toy_dataset_config,
        runs=runs,
        metrics=["dice"],
        treatments=("tta4",),
        expected_seeds=(0,),
        output_dir=tmp_path / "aggregate",
        n_bootstrap=200,
    )

    for name in (
        "false_positive_by_seed",
        "false_positive_seed_averaged",
        "challenge_scoring",
        "challenge_scoring_by_seed",
        "challenge_scoring_by_fold",
    ):
        assert name in result.tables, name
        assert not result.tables[name].empty, name
        assert result.paths and name in result.paths, name

    false_positive = result.tables["false_positive_seed_averaged"].set_index("method")
    assert int(false_positive.loc["baseline", "n_cases"]) == 4  # 2 per fold
    assert int(false_positive.loc["baseline", "n_false_positive"]) == 2
    assert false_positive.loc["baseline", "pred_volume_mm3_total"] == pytest.approx(960.0)

    challenge = result.tables["challenge_scoring"].set_index("method")
    assert int(challenge.loc["baseline", "n_empty_reference"]) == 4
    assert set(result.tables["challenge_scoring_by_fold"]["fold"]) == {0, 1}



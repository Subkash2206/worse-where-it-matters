"""Paired statistics: Wilcoxon primary test, bootstrap CI, effect size, Holm."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from melba.stats import (
    analyze_long,
    analyze_pooled_folds,
    bootstrap_ci_mean,
    holm_adjust,
    paired_comparison,
    paired_values,
    pool_folds,
    rank_biserial,
)


def test_rank_biserial_known_values() -> None:
    assert rank_biserial([1, 2, 3, 4]) == pytest.approx(1.0)
    assert rank_biserial([-1, -2, -3, -4]) == pytest.approx(-1.0)
    assert rank_biserial([1, -1, 2, -2]) == pytest.approx(0.0)
    assert rank_biserial([0, 0, 0]) == 0.0
    # zeros are ignored, as in the Wilcoxon signed-rank test
    assert rank_biserial([0, 1, 2]) == pytest.approx(1.0)


def test_bootstrap_ci_is_deterministic_and_ordered() -> None:
    values = np.linspace(0.0, 1.0, 40)
    first = bootstrap_ci_mean(values, n_bootstrap=500, seed=11)
    second = bootstrap_ci_mean(values, n_bootstrap=500, seed=11)
    other = bootstrap_ci_mean(values, n_bootstrap=500, seed=12)
    assert first == second
    assert first != other
    assert first[0] <= float(np.mean(values)) <= first[1]
    constant = bootstrap_ci_mean([0.1] * 10, n_bootstrap=100, seed=0)
    assert constant == pytest.approx((0.1, 0.1))
    assert all(math.isnan(value) for value in bootstrap_ci_mean([], n_bootstrap=10))


def test_holm_adjust_known_values() -> None:
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm_adjust([0.5]), [0.5])
    # a NaN is ignored, so the remaining p-value is multiplied by m = 1
    nan_result = holm_adjust([0.01, float("nan")])
    assert nan_result[0] == pytest.approx(0.01)
    assert math.isnan(nan_result[1])


def test_paired_comparison_detects_consistent_improvement() -> None:
    rng = np.random.default_rng(0)
    control = rng.normal(0.70, 0.05, size=20)
    treatment = control + rng.normal(0.05, 0.01, size=20)
    result = paired_comparison(control, treatment, n_bootstrap=500, seed=3)
    assert result["n_pairs"] == 20
    assert result["delta_mean"] == pytest.approx(0.05, abs=0.01)
    assert result["p_value"] < 0.01
    assert result["effect_size"] == pytest.approx(1.0)
    assert result["ci_low"] > 0
    assert result["test"] == "wilcoxon"
    assert result["p_value_secondary"] < 0.01  # t-test kept as secondary evidence


def test_secondary_t_test_is_nan_when_deltas_are_constant() -> None:
    result = paired_comparison([0.5] * 6, [0.6] * 6, n_bootstrap=100)
    assert math.isnan(result["p_value_secondary"])
    assert result["p_value"] < 0.05


def test_paired_comparison_all_zero_deltas() -> None:
    result = paired_comparison([0.5] * 8, [0.5] * 8, n_bootstrap=100)
    assert result["delta_mean"] == 0.0
    assert result["p_value"] == 1.0
    assert result["effect_size"] == 0.0


def test_paired_comparison_handles_nans_and_empty_inputs() -> None:
    result = paired_comparison([0.5, np.nan, 0.6], [0.6, 0.7, 0.7], n_bootstrap=100)
    assert result["n_pairs"] == 2
    assert result["n_dropped"] == 1

    empty = paired_comparison([], [], n_bootstrap=10)
    assert empty["n_pairs"] == 0
    assert math.isnan(empty["delta_mean"])

    with pytest.raises(ValueError):
        paired_comparison([0.1, 0.2], [0.1])


def _long_frame(groups: int = 6, cases_per_group: int = 2) -> pd.DataFrame:
    rows = []
    for group in range(groups):
        for case in range(cases_per_group):
            case_id = f"g{group}_c{case}"
            group_id = f"video_{group}"
            rows.extend(
                [
                    {
                        "case_id": case_id,
                        "group_id": group_id,
                        "method": "baseline",
                        "metric": "dice",
                        "value": 0.50 + group * 0.01,
                    },
                    {
                        "case_id": case_id,
                        "group_id": group_id,
                        "method": "tta8",
                        "metric": "dice",
                        "value": 0.60 + group * 0.01,
                    },
                    {
                        "case_id": case_id,
                        "group_id": group_id,
                        "method": "baseline",
                        "metric": "hd95",
                        "value": 5.0 + group * 0.1,
                    },
                    {
                        "case_id": case_id,
                        "group_id": group_id,
                        "method": "tta8",
                        "metric": "hd95",
                        "value": 4.0 + group * 0.1,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_paired_values_group_aggregation() -> None:
    frame = _long_frame()
    grouped = paired_values(frame, "dice", treatment="tta8", control="baseline", unit="group")
    assert len(grouped) == 6
    assert grouped.columns.tolist() == ["unit_id", "baseline", "tta8"]
    assert (grouped["tta8"] - grouped["baseline"]).to_numpy() == pytest.approx([0.1] * 6)

    per_case = paired_values(frame, "dice", treatment="tta8", control="baseline", unit="case")
    assert len(per_case) == 12


def test_analyze_long_reports_direction_and_correction() -> None:
    table = analyze_long(
        _long_frame(), treatment="tta8", control="baseline", n_bootstrap=300, seed=1
    )
    dice_row = table.set_index("metric").loc["dice"]
    hd_row = table.set_index("metric").loc["hd95"]

    assert dice_row["unit"] == "group"
    assert dice_row["n_units"] == 6
    assert dice_row["n_pairs"] == 6
    assert dice_row["preferred"] == "tta8"
    assert dice_row["metric_direction"] == "higher"

    assert hd_row["delta_mean"] == pytest.approx(-1.0)
    assert hd_row["preferred"] == "tta8"
    assert hd_row["metric_direction"] == "lower"

    assert "p_value_adjusted" in table.columns
    assert "p_value_secondary" in table.columns
    assert table["p_value_adjusted"].notna().all()
    assert (table["p_value_adjusted"] >= table["p_value"] - 1e-12).all()


def test_analyze_long_case_unit_and_errors() -> None:
    table = analyze_long(
        _long_frame(), treatment="tta8", control="baseline", unit="case", n_bootstrap=100
    )
    assert table["n_units"].tolist() == [12, 12]

    with pytest.raises(ValueError, match="not found"):
        analyze_long(_long_frame(), treatment="missing", control="baseline", n_bootstrap=100)
    with pytest.raises(ValueError, match="No rows for metric"):
        paired_values(_long_frame(), "nope", treatment="tta8", control="baseline")
    with pytest.raises(ValueError, match="unit must be"):
        paired_values(_long_frame(), "dice", treatment="tta8", control="baseline", unit="patient")
    with pytest.raises(ValueError, match="Unknown correction"):
        analyze_long(_long_frame(), treatment="tta8", correction="bonferroni")


def test_analyze_long_with_no_correction() -> None:
    table = analyze_long(_long_frame(), treatment="tta8", correction="none", n_bootstrap=100)
    np.testing.assert_allclose(table["p_value_adjusted"], table["p_value"])


def _pooled_frames(videos: int = 10, cases_per_video: int = 2, n_folds: int = 5):
    rows = []
    for video in range(videos):
        fold = video % n_folds
        for case in range(cases_per_video):
            case_id = f"v{video}_f{case}"
            group_id = f"video_{video}"
            base_dice = 0.50 + 0.01 * video
            rows.extend(
                [
                    {"fold": fold, "case_id": case_id, "group_id": group_id, "method": "baseline", "metric": "dice", "value": base_dice},
                    {"fold": fold, "case_id": case_id, "group_id": group_id, "method": "tta8", "metric": "dice", "value": base_dice + 0.10},
                    {"fold": fold, "case_id": case_id, "group_id": group_id, "method": "baseline", "metric": "hd95", "value": 5.0 + 0.01 * video},
                    {"fold": fold, "case_id": case_id, "group_id": group_id, "method": "tta8", "metric": "hd95", "value": 4.0 + 0.01 * video},
                ]
            )
    frame = pd.DataFrame(rows)
    return [frame.loc[frame["fold"] == fold].copy() for fold in range(n_folds)]


def test_pool_folds_concatenates_and_validates() -> None:
    frames = _pooled_frames()
    pooled = pool_folds(frames)
    assert pooled["fold"].nunique() == 5
    assert pooled["case_id"].nunique() == 20
    assert pooled["group_id"].nunique() == 10

    # a case evaluated in two different folds must be rejected
    duplicated = [frames[0], frames[0].assign(fold=frames[0]["fold"] + 1)]
    with pytest.raises(ValueError, match="more than one fold"):
        pool_folds(duplicated)

    # a fold column is optional for single-split results
    single = _pooled_frames(n_folds=1)[0].drop(columns=["fold"])
    assert pool_folds([single])["fold"].unique().tolist() == [0]


def test_analyze_pooled_folds_uses_video_level_units() -> None:
    table = analyze_pooled_folds(
        _pooled_frames(), treatment="tta8", control="baseline", n_bootstrap=200, seed=0
    )
    dice = table.set_index("metric").loc["dice"]
    hd95 = table.set_index("metric").loc["hd95"]

    assert dice["n_folds"] == 5
    assert dice["n_groups_pooled"] == 10
    assert dice["n_cases_pooled"] == 20
    assert dice["n_units"] == 10  # one paired value per video
    assert dice["n_pairs"] == 10
    assert dice["preferred"] == "tta8"
    assert hd95["preferred"] == "tta8"
    assert dice["p_value"] < 0.01
    assert (table["p_value_adjusted"] >= table["p_value"] - 1e-12).all()


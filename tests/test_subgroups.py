"""Size-subgroup analysis and cross-model pooling for external validation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.eval.external import (  # noqa: E402
    EXTERNAL_SCOPE,
    case_averaged_frame,
    external_model_spread,
    external_tables,
    label_scope,
)
from melba.eval.subgroups import (  # noqa: E402
    SIZE_BIN_LABELS,
    add_size_bins,
    empty_reference_subgroup_table,
    subgroup_delta_table,
    subgroup_interaction_table,
    subgroup_summary_table,
    tertile_cutoffs,
)


def _long_frame(
    *,
    cases: int = 30,
    videos: int = 6,
    tta_gain_small: float = 0.05,
    tta_gain_large: float = 0.00,
    model: str = "m0",
    misses: int = 0,
) -> pd.DataFrame:
    """Synthetic CVC-like frame: by default TTA helps small cases more than large ones."""
    rows = []
    for index in range(cases):
        group = f"video_{index % videos}"
        case = f"case_{index:02d}"
        size = 500 + index * 500  # grows with the index -> small/medium/large bins
        gain = tta_gain_small if index < cases / 3 else tta_gain_large
        missed = index < misses
        for method, dice in (("baseline", 0.70), ("tta4", 0.70 + gain)):
            flag = "pred_empty" if (missed and method == "baseline") else "both_nonempty"
            base = {
                "run_id": model,
                "created_utc": "2026-01-01T00:00:00Z",
                "config_path": "configs/x.yaml",
                "config_hash": "deadbeef",
                "seed": 0,
                "dataset": "cvc_clinicdb",
                "model": model,
                "split": "test",
                "fold": 0,
                "method": method,
                "aggregator": "mean",
                "n_views": 1,
                "view_names": "identity",
                "threshold": 0.5,
                "spacing_mm": "1,1",
                "case_id": case,
                "group_id": group,
                "empty_flag": flag,
                "missed_detection": int(flag == "pred_empty"),
            }
            rows.append(
                {**base, "metric": "dice", "value": 0.0 if flag == "pred_empty" else dice}
            )
            rows.append({**base, "metric": "hd95", "value": 40.0 - 100 * gain})
    return pd.DataFrame(rows)


def _sizes(frame: pd.DataFrame) -> pd.DataFrame:
    dice = frame.loc[frame["metric"] == "dice"]
    sizes = dice[["case_id", "group_id", "fold"]].drop_duplicates("case_id").copy()
    sizes["gt_pixels"] = [500 + index * 500 for index in range(len(sizes))]
    sizes["gt_mm3"] = sizes["gt_pixels"].astype(float)
    return sizes.reset_index(drop=True)


def test_tertile_cutoffs_split_into_three_equal_groups() -> None:
    values = list(range(1, 31))
    low, high = tertile_cutoffs(values)
    assert low == pytest.approx(np.percentile(values, 100 / 3))
    assert high == pytest.approx(np.percentile(values, 200 / 3))
    with pytest.raises(ValueError, match="three values"):
        tertile_cutoffs([1.0, 2.0])


def test_bins_are_assigned_by_the_reported_cutoffs() -> None:
    frame = _long_frame()
    enriched, cutoffs = add_size_bins(frame, _sizes(frame))
    assert cutoffs == sorted(cutoffs) and len(cutoffs) == 2
    per_bin = (
        enriched.loc[enriched["metric"] == "dice"]
        .drop_duplicates("case_id")["size_bin"]
        .value_counts()
    )
    assert set(per_bin.index) == set(SIZE_BIN_LABELS)
    assert per_bin.max() - per_bin.min() <= 1  # equal-size tertiles
    ordered = (
        enriched.loc[enriched["metric"] == "dice"]
        .drop_duplicates("case_id")
        .sort_values("gt_pixels")
    )
    assert ordered["size_bin"].iloc[0] == SIZE_BIN_LABELS[0]
    assert ordered["size_bin"].iloc[-1] == SIZE_BIN_LABELS[-1]


def test_summary_table_reports_ci_and_misses() -> None:
    frame = _long_frame(misses=3)
    enriched, _ = add_size_bins(frame, _sizes(frame))
    table = subgroup_summary_table(enriched, metrics=("dice",), n_bootstrap=200, seed=0)
    assert set(table["size_bin"]) == set(SIZE_BIN_LABELS)
    assert set(table["method"]) == {"baseline", "tta4"}
    row = table.loc[(table["method"] == "baseline") & (table["size_bin"] == "small")].iloc[0]
    assert row["n_cases"] == 10
    assert row["ci_low"] <= row["dice_mean"] <= row["ci_high"]
    assert row["n_missed"] == 3
    assert row["missed_rate"] == pytest.approx(0.3)


def test_delta_table_is_paired_inside_each_bin() -> None:
    frame = _long_frame(tta_gain_small=0.05, tta_gain_large=0.0)
    enriched, _ = add_size_bins(frame, _sizes(frame))
    table = subgroup_delta_table(
        enriched, metrics=("dice",), treatments=("tta4",), n_bootstrap=200, seed=0
    )
    assert set(table["size_bin"]) == set(SIZE_BIN_LABELS)
    by_bin = table.set_index("size_bin")["delta_mean"]
    assert by_bin["small"] == pytest.approx(0.05, abs=1e-9)
    assert by_bin["large"] == pytest.approx(0.0, abs=1e-9)


def test_interaction_detects_a_size_dependent_gain() -> None:
    frame = _long_frame(tta_gain_small=0.05, tta_gain_large=0.0)
    enriched, _ = add_size_bins(frame, _sizes(frame))
    table = subgroup_interaction_table(
        enriched, metrics=("dice",), treatments=("tta4",), n_bootstrap=300, seed=0
    )
    row = table.iloc[0]
    assert row["delta_small"] > row["delta_large"]
    assert row["delta_gap"] == pytest.approx(row["delta_large"] - row["delta_small"])
    assert row["gap_ci_low"] <= row["delta_gap"] <= row["gap_ci_high"]
    assert 0.0 <= row["kruskal_p"] <= 1.0


def test_interaction_is_null_when_the_gain_is_uniform() -> None:
    frame = _long_frame(tta_gain_small=0.03, tta_gain_large=0.03)
    enriched, _ = add_size_bins(frame, _sizes(frame))
    table = subgroup_interaction_table(
        enriched, metrics=("dice",), treatments=("tta4",), n_bootstrap=300, seed=0
    )
    row = table.iloc[0]
    assert abs(row["delta_gap"]) < 1e-9
    assert row["gap_ci_low"] <= 0.0 <= row["gap_ci_high"]


def test_empty_reference_subgroup_is_empty_for_polyp_data() -> None:
    frame = _long_frame()
    enriched, _ = add_size_bins(frame, _sizes(frame))
    table = empty_reference_subgroup_table(enriched, n_bootstrap=100)
    assert set(table["method"]) == {"baseline", "tta4"}
    assert table["n_cases"].eq(0).all()


def test_external_pooling_averages_cases_not_models() -> None:
    frames = {f"model{index}": _long_frame(model=f"model{index}") for index in range(3)}
    averaged = case_averaged_frame(frames)
    assert averaged["scope"].eq(EXTERNAL_SCOPE).all()
    assert averaged["n_models"].eq(3).all()
    # one row per (case, method, metric): the models did not multiply the sample
    assert not averaged.duplicated(["case_id", "method", "metric"]).any()
    assert averaged["case_id"].nunique() == 30
    row = averaged.loc[
        (averaged["case_id"] == "case_00") & (averaged["method"] == "tta4")
        & (averaged["metric"] == "dice")
    ].iloc[0]
    assert row["value"] == pytest.approx(0.75)


def test_model_spread_reports_the_checkpoint_range() -> None:
    frames = {
        "a": _long_frame(model="a", tta_gain_small=0.02),
        "b": _long_frame(model="b", tta_gain_small=0.06),
    }
    spread = external_model_spread(frames, metrics=["dice"])
    row = spread.loc[(spread["method"] == "tta4") & (spread["metric"] == "dice")].iloc[0]
    assert row["n_models"] == 2
    assert row["min"] < row["max"]
    assert row["range"] == pytest.approx(row["max"] - row["min"])
    assert row["scope"] == EXTERNAL_SCOPE


def test_external_tables_are_labelled_and_scoped() -> None:
    frames = {f"m{index}": _long_frame(model=f"m{index}") for index in range(2)}
    tables = external_tables(
        frames, metrics=("dice", "hd95"), treatments=("tta4",), n_bootstrap=100, seed=0
    )
    assert {"external_model_spread", "external_paired", "external_detection"} <= set(tables)
    for name, table in tables.items():
        assert not table.empty, name
        assert table["scope"].eq(EXTERNAL_SCOPE).all(), name
    detection = tables["external_detection"].set_index("method")
    assert detection.loc["baseline", "n_cases"] == 30
    # no empty reference masks in the polyp data -> nothing to report there
    assert tables["external_false_positive"]["n_cases"].eq(0).all()


def test_label_scope_marks_internal_frames() -> None:
    labelled = label_scope(_long_frame(), "internal")
    assert labelled["scope"].eq("internal").all()



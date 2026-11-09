"""Pooled cross-validation tables and the seed ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melba.eval import (
    RunFiles,
    aggregate_all,
    aggregate_output_dir,
    compare_methods,
    detection_table,
    discover_runs,
    ensemble_frame,
    map_dir_for,
    per_case_path,
    pooled_seed_table,
    seed_averaged_frame,
    string_ids,
)
from melba.metrics import MetricSettings


def _long_frame(
    fold: int,
    seed: int,
    *,
    offset: int,
    dice: dict[str, float],
    hd95: float = 5.0,
    missed: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Two frames for each of three videos; every fold owns distinct videos."""
    rows = []
    for video in range(3):
        group = f"video_{offset + video}"
        for frame in range(2):
            case_id = f"{group}_f{frame}"
            for method, value in dice.items():
                flag = "pred_empty" if (missed or {}).get(method) else "both_nonempty"
                for metric, metric_value in (("dice", value), ("hd95", hd95)):
                    rows.append(
                        {
                            "run_id": f"run-fold{fold}-seed{seed}",
                            "created_utc": "2026-01-01T00:00:00Z",
                            "config_path": "",
                            "config_hash": "deadbeef",
                            "seed": seed,
                            "dataset": "cvc_clinicdb",
                            "model": "smp_unet",
                            "split": "test",
                            "fold": fold,
                            "method": method,
                            "aggregator": "mean",
                            "n_views": 1,
                            "view_names": "identity",
                            "threshold": 0.5,
                            "spacing_mm": "1,1",
                            "case_id": case_id,
                            "group_id": group,
                            "empty_flag": flag,
                            "missed_detection": int(flag == "pred_empty"),
                            "metric": metric,
                            "value": metric_value,
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def frames_by_seed() -> dict[int, list[pd.DataFrame]]:
    values = {
        0: {"baseline": 0.70, "tta4": 0.74},
        1: {"baseline": 0.72, "tta4": 0.76},
    }
    grouped: dict[int, list[pd.DataFrame]] = {}
    for seed, dice in values.items():
        grouped[seed] = [
            _long_frame(0, seed, offset=0, dice=dice),
            _long_frame(1, seed, offset=3, dice=dice),
        ]
    return grouped


def test_pooled_seed_table_reports_mean_and_std_across_seeds(frames_by_seed) -> None:
    table = pooled_seed_table(frames_by_seed, metrics=["dice", "hd95"])
    baseline = table.loc[(table["method"] == "baseline") & (table["metric"] == "dice")].iloc[0]

    assert baseline["n_seeds"] == 2
    assert baseline["n_units"] == 6  # 6 videos, each a test unit exactly once
    assert baseline["n_cases"] == 12
    assert baseline["mean"] == pytest.approx(0.71)
    assert baseline["std"] == pytest.approx(float(np.std([0.70, 0.72], ddof=1)))
    assert baseline["mean_seed0"] == pytest.approx(0.70)
    assert baseline["mean_seed1"] == pytest.approx(0.72)
    assert set(table["metric"]) == {"dice", "hd95"}


def test_seed_averaged_frame_averages_values_and_majority_flags(frames_by_seed) -> None:
    averaged = seed_averaged_frame(frames_by_seed)
    row = averaged.loc[
        (averaged["method"] == "tta4") & (averaged["metric"] == "dice") & (averaged["fold"] == 0)
    ].iloc[0]
    assert row["value"] == pytest.approx(0.75)  # mean of 0.74 and 0.76
    assert row["n_seeds"] == 2
    assert set(averaged["seed"]) == {-1}
    # one row per case/method/metric: no duplicated test units after averaging
    assert not averaged.duplicated(["case_id", "method", "metric"]).any()


def test_seed_averaged_frame_marks_majority_misses() -> None:
    frames = {
        0: [_long_frame(0, 0, offset=0, dice={"baseline": 0.8}, missed={"baseline": True})],
        1: [_long_frame(0, 1, offset=0, dice={"baseline": 0.8})],
        2: [_long_frame(0, 2, offset=0, dice={"baseline": 0.8})],
    }
    averaged = seed_averaged_frame(frames)
    assert set(averaged["missed_detection"]) == {0}
    assert set(averaged["empty_flag"]) == {"both_nonempty"}  # 2 of 3 seeds detected it


def test_pooled_paired_tests_and_detection(frames_by_seed) -> None:
    averaged = seed_averaged_frame(frames_by_seed)
    table = compare_methods(averaged, [("tta4", "baseline")], metrics=["dice"])
    assert set(table["treatment"]) == {"tta4"}
    assert table["n_units"].iloc[0] == 6
    assert table["delta_mean"].iloc[0] == pytest.approx(0.04)

    detection = detection_table(averaged, methods=["baseline", "tta4"])
    assert {"method", "n_cases", "n_missed", "mcnemar_p"} <= set(detection.columns)
    assert detection.loc[detection["method"] == "baseline", "n_cases"].iloc[0] == 12


def test_ensemble_frame_averages_probability_maps(toy_dataset_config, tmp_path) -> None:
    gt_mask = np.zeros((8, 8), dtype=np.float32)
    gt_mask[2:4, 2:4] = 1.0
    ground_truth = {
        0: {
            "case_pos": {"mask": gt_mask, "spacing": (1.0, 1.0), "group_id": "video_a"},
            "case_neg": {"mask": gt_mask, "spacing": (1.0, 1.0), "group_id": "video_b"},
        }
    }
    runs = []
    for seed, positive in ((0, 0.9), (1, 0.3), (2, 0.6)):
        map_dir = tmp_path / "maps" / f"fold0_seed{seed}"
        map_dir.mkdir(parents=True)
        np.save(map_dir / "case_pos__baseline__mean.npy", np.full((8, 8), positive, np.float16))
        np.save(map_dir / "case_neg__baseline__mean.npy", np.full((8, 8), 0.1, np.float16))
        runs.append(RunFiles(fold=0, seed=seed, per_case=tmp_path / "x.csv", map_dir=map_dir))

    settings = MetricSettings(names=("dice", "hd95", "assd"), empty_policy="nan")
    frame = ensemble_frame(
        toy_dataset_config,
        runs,
        name="ensemble3",
        source_method="baseline",
        ground_truth=ground_truth,
        settings=settings,
    )

    positive_rows = frame.loc[frame["case_id"] == "case_pos"]
    assert set(positive_rows["method"]) == {"ensemble3"}
    assert positive_rows["n_views"].iloc[0] == 3
    # mean probability (0.9 + 0.3 + 0.6) / 3 = 0.6 > 0.5: the whole 8x8 grid is positive
    dice = positive_rows.loc[positive_rows["metric"] == "dice", "value"].iloc[0]
    assert dice == pytest.approx(2 * 4 / (64 + 4))
    assert set(positive_rows["empty_flag"]) == {"both_nonempty"}

    negative_rows = frame.loc[frame["case_id"] == "case_neg"]
    assert set(negative_rows["empty_flag"]) == {"pred_empty"}
    assert int(negative_rows["missed_detection"].iloc[0]) == 1
    assert negative_rows.loc[negative_rows["metric"] == "dice", "value"].iloc[0] == 0.0
    assert np.isnan(negative_rows.loc[negative_rows["metric"] == "hd95", "value"].iloc[0])
    penalized = negative_rows.loc[negative_rows["metric"] == "hd95_penalized", "value"].iloc[0]
    assert not np.isnan(penalized)


def test_ensemble_frame_reports_missing_maps(toy_dataset_config, tmp_path) -> None:
    runs = [
        RunFiles(
            fold=0,
            seed=seed,
            per_case=tmp_path / "x.csv",
            map_dir=tmp_path / f"fold0_seed{seed}",
        )
        for seed in (0, 1)
    ]
    for run in runs:
        run.map_dir.mkdir(parents=True)
    ground_truth = {
        0: {"case_a": {"mask": np.zeros((4, 4)), "spacing": (1.0, 1.0), "group_id": "v"}}
    }
    with pytest.raises(FileNotFoundError, match="probability maps"):
        ensemble_frame(
            toy_dataset_config,
            runs,
            name="ensemble3",
            source_method="baseline",
            ground_truth=ground_truth,
            settings=MetricSettings(names=("dice",)),
        )


def test_paths_and_discovery(toy_dataset_config, tmp_path) -> None:
    config = toy_dataset_config
    path = per_case_path(config, 2, 1, output_dir=tmp_path)
    assert path.name == "cvc_clinicdb_test_fold2_seed1_per_case.csv"
    assert map_dir_for(config, 2, 1, output_dir=tmp_path).name == "fold2_seed1"

    path.write_text("case_id\n", encoding="utf-8")
    runs = discover_runs(config, folds=[2], seeds=[0, 1], output_dir=tmp_path)
    assert [(run.fold, run.seed) for run in runs] == [(2, 1)]



def test_string_ids_normalises_numeric_ids() -> None:
    frame = pd.DataFrame({"case_id": [1, 2], "group_id": [10, 20], "value": [1.0, 2.0]})
    normalised = string_ids(frame)
    assert normalised["case_id"].tolist() == ["1", "2"]
    assert normalised["group_id"].tolist() == ["10", "20"]
    assert frame["case_id"].tolist() == [1, 2]  # the input frame is untouched


def test_aggregate_all_merges_ensemble_with_numeric_csv_ids(toy_dataset_config, tmp_path) -> None:
    """The CSV parses case ids as ints, the maps key them as strings: they must merge."""
    frame = _long_frame(0, 0, offset=0, dice={"baseline": 0.6, "tta4": 0.7})
    original = frame[["case_id"]].drop_duplicates()["case_id"].tolist()
    # give each case a slightly different value so the paired t-test is well conditioned
    offsets = {case: 0.01 * index for index, case in enumerate(original)}
    frame["value"] = frame["value"] + frame["case_id"].map(offsets)
    mapping = {case: str(index + 1) for index, case in enumerate(original)}
    frame["case_id"] = frame["case_id"].map(mapping).astype(int)

    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    csv_path = per_case_path(toy_dataset_config, 0, 0, output_dir=output_dir)
    frame.to_csv(csv_path, index=False)

    map_dir = tmp_path / "maps" / "fold0_seed0"
    map_dir.mkdir(parents=True)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:4, 2:4] = 1.0
    ground_truth: dict[int, dict[str, dict[str, object]]] = {0: {}}
    for _, row in frame.drop_duplicates("case_id").iterrows():
        case = str(row["case_id"])
        group = str(row["group_id"])
        ground_truth[0][case] = {"mask": mask, "spacing": (1.0, 1.0), "group_id": group}
        for method in ("baseline", "tta4"):
            np.save(map_dir / f"{case}__{method}__mean.npy", np.full((8, 8), 0.9, np.float16))

    result = aggregate_all(
        toy_dataset_config,
        runs=[RunFiles(fold=0, seed=0, per_case=csv_path, map_dir=map_dir)],
        metrics=["dice"],
        expected_seeds=(0,),
        ensemble_min_seeds=1,  # this fixture has a single seed on purpose
        ground_truth=ground_truth,
        output_dir=tmp_path / "aggregate",
        n_bootstrap=200,
    )

    comparison = result.tables["ensemble_comparison"]
    assert {"ensemble3", "ensemble3_tta4"} <= set(comparison["treatment"])
    ensemble_vs_baseline = comparison.loc[
        (comparison["treatment"] == "ensemble3") & (comparison["control"] == "baseline")
    ]
    assert len(ensemble_vs_baseline) == 1
    assert int(ensemble_vs_baseline["n_units"].iloc[0]) == 3  # 3 videos, not 0
    assert int(ensemble_vs_baseline["n_cases_used"].iloc[0]) == 6

    detection = result.tables["ensemble_detection"]
    assert {"baseline", "tta4", "ensemble3", "ensemble3_tta4"} <= set(detection["method"])
    assert result.paths and "ensemble_comparison" in result.paths
    assert (tmp_path / "aggregate" / "manifest.json").is_file()



def test_default_aggregate_dir_is_per_experiment(base_config) -> None:
    """The aggregate step must not overwrite another experiment's tables."""
    unet = aggregate_output_dir(base_config)
    swin = aggregate_output_dir(
        base_config.with_overrides({"experiment.name": "cvc_clinicdb_swinunetr_2d"})
    )
    assert unet.name == "aggregate_cvc_clinicdb_unet_resnet50"
    assert swin.name == "aggregate_cvc_clinicdb_swinunetr_2d"
    assert unet != swin
    assert unet.parent.name == "results"


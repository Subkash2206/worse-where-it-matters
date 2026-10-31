"""End-to-end evaluation: long-format output, provenance and scientific guards."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from melba.config import ConfigError
from melba.data import SplitError, build_dataset, read_split_csv
from melba.eval import evaluate, read_long_csv
from melba.eval.runner import resample_probability_map


class TinyModel(nn.Module):
    """Deterministic, parameter-free model producing a smooth logit field."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = x.mean(dim=1, keepdim=True)
        return (signal - 0.5) * 5.0


class ConvModel(nn.Module):
    """Small parameterised model (weights depend on the global torch seed)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.mean(dim=1, keepdim=True) - 0.5)


class ZeroModel(nn.Module):
    """Predicts nothing (all probabilities ~0) to exercise the empty-mask policy."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[:, :1], -20.0)


class CountingModel(nn.Module):
    """Counts forward passes (used to verify warm-up runs)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return (x.mean(dim=1, keepdim=True) - 0.5) * 5.0


def _conv_model() -> ConvModel:
    torch.manual_seed(0)
    return ConvModel()


@pytest.fixture
def eval_config(toy_dataset_config):
    return toy_dataset_config.with_overrides(
        {"metrics.names": ["dice", "hd95", "assd", "boundary_f1"]}
    )


def test_dataset_items(eval_config) -> None:
    dataset = build_dataset(eval_config, "test")
    assert len(dataset) >= 1
    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["image"].dtype == torch.float32
    assert float(sample["image"].min()) >= 0.0 and float(sample["image"].max()) <= 1.0
    assert sample["mask"].shape == (1, 64, 64)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    # evaluation keeps the native mask so metrics are computed on the native grid
    assert sample["mask_native"].shape == (1, 40, 56)
    assert set(torch.unique(sample["mask_native"]).tolist()) <= {0.0, 1.0}
    assert sample["spacing"] == (1.0, 1.0)
    assert sample["split"] == "test"
    assert sample["fold"] == 0
    assert sample["group_id"].startswith("video_")


def test_test_split_is_a_whole_video_group(eval_config) -> None:
    dataset = build_dataset(eval_config, "test")
    groups = {sample["group_id"] for sample in (dataset[index] for index in range(len(dataset)))}
    cases = [dataset[index]["case_id"] for index in range(len(dataset))]
    assert len(cases) == 2 * len(groups)  # two frames per toy video, no mixing
    assert len(groups) == 2  # fold 0 of the 6-video toy set holds two videos


def test_training_items_have_no_native_mask(eval_config) -> None:
    dataset = build_dataset(eval_config, "train", train=True)
    sample = dataset[0]
    assert "mask_native" not in sample
    assert sample["mask"].shape == (1, *sample["image"].shape[1:])


def test_dataset_requires_split_file(eval_config, tmp_path) -> None:
    config = eval_config.with_overrides(
        {"dataset.split.output_pattern": str(tmp_path / "missing_fold{fold}.csv")}
    )
    with pytest.raises(SplitError, match="Split file not found"):
        build_dataset(config, "test")


def test_evaluation_writes_long_format(eval_config, tmp_path) -> None:
    result = evaluate(
        eval_config, model=TinyModel(), limit=2, output_dir=tmp_path / "out", progress=False
    )
    frame = read_long_csv(result.per_case_csv)

    gt_metrics = {"dice", "hd95", "assd", "boundary_f1"}
    timing_metrics = {"latency_ms", "peak_memory_mb"}
    assert set(frame["metric"]) == gt_metrics | timing_metrics
    assert frame["method"].nunique() == 4
    assert frame["case_id"].nunique() == 2
    assert len(frame) == 2 * 4 * len(gt_metrics | timing_metrics)
    assert set(frame["split"]) == {"test"}
    assert set(frame["fold"]) == {0}
    assert (frame["spacing_mm"] == "1,1").all()
    assert (frame["threshold"] == 0.5).all()
    assert (frame["seed"] == eval_config["seed"]).all()
    assert (frame["config_hash"] == eval_config.fingerprint).all()
    assert set(frame["empty_flag"]) <= {
        "both_nonempty",
        "both_empty",
        "pred_empty",
        "gt_empty",
    }
    # one row per case/method/metric combination, never an aggregate row
    assert frame.groupby(["case_id", "method", "metric"]).size().max() == 1
    assert (frame.loc[frame["metric"] == "latency_ms", "value"] > 0).all()

    assert result.summary_csv is not None and result.summary_csv.is_file()
    summary = pd.read_csv(result.summary_csv)
    assert {"method", "metric", "n", "mean", "n_cases"} <= set(summary.columns)
    assert (summary["n_cases"] == 2).all()

    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["split"] == "test"
    assert metadata["fold"] == 0
    assert metadata["selection_split"] == "val"
    assert metadata["cases_evaluated"] == 2
    assert metadata["checkpoint"]["path"] is None
    assert metadata["config_hash"] == eval_config.fingerprint
    assert "environment" in metadata and "git_revision" in metadata
    assert metadata["metrics"]["empty_policy"] == "nan"
    assert metadata["timing"]["warmup_runs"] == 3
    assert [method["name"] for method in metadata["methods"]] == [
        "baseline",
        "tta2",
        "tta4",
        "tta8",
    ]


def test_evaluation_refuses_selection_split(eval_config, tmp_path) -> None:
    with pytest.raises(ConfigError, match="selection split"):
        evaluate(
            eval_config,
            split="val",
            model=TinyModel(),
            limit=1,
            output_dir=tmp_path / "val",
            progress=False,
        )


def test_evaluation_refuses_train_split(eval_config, tmp_path) -> None:
    with pytest.raises(ConfigError, match="training split"):
        evaluate(
            eval_config,
            split="train",
            model=TinyModel(),
            limit=1,
            output_dir=tmp_path / "train",
            progress=False,
        )


def test_flagged_selection_split_is_allowed(eval_config, tmp_path) -> None:
    result = evaluate(
        eval_config,
        split="val",
        model=TinyModel(),
        limit=1,
        output_dir=tmp_path / "val",
        allow_selection_split=True,
        progress=False,
    )
    assert result.split == "val"
    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["split"] == metadata["selection_split"]


def test_evaluation_requires_a_checkpoint_when_building_its_own_model(
    eval_config, tmp_path
) -> None:
    with pytest.raises(ConfigError, match="model.checkpoint"):
        evaluate(eval_config, limit=1, output_dir=tmp_path / "ckpt", progress=False)


def test_evaluation_is_deterministic(eval_config, tmp_path) -> None:
    first = evaluate(
        eval_config, model=_conv_model(), limit=2, output_dir=tmp_path / "a", progress=False
    )
    second = evaluate(
        eval_config, model=_conv_model(), limit=2, output_dir=tmp_path / "b", progress=False
    )
    left = read_long_csv(first.per_case_csv).sort_values(["case_id", "method", "metric"])
    right = read_long_csv(second.per_case_csv).sort_values(["case_id", "method", "metric"])
    assert left["case_id"].tolist() == right["case_id"].tolist()

    # timings are hardware-dependent; every other value must be bit-identical
    timing = {"latency_ms", "peak_memory_mb"}
    left_stable = left.loc[~left["metric"].isin(timing)].reset_index(drop=True)
    right_stable = right.loc[~right["metric"].isin(timing)].reset_index(drop=True)
    pd.testing.assert_series_equal(left_stable["value"], right_stable["value"])
    # TTA must actually change the prediction relative to the baseline
    pivot = left_stable.pivot_table(index=["case_id", "metric"], columns="method", values="value")
    assert not np.allclose(pivot["baseline"], pivot["tta8"], atol=1e-9)


def test_dataset_detects_leakage_in_split_file(eval_config, tmp_path) -> None:
    from melba.data import split_file_for_fold

    leaky = tmp_path / "leaky_fold0.csv"
    records = read_split_csv(split_file_for_fold(eval_config, 0))
    case_ids = [record.case_id for record in records]
    lines = ["case_id,group_id,split"]
    # put two cases of the same video into different splits
    lines.append(f"{case_ids[0]},shared_video,train")
    lines.append(f"{case_ids[1]},shared_video,test")
    leaky.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = eval_config.with_overrides({"dataset.split.output_pattern": str(leaky)})
    with pytest.raises(SplitError, match="more than one split"):
        build_dataset(config, "test")


def test_metrics_are_computed_on_the_native_grid(eval_config, tmp_path) -> None:
    """The toy frames are 40x56 but the network input is 64x64."""
    config = eval_config.with_overrides({"metrics.empty_policy": "penalize"})
    result = evaluate(
        config, model=ZeroModel(), limit=1, output_dir=tmp_path / "native", progress=False
    )
    frame = read_long_csv(result.per_case_csv)
    hd95 = frame.loc[frame["metric"] == "hd95", "value"].to_numpy()
    # empty prediction under `empty_policy: penalize` -> native grid diagonal
    assert np.allclose(hd95, float(np.hypot(39, 55)), rtol=1e-6)
    assert set(frame.loc[frame["metric"] == "dice", "value"]) == {0.0}
    assert set(frame.loc[frame["metric"] == "hd95", "empty_flag"]) == {"pred_empty"}
    assert set(frame["spacing_mm"]) == {"1,1"}


def test_uncertainty_rows_warmup_and_timing(eval_config, tmp_path) -> None:
    uncertainty = [
        "entropy_mean",
        "variance_mean",
        "entropy_mean_roi",
        "variance_mean_roi",
        "view_disagreement",
        "per_view_dice_mean",
        "per_view_dice_std",
    ]
    config = eval_config.with_overrides(
        {"metrics.names": ["dice", "hd95"] + uncertainty, "evaluation.warmup_runs": 3}
    )
    model = CountingModel()
    result = evaluate(
        config, model=model, limit=1, output_dir=tmp_path / "uncertainty", progress=False
    )
    frame = read_long_csv(result.per_case_csv)

    assert set(uncertainty) <= set(frame["metric"])
    assert {"latency_ms", "peak_memory_mb"} <= set(frame["metric"])
    # 4 methods (1+2+4+8 views) x (3 warm-ups + 1 timed pass)
    assert model.calls == (1 + 2 + 4 + 8) * 4

    baseline_dice = frame.loc[
        (frame["method"] == "baseline") & (frame["metric"] == "per_view_dice_mean"), "value"
    ]
    np.testing.assert_allclose(baseline_dice, 1.0)
    multi_view_variance = frame.loc[
        frame["method"].isin(["tta2", "tta4", "tta8"]) & (frame["metric"] == "variance_mean"),
        "value",
    ]
    assert (multi_view_variance >= 0).all()
    # no GPU -> peak memory is undefined and written as an empty field
    assert frame.loc[frame["metric"] == "peak_memory_mb", "value"].isna().all()


def test_probability_maps_can_be_saved_as_float16(eval_config, tmp_path) -> None:
    map_dir = tmp_path / "maps"
    config = eval_config.with_overrides(
        {
            "evaluation.save_probability_maps": True,
            "evaluation.probability_map_dir": str(map_dir),
        }
    )
    evaluate(config, model=TinyModel(), limit=1, output_dir=tmp_path / "maps_run", progress=False)

    files = sorted(map_dir.glob("*.npy"))
    assert len(files) == 4 * 2  # mean and variance per method
    saved = np.load(files[0])
    assert saved.dtype == np.float16
    assert saved.shape == (40, 56)  # native grid, not the 64x64 network grid


def test_resample_probability_map() -> None:
    probability = np.zeros((1, 4, 5), dtype=np.float32)
    probability[0, 1:3, 1:4] = 0.7
    np.testing.assert_allclose(resample_probability_map(probability, (4, 5)), probability)

    upsampled = resample_probability_map(probability, (8, 10))
    assert upsampled.shape == (1, 8, 10)
    assert 0.0 <= float(upsampled.min()) and float(upsampled.max()) <= 1.0

    volume = np.random.default_rng(0).random((2, 6, 8, 10), dtype=np.float32)
    downsampled = resample_probability_map(volume, (3, 4, 5))
    assert downsampled.shape == (2, 3, 4, 5)

    with pytest.raises(ValueError, match="probability map"):
        resample_probability_map(np.zeros((1, 5)), (3,))



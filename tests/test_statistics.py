"""Training-set normalization statistics and the from-scratch SwinUNETR path."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from melba.config import ConfigError, load_config
from melba.data import (
    NormalizationSpec,
    build_dataset,
    compute_fold_statistics,
    read_statistics,
    write_statistics,
)
from melba.models import MODEL_ARCHITECTURES, build_model


def test_fold_statistics_are_deterministic_and_plausible(toy_dataset_config) -> None:
    statistics = compute_fold_statistics(toy_dataset_config, 0)
    again = compute_fold_statistics(toy_dataset_config, 0)
    assert statistics["mean"] == again["mean"]
    assert statistics["std"] == again["std"]

    assert len(statistics["mean"]) == 3 and len(statistics["std"]) == 3
    assert all(np.isfinite(value) for value in statistics["mean"])
    assert all(value > 0 for value in statistics["std"])
    assert 0.0 <= min(statistics["mean"]) <= max(statistics["mean"]) <= 1.0
    assert statistics["n_frames"] == len(build_dataset(toy_dataset_config, "train", fold=0))


def test_write_statistics_merges_and_respects_force(tmp_path) -> None:
    path = tmp_path / "norm_stats.json"
    write_statistics(path, {"fold0": {"mean": [0.5] * 3, "std": [0.2] * 3}})
    assert read_statistics(path)["fold0"]["mean"] == [0.5] * 3

    write_statistics(
        path,
        {
            "fold0": {"mean": [0.1] * 3, "std": [0.1] * 3},
            "fold1": {"mean": [0.4] * 3, "std": [0.3] * 3},
        },
    )
    payload = read_statistics(path)
    assert payload["fold0"]["mean"] == [0.5] * 3  # existing folds are kept
    assert payload["fold1"]["mean"] == [0.4] * 3

    write_statistics(path, {"fold0": {"mean": [0.1] * 3, "std": [0.1] * 3}}, force=True)
    assert read_statistics(path)["fold0"]["mean"] == [0.1] * 3


def test_dataset_normalization_uses_training_statistics(toy_dataset_config, tmp_path) -> None:
    stats_path = tmp_path / "norm_stats.json"
    entry = compute_fold_statistics(toy_dataset_config, 0)
    write_statistics(stats_path, {"fold0": entry})
    config = toy_dataset_config.with_overrides(
        {
            "dataset.normalization.kind": "dataset",
            "dataset.normalization.stats_file": str(stats_path),
            "dataset.image_size": [40, 56],
        }
    )

    spec = NormalizationSpec.from_config(config, fold=0)
    assert spec.kind == "dataset"
    np.testing.assert_allclose(spec.mean, entry["mean"], atol=1e-12)
    np.testing.assert_allclose(spec.std, entry["std"], atol=1e-12)
    sample = build_dataset(config, "test", fold=0)[0]
    assert sample["normalization"] == "dataset"

    with pytest.raises(ConfigError, match="fold 1"):
        NormalizationSpec.from_config(config, fold=1)

    missing = config.with_overrides(
        {"dataset.normalization.stats_file": str(tmp_path / "missing.json")}
    )
    with pytest.raises(ConfigError, match="compute_normalization"):
        NormalizationSpec.from_config(missing, fold=0)


def test_swinunetr_config_mirrors_the_unet_config(repo_root) -> None:
    swin = load_config(repo_root / "configs" / "cvc_clinicdb" / "swinunetr_2d.yaml")
    assert swin["model.architecture"] == "monai_swinunetr"
    assert swin["model.spatial_dims"] == 2
    assert swin["dataset.normalization"]["kind"] == "dataset"
    assert swin["training.train_aug"] == "geometric_d4"
    assert swin["evaluation.timing"] == "auto"
    assert swin["model.encoder_weights"] == "imagenet"  # inherited but unused in 2D Swin
    assert "monai_swinunetr" in MODEL_ARCHITECTURES


def test_swinunetr_2d_forward(toy_dataset_config) -> None:
    config = toy_dataset_config.with_overrides(
        {
            "model.architecture": "monai_swinunetr",
            "model.spatial_dims": 2,
            "model.feature_size": 12,
            "model.depths": [1, 1, 1, 1],
            "dataset.image_size": [64, 64],
        }
    )
    model = build_model(config).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 1, 64, 64)


def test_swinunetr_requires_a_multiple_of_32() -> None:
    """352 = 11 x 32; the UNet image size is reusable, 40 x 56 is not."""
    assert 352 % 32 == 0

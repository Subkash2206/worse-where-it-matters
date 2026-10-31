"""Input normalization: smp preprocessing statistics and dataset integration."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from melba.config import ConfigError
from melba.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NormalizationSpec,
    build_dataset,
)


def test_imagenet_statistics_match_smp(base_config) -> None:
    import segmentation_models_pytorch as smp

    params = smp.encoders.get_preprocessing_params("resnet50", pretrained="imagenet")
    spec = NormalizationSpec.from_config(base_config)

    assert spec.kind == "imagenet"
    assert spec.enabled
    np.testing.assert_allclose(spec.mean, params["mean"], atol=1e-12)
    np.testing.assert_allclose(spec.std, params["std"], atol=1e-12)
    np.testing.assert_allclose(spec.mean, IMAGENET_MEAN, atol=1e-12)
    np.testing.assert_allclose(spec.std, IMAGENET_STD, atol=1e-12)


def test_apply_known_values() -> None:
    spec = NormalizationSpec(kind="custom", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    image = np.zeros((3, 2, 2), dtype=np.float32)
    image[1] = 1.0
    normalized = spec.apply(image)
    np.testing.assert_allclose(normalized[0], -1.0)
    np.testing.assert_allclose(normalized[1], 1.0)
    np.testing.assert_allclose(normalized[2], -1.0)
    assert normalized.dtype == np.float32

    identity = NormalizationSpec(kind="none")
    np.testing.assert_allclose(identity.apply(image), image)
    assert not identity.enabled


def test_config_overrides_and_none(base_config) -> None:
    custom = NormalizationSpec.from_config(
        base_config.with_overrides(
            {"dataset.normalization.kind": "custom", "dataset.normalization.mean": [0.1] * 3,
             "dataset.normalization.std": [0.2] * 3}
        )
    )
    assert custom.kind == "custom"
    assert custom.mean == (0.1, 0.1, 0.1)

    overridden = NormalizationSpec.from_config(
        base_config.with_overrides({"dataset.normalization.mean": [0.0, 0.0, 0.0]})
    )
    assert overridden.kind == "imagenet"
    assert overridden.mean == (0.0, 0.0, 0.0)

    disabled = NormalizationSpec.from_config(
        base_config.with_overrides({"dataset.normalization.kind": "none"})
    )
    assert not disabled.enabled

    with pytest.raises(ConfigError, match="normalization.kind"):
        NormalizationSpec.from_config(
            base_config.with_overrides({"dataset.normalization.kind": "bogus"})
        )


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="Unknown normalization kind"):
        NormalizationSpec(kind="bogus")
    with pytest.raises(ValueError, match="needs mean and std"):
        NormalizationSpec(kind="custom", mean=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="positive"):
        NormalizationSpec(kind="custom", mean=(0.5,) * 3, std=(0.5, 0.0, 0.5))
    with pytest.raises(ValueError, match="same length"):
        NormalizationSpec(kind="custom", mean=(0.5,) * 3, std=(0.5,) * 2)
    spec = NormalizationSpec(kind="imagenet", mean=IMAGENET_MEAN, std=IMAGENET_STD)
    with pytest.raises(ValueError, match="channels"):
        spec.apply(np.zeros((1, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="H, W"):
        spec.apply(np.zeros((3, 4), dtype=np.float32))


def test_dataset_applies_imagenet_normalization(toy_dataset_config) -> None:
    config = toy_dataset_config.with_overrides({"dataset.image_size": [40, 56]})
    dataset = build_dataset(config, "test")
    sample = dataset[0]
    assert sample["normalization"] == "imagenet"

    raw_rgb = tifffile.imread(dataset.items[0].image_path).astype(np.float32) / 255.0
    expected = (
        raw_rgb.transpose(2, 0, 1) - np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(-1, 1, 1)
    ) / np.asarray(IMAGENET_STD, dtype=np.float32).reshape(-1, 1, 1)

    np.testing.assert_allclose(sample["image"].numpy(), expected, rtol=1e-5, atol=1e-5)
    # the un-normalized [0, 1] image must differ: this catches a silent regression
    assert not np.allclose(sample["image"].numpy(), raw_rgb.transpose(2, 0, 1), atol=1e-3)
    assert float(sample["image"].min()) < -1.0  # mean subtraction makes values negative


def test_dataset_can_disable_normalization(toy_dataset_config) -> None:
    config = toy_dataset_config.with_overrides(
        {"dataset.image_size": [40, 56], "dataset.normalization.kind": "none"}
    )
    dataset = build_dataset(config, "test")
    sample = dataset[0]
    assert sample["normalization"] == "none"
    raw_rgb = tifffile.imread(dataset.items[0].image_path).astype(np.float32) / 255.0
    np.testing.assert_allclose(sample["image"].numpy(), raw_rgb.transpose(2, 0, 1), rtol=1e-6)
    assert 0.0 <= float(sample["image"].min()) and float(sample["image"].max()) <= 1.0

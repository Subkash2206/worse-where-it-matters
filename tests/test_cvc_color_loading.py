"""Colour loading of the real CVC-ClinicDB TIFFs: tifffile RGB, no BGR swap.

These tests need the dataset on disk and skip otherwise.  They are the regression
guard for a real bug: OpenCV collapses these malformed TIFFs to one channel and
replicates it, so the network silently saw grey frames.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import tifffile
import torch

from melba.data import build_dataset
from melba.data.cvc import _as_mask, _as_rgb_uint8
from melba.data.normalization import IMAGENET_MEAN, IMAGENET_STD

FRAMES: tuple[str, ...] = ("1", "2", "3", "10", "612")
NATIVE_SHAPE = (288, 384)


@pytest.fixture(scope="module")
def real_cvc(repo_root) -> object:
    root = repo_root / "cvc_clinicdb"
    if not (root / "original").is_dir():
        pytest.skip("CVC-ClinicDB is not available in this checkout")
    return root


def test_real_frames_are_rgb_with_distinct_channels(real_cvc) -> None:
    for name in FRAMES:
        image = tifffile.imread(real_cvc / "original" / f"{name}.tif")
        assert image.shape == (*NATIVE_SHAPE, 3), (name, image.shape)
        assert image.dtype == np.uint8
        rgb = _as_rgb_uint8(image, f"{name}.tif")
        assert not np.array_equal(rgb[..., 0], rgb[..., 1])
        assert not np.array_equal(rgb[..., 1], rgb[..., 2])
        assert float(rgb.std(axis=(0, 1)).min()) > 1.0


def test_real_masks_are_single_channel_and_binarise(real_cvc) -> None:
    intermediate_values: set[int] = set()
    for name in FRAMES:
        raw = tifffile.imread(real_cvc / "ground_truth" / f"{name}.tif")
        mask = _as_mask(raw, f"{name}.tif")
        assert mask.ndim == 2
        assert mask.shape == NATIVE_SHAPE
        assert mask.max() == 255
        binary = mask > 127
        assert set(np.unique(binary).tolist()) <= {False, True}
        assert bool(binary.any())  # every sampled frame contains a lesion
        intermediate_values |= set(np.unique(mask).tolist()) - {0, 255}
    # the raw masks are not strictly 0/255 (anti-aliased borders), hence the threshold
    assert intermediate_values


def test_dataset_sample_keeps_three_distinct_channels(base_config) -> None:
    dataset = build_dataset(base_config, "test", fold=0)
    assert len(dataset) > 0
    for index in range(3):
        sample = dataset[index]
        image = sample["image"]
        assert image.shape[0] == 3
        assert image.shape[1:] == (352, 352)
        assert sample["original_size"] == NATIVE_SHAPE
        assert sample["mask_native"].shape == (1, *NATIVE_SHAPE)
        assert sample["normalization"] == "imagenet"
        for first, second in ((0, 1), (1, 2)):
            assert not torch.allclose(image[first], image[second])
        assert set(torch.unique(sample["mask_native"]).tolist()) <= {0.0, 1.0}


def test_rgb_channel_order_is_preserved(base_config) -> None:
    """A BGR swap would reorder the three very different channel means."""
    dataset = build_dataset(base_config, "test", fold=0)
    item = dataset.items[0]
    height, width = dataset.image_size

    # mirror the dataset exactly: resize the uint8 frame, then scale to [0, 1]
    raw = _as_rgb_uint8(tifffile.imread(item.image_path), item.image_path)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb_chw = np.ascontiguousarray(resized.astype(np.float32).transpose(2, 0, 1) / 255.0)

    unnormalized = build_dataset(
        base_config.with_overrides({"dataset.normalization.kind": "none"}), "test", fold=0
    )[0]["image"]
    torch.testing.assert_close(
        unnormalized, torch.from_numpy(rgb_chw), rtol=1e-5, atol=1 / 255
    )

    # per-channel means keep the raw tifffile RGB order within 0.01 ...
    raw_means = np.asarray(raw, dtype=np.float64).mean(axis=(0, 1)) / 255.0
    sample_means = unnormalized.numpy().mean(axis=(1, 2))
    np.testing.assert_allclose(sample_means, raw_means, atol=0.01)
    # ... whereas a BGR swap could not (the channel means differ by far more than 0.01)
    assert float(np.abs(sample_means - raw_means[::-1]).max()) > 0.01

    normalized = dataset[0]["image"]
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(-1, 1, 1)
    torch.testing.assert_close(
        normalized, torch.from_numpy((rgb_chw - mean) / std), rtol=1e-5, atol=1e-5
    )


def test_as_rgb_rejects_grayscale_and_handles_wide_types() -> None:
    with pytest.raises(RuntimeError, match="single-channel"):
        _as_rgb_uint8(np.zeros((8, 8), np.uint8), "grey.tif")
    with pytest.raises(RuntimeError, match="single-channel"):
        _as_rgb_uint8(np.zeros((8, 8, 1), np.uint8), "grey.tif")
    with pytest.raises(RuntimeError, match="channel"):
        _as_rgb_uint8(np.zeros((8, 8, 2), np.uint8), "two.tif")

    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[..., 3] = 255
    assert _as_rgb_uint8(rgba, "rgba.tif").shape == (8, 8, 3)
    assert int(_as_rgb_uint8(np.full((4, 4, 3), 65535, np.uint16), "16bit.tif").max()) == 255


def test_as_mask_handles_antialiased_borders() -> None:
    mask = np.array([[0, 8, 112, 127, 128, 255]], np.uint8)
    out = _as_mask(mask, "mask.tif")
    assert out.dtype == np.uint8
    assert (out > 127).tolist() == [[False, False, False, False, True, True]]
    with pytest.raises(RuntimeError, match="single-channel"):
        _as_mask(np.zeros((4, 4, 3), np.uint8), "mask.tif")
    with pytest.raises(RuntimeError, match="integer"):
        _as_mask(np.zeros((4, 4), np.float32), "mask.tif")

"""Intensity TTA: transforms, view sets and determinism."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import load_config  # noqa: E402
from melba.transforms import build_views  # noqa: E402
from melba.transforms.intensity import (  # noqa: E402
    Brightness,
    Contrast,
    Gamma,
    GaussianNoise,
)

CONFIGS = {
    "cvc_unet": "configs/cvc_clinicdb/unet_resnet50_intensity_tta.yaml",
    "cvc_swin": "configs/cvc_clinicdb/swinunetr_2d_intensity_tta.yaml",
    "brats": "configs/brats2021/segresnet_intensity_tta.yaml",
}


@pytest.mark.parametrize(
    "transform",
    [Brightness(1.1), Contrast(1.1), Gamma(1.25), GaussianNoise(0.02, seed=0)],
)
def test_intensity_transform_leaves_the_prediction_geometry_untouched(transform) -> None:
    torch.manual_seed(0)
    image = torch.randn(3, 16, 18)
    changed = transform.apply(image)
    assert changed.shape == image.shape
    assert not torch.allclose(changed, image)
    # the probability map is already in the original geometry: invert must be the identity
    prediction = torch.rand(1, 16, 18)
    assert torch.equal(transform.invert(prediction, image.shape), prediction)


def test_factors_are_applied_per_channel_identically() -> None:
    """BraTS has four modalities: the same factor must hit each of them."""
    torch.manual_seed(1)
    volume = torch.randn(4, 8, 9, 10)
    ratio = Brightness(1.1).apply(volume) / volume
    assert torch.allclose(ratio, torch.full_like(volume, 1.1), atol=1e-5)
    centred = Contrast(1.1).apply(volume)
    # contrast preserves the per-channel mean and scales the deviation by 1.1
    mean_before = volume.mean(dim=(1, 2, 3))
    mean_after = centred.mean(dim=(1, 2, 3))
    assert torch.allclose(mean_before, mean_after, atol=1e-5)


def test_gamma_is_monotone_and_noise_is_deterministic() -> None:
    values = torch.linspace(-2, 2, 32).reshape(1, 4, 8)
    gamma = Gamma(0.8)
    out = gamma.apply(values)
    order_before = torch.argsort(values.flatten())
    order_after = torch.argsort(out.flatten())
    assert torch.equal(order_before, order_after)  # monotone mapping
    first = GaussianNoise(0.02, seed=7).apply(values)
    second = GaussianNoise(0.02, seed=7).apply(values)
    assert torch.equal(first, second)
    other = GaussianNoise(0.02, seed=8).apply(values)
    assert not torch.equal(first, other)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_config_view_sets_have_the_expected_membership(name: str) -> None:
    config = load_config(_REPO_ROOT / CONFIGS[name])
    methods = {method["name"]: method for method in config.get("tta.methods")}
    assert {"baseline", "tta4", "intensity4", "intensity8", "geo4_int4"} <= set(methods)
    counts = {
        key: len(build_views(method["views"])) for key, method in methods.items()
    }
    assert counts["baseline"] == 1 and counts["tta4"] == 4
    assert counts["intensity4"] == 4 and counts["intensity8"] == 8
    assert counts["geo4_int4"] == 8
    intensity4 = {view.name for view in build_views(methods["intensity4"]["views"])}
    assert intensity4 == {"identity", "bright_11", "contrast_11", "gamma_125"}
    intensity8 = {view.name for view in build_views(methods["intensity8"]["views"])}
    assert intensity8 == {
        "identity", "bright_11", "contrast_11", "gamma_125",
        "bright_09", "contrast_09", "gamma_08", "noise_002",
    }

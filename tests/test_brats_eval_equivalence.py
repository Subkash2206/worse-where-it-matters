"""Equivalence of the bbox-restricted metric path with the previous full-grid one.

Runs on the real fold-1 test cases (skipped when the preprocessed cache is missing), so
the numbers the new evaluation writes are the numbers the old one wrote: Dice identical,
distances identical to well below the 1e-3 mm tolerance.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import load_config  # noqa: E402
from melba.data.brats import (  # noqa: E402
    BraTSItem,
    load_case_for_pipeline,
    place_on_full_grid,
)
from melba.data.splits import read_split_csv  # noqa: E402
from melba.metrics import MetricSettings, compute_case_metrics  # noqa: E402
from melba.metrics.surface import (  # noqa: E402
    average_surface_distance,
    boundary_f1,
    foreground_bbox,
    hausdorff_distance,
    hausdorff_distance_95,
    hausdorff_distance_95_pooled,
    surface_metric_bundle,
)

CONFIG = _REPO_ROOT / "configs" / "brats2021" / "segresnet_reduced.yaml"
SPACING = (0.9, 1.1, 2.5)
CACHE_AVAILABLE = (_REPO_ROOT / "cache" / "brats2021_pre" / "index.json").is_file()
DISTANCE_TOLERANCE_MM = 1e-3


def _pair(
    seed: int = 0, shape: tuple[int, int, int] = (40, 44, 36)
) -> tuple[np.ndarray, np.ndarray]:
    """A blob pair inside a larger background: reference plus a smooth noisy prediction."""
    from scipy import ndimage

    target = np.zeros(shape, dtype=np.float32)
    target[12:24, 14:26, 12:22] = 1.0
    target[8:12, 20:24, 14:18] = 1.0
    rng = np.random.default_rng(seed)
    # smooth perturbation, like a real probability map (white noise would sprinkle
    # foreground everywhere and blur the point of the rim test)
    noise = ndimage.gaussian_filter(
        rng.normal(0.0, 1.0, size=shape).astype(np.float32), sigma=2.0
    )
    noise = noise / max(float(np.abs(noise).max()), 1e-6) * 0.35
    prediction = np.clip(target + noise, 0.0, 1.0)
    # keep the prediction away from the array border, so the bbox rim is real background
    prediction[:2] = prediction[-2:] = 0.0
    prediction[:, :2] = prediction[:, -2:] = 0.0
    prediction[:, :, :2] = prediction[:, :, -2:] = 0.0
    return prediction, target


def test_torch_uncertainty_matches_the_numpy_path() -> None:
    """The GPU reduction must reproduce the numpy descriptors (CPU run here)."""
    import torch

    from melba.metrics import compute_view_uncertainty, compute_view_uncertainty_torch
    from melba.metrics.uncertainty import _band_torch, foreground_band

    rng = np.random.default_rng(7)
    views = rng.random((4, 1, 12, 14, 10), dtype=np.float32)
    views[:, :, 3:7, 4:9, 3:8] = np.clip(
        views[:, :, 3:7, 4:9, 3:8] + 0.5, 0.0, 1.0
    )  # a foreground blob, so the ROI band is not empty

    numpy_values = compute_view_uncertainty(views, threshold=0.5, boundary_band_px=2)
    torch_values = compute_view_uncertainty_torch(
        torch.from_numpy(views), threshold=0.5, boundary_band_px=2
    )
    assert set(numpy_values) == set(torch_values)
    for name, value in numpy_values.items():
        assert torch_values[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name

    # the ROI band itself is the same voxel set as scipy's dilation
    mask = np.zeros((9, 11, 7), dtype=bool)
    mask[4, 5, 3] = True
    band_numpy = foreground_band(mask, 3)
    band_torch = _band_torch(torch.from_numpy(mask), 3).numpy()
    np.testing.assert_array_equal(band_torch, band_numpy)


def test_torch_uncertainty_single_view_agrees_with_numpy() -> None:
    import torch

    from melba.metrics import compute_view_uncertainty, compute_view_uncertainty_torch

    rng = np.random.default_rng(3)
    views = rng.random((1, 1, 8, 9, 10), dtype=np.float32)
    numpy_values = compute_view_uncertainty(views)
    torch_values = compute_view_uncertainty_torch(torch.from_numpy(views))
    assert torch_values["view_pairwise_dice"] == 1.0
    assert torch_values["view_disagreement"] == 0.0
    for name, value in numpy_values.items():
        assert torch_values[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name


def test_predict_per_view_tensor_matches_the_numpy_wrapper() -> None:
    """The tensor variant is what the runner uses; it must equal the numpy one."""
    import torch

    from melba.inference import predict_per_view, predict_per_view_tensor, TTAMethod
    from melba.transforms import build_views

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 1, kernel_size=3, padding=1), torch.nn.Sigmoid()
    ).eval()
    method = TTAMethod(
        name="tta2",
        views=tuple(
            build_views(
                [
                    {"name": "identity", "ops": []},
                    {"name": "flip_h", "ops": [{"type": "flip", "axes": ["h"]}]},
                ]
            )
        ),
        aggregator="mean",
    )
    image = torch.randn(1, 3, 16, 18)
    tensor = predict_per_view_tensor(
        model, image, torch.device("cpu"), method, activation="none", amp=False
    )
    array = predict_per_view(
        model, image, torch.device("cpu"), method, activation="none", amp=False
    )
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == tuple(array.shape)
    np.testing.assert_array_equal(tensor.numpy(), array)

    prediction, target = _pair()
    settings = MetricSettings(
        names=("hd", "hd95", "hd95_pooled", "assd", "boundary_f1"), connectivity=1
    )
    pred, ref = prediction > 0.5, target > 0.5
    bundle = surface_metric_bundle(
        pred, ref, SPACING, tolerance_mm=settings.boundary_tolerance_mm
    )
    assert bundle["hd"] == hausdorff_distance(pred, ref, SPACING, 1)
    assert bundle["hd95"] == hausdorff_distance_95(pred, ref, SPACING, percentile=95.0)
    assert bundle["hd95_pooled"] == hausdorff_distance_95_pooled(
        pred, ref, SPACING, percentile=95.0
    )
    assert bundle["assd"] == average_surface_distance(pred, ref, SPACING, 1)
    assert bundle["boundary_f1"] == boundary_f1(
        pred, ref, SPACING, tolerance_mm=settings.boundary_tolerance_mm
    )


def test_bbox_restriction_is_exact_for_every_metric() -> None:
    """Same masks, whole array vs union box: every value must be identical."""
    settings = MetricSettings(
        names=(
            "dice", "jaccard", "precision", "recall", "hd", "hd95",
            "hd95_pooled", "assd", "boundary_f1", "pred_volume_mm3",
        )
    )
    for seed in range(3):
        prediction, target = _pair(seed)
        boxed = compute_case_metrics(prediction, target, spacing=SPACING, settings=settings)
        whole = compute_case_metrics(
            prediction, target, spacing=SPACING, settings=settings, restrict_to_bbox=False
        )
        for name in settings.gt_names:
            assert boxed[name] == pytest.approx(whole[name], rel=0, abs=1e-12), name


def test_bbox_crop_keeps_a_background_rim() -> None:
    prediction, target = _pair()
    slices, touches_border = foreground_bbox(prediction > 0.5, target > 0.5, margin=2)
    assert not touches_border  # the blobs sit well inside the toy volume
    cropped = (prediction > 0.5)[slices]
    # no foreground may touch the cropped border, otherwise a surface would change
    assert not cropped[0].any() and not cropped[-1].any()
    assert not cropped[:, 0].any() and not cropped[:, -1].any()
    assert not cropped[:, :, 0].any() and not cropped[:, :, -1].any()


def test_bbox_restriction_handles_empty_masks() -> None:
    empty = np.zeros((8, 9, 10), dtype=np.float32)
    target = np.zeros((8, 9, 10), dtype=np.float32)
    target[2:5, 3:6, 4:7] = 1.0
    flagged = MetricSettings(
        names=("dice", "hd95", "boundary_f1"), score_empty_reference=False
    )
    metrics = compute_case_metrics(empty, target, spacing=1.0, settings=flagged)
    assert metrics["empty_flag"] == "pred_empty"
    assert metrics["dice"] == 0.0 and math.isnan(metrics["hd95"])
    # both empty: distances are 0, and the score is left undefined by the BraTS policy
    both = compute_case_metrics(empty, empty, spacing=1.0, settings=flagged)
    assert both["empty_flag"] == "both_empty"
    assert both["hd95"] == 0.0 and math.isnan(both["boundary_f1"])
    legacy = MetricSettings(names=("hd95", "boundary_f1"))
    perfect = compute_case_metrics(empty, empty, spacing=1.0, settings=legacy)
    assert perfect["hd95"] == 0.0 and perfect["boundary_f1"] == 1.0


def test_penalty_uses_the_original_grid_shape() -> None:
    """A cropped evaluation must not shrink the empty-prediction penalty."""
    prediction, target = _pair()
    settings = MetricSettings(names=("hd95",), empty_policy="penalize")
    empty = np.zeros_like(prediction)
    cropped = compute_case_metrics(
        empty, target, spacing=SPACING, settings=settings, penalty_shape=(100, 100, 100)
    )
    full = compute_case_metrics(
        empty,
        target,
        spacing=SPACING,
        settings=settings,
        penalty_shape=(100, 100, 100),
        restrict_to_bbox=False,
    )
    assert cropped["hd95"] == full["hd95"]
    assert cropped["hd95"] > 0.0


@pytest.mark.skipif(not CACHE_AVAILABLE, reason="needs the BraTS 2021 cache")
def test_metric_path_matches_the_previous_full_grid_path_on_fold1_cases() -> None:
    """Five real fold-1 test cases: new (bbox + one transform) vs old (padded, five)."""
    config = load_config(CONFIG, {"seed": 0, "evaluation.fold": 1})
    settings = MetricSettings.from_config(config)
    cases = [
        record.case_id
        for record in read_split_csv(_REPO_ROOT / "data" / "splits" / "brats2021_fold1.csv")
        if record.split == "test"
    ][:5]
    cache_dir = config.resolve(str(config.require("dataset.cache_dir")))
    root = config.resolve(str(config.require("dataset.root")))
    assert len(cases) == 5

    for case in cases:
        item = BraTSItem(case_id=case, group_id=case, split="test", directory=root / case)
        _image, mask, spacing, meta = load_case_for_pipeline(
            item, use_cache=True, cache_dir=cache_dir
        )
        target = mask.astype(np.float32)
        rng = np.random.default_rng(0)
        prediction = np.clip(
            target + rng.normal(0.0, 0.05, size=target.shape).astype(np.float32), 0.0, 1.0
        )
        offsets = tuple(int(value) for value in meta["offsets"])
        original = tuple(int(value) for value in meta["original_shape"])

        # new path: what the runner does now (cached crop, bbox restriction, one transform)
        new = compute_case_metrics(
            prediction,
            target,
            spacing=spacing,
            settings=settings,
            penalty_shape=original,
        )
        # previous path: pad to 240x240x155 and recompute a transform pair per metric
        full_prediction = place_on_full_grid(
            prediction, offsets=offsets, original_shape=original
        )
        full_target = place_on_full_grid(target, offsets=offsets, original_shape=original)
        old = compute_case_metrics(
            full_prediction,
            full_target,
            spacing=spacing,
            settings=settings,
            restrict_to_bbox=False,
        )

        for name in ("dice", "jaccard", "precision", "recall", "pred_volume_mm3"):
            assert new[name] == pytest.approx(old[name], rel=0, abs=1e-12), (case, name)
        for name in ("hd", "hd95", "hd95_pooled", "assd"):
            if math.isnan(old[name]):
                assert math.isnan(new[name]), (case, name)
            else:
                assert abs(new[name] - old[name]) <= DISTANCE_TOLERANCE_MM, (case, name)
        assert abs(new["boundary_f1"] - old["boundary_f1"]) <= 1e-9, case
        assert new["empty_flag"] == old["empty_flag"], case


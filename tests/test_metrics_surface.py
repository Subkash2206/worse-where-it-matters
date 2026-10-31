"""Surface metrics: spacing, percentiles, surfaces and empty-mask policies.

These tests encode the fixes for the old implementation, which returned the
maximum directed Hausdorff distance over all foreground pixels and ignored
spacing.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist

from melba.metrics import (
    MetricSettings,
    average_surface_distance,
    boundary_f1,
    compute_case_metrics,
    grid_diagonal_mm,
    hausdorff_distance,
    hausdorff_distance_95,
    hausdorff_distance_95_pooled,
    surface_distances,
    surface_mask,
)


def _two_dots(shape=(9, 9), first=(4, 4), second=(4, 7)):
    a = np.zeros(shape, dtype=bool)
    b = np.zeros(shape, dtype=bool)
    a[first] = True
    b[second] = True
    return a, b


def test_identical_masks_are_perfect() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 3:7] = True
    assert hausdorff_distance(mask, mask) == 0.0
    assert hausdorff_distance_95(mask, mask) == 0.0
    assert average_surface_distance(mask, mask) == 0.0
    assert boundary_f1(mask, mask, tolerance_mm=0.0) == 1.0


def test_single_voxel_distance_matches_geometry() -> None:
    pred, target = _two_dots()
    assert hausdorff_distance(pred, target) == pytest.approx(3.0)
    assert hausdorff_distance_95(pred, target) == pytest.approx(3.0)
    assert average_surface_distance(pred, target) == pytest.approx(3.0)
    assert boundary_f1(pred, target, tolerance_mm=2.0) == 0.0
    assert boundary_f1(pred, target, tolerance_mm=3.0) == 1.0


def test_spacing_scales_distances() -> None:
    pred, target = _two_dots()
    # shift is along axis 1
    assert hausdorff_distance(pred, target, spacing=(1.0, 3.0)) == pytest.approx(9.0)
    assert average_surface_distance(pred, target, spacing=(7.0, 3.0)) == pytest.approx(9.0)

    pred_rows = np.zeros((9, 9), dtype=bool)
    target_rows = np.zeros((9, 9), dtype=bool)
    pred_rows[2, 4] = True
    target_rows[5, 4] = True
    assert hausdorff_distance(pred_rows, target_rows, spacing=(2.5, 1.0)) == pytest.approx(7.5)


def test_hd95_is_robust_to_a_small_outlier_blob() -> None:
    """The old code returned HD100 over foreground pixels; HD95 must not do that."""
    target = np.zeros((60, 60), dtype=bool)
    target[5:25, 5:25] = True
    pred = target.copy()
    pred[50:52, 50:52] = True  # small disconnected false positive

    hd = hausdorff_distance(pred, target)
    hd95 = hausdorff_distance_95(pred, target)
    assert hd > 20.0
    assert hd95 < 2.0
    assert hd95 < hd


def test_surface_distances_match_brute_force() -> None:
    rng = np.random.default_rng(1234)
    for _ in range(5):
        pred = rng.random((14, 17)) > 0.55
        target = rng.random((14, 17)) > 0.6
        spacing = (0.7, 2.3)
        d_pred, d_target = surface_distances(pred, target, spacing=spacing)
        pred_points = np.argwhere(surface_mask(pred)) * np.asarray(spacing)
        target_points = np.argwhere(surface_mask(target)) * np.asarray(spacing)
        reference_pred = cdist(pred_points, target_points).min(axis=1)
        reference_target = cdist(target_points, pred_points).min(axis=1)
        np.testing.assert_allclose(np.sort(d_pred), np.sort(reference_pred), atol=1e-9)
        np.testing.assert_allclose(np.sort(d_target), np.sort(reference_target), atol=1e-9)


def test_metric_ordering_and_percentile_definition() -> None:
    rng = np.random.default_rng(7)
    pred = rng.random((20, 20)) > 0.6
    target = rng.random((20, 20)) > 0.6
    hd = hausdorff_distance(pred, target, spacing=1.0)
    hd95 = hausdorff_distance_95(pred, target, spacing=1.0)
    hd95_pooled = hausdorff_distance_95_pooled(pred, target, spacing=1.0)
    assd = average_surface_distance(pred, target, spacing=1.0)
    assert 0.0 <= assd <= hd95 <= hd

    d_pred, d_target = surface_distances(pred, target, spacing=1.0)
    pooled = np.concatenate([d_pred, d_target])
    # MONAI convention: reduce each direction, then take the maximum
    assert hd95 == pytest.approx(
        max(float(np.percentile(d_pred, 95)), float(np.percentile(d_target, 95)))
    )
    assert hd95_pooled == pytest.approx(float(np.percentile(pooled, 95)))
    assert hd == pytest.approx(float(pooled.max()))


def test_hd95_differs_from_pooled_on_asymmetric_errors() -> None:
    """A small far-away blob moves the directed percentile but not the pooled one."""
    target = np.zeros((60, 60), dtype=bool)
    target[5:25, 5:25] = True
    pred = target.copy()
    pred[50:52, 50:52] = True

    d_pred, d_target = surface_distances(pred, target, spacing=1.0)
    pooled = np.concatenate([d_pred, d_target])
    monai_style = hausdorff_distance_95(pred, target)
    pooled_style = hausdorff_distance_95_pooled(pred, target)

    assert monai_style == pytest.approx(
        max(float(np.percentile(d_pred, 95)), float(np.percentile(d_target, 95)))
    )
    assert pooled_style == pytest.approx(float(np.percentile(pooled, 95)))
    assert monai_style > pooled_style


def _random_blob(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    centre = [int(rng.integers(4, size - 4)) for size in shape]
    radii = [int(rng.integers(2, 4)) for _ in shape]
    grid = np.ogrid[tuple(slice(0, size) for size in shape)]
    distance = sum(((axis - c) / r) ** 2 for axis, c, r in zip(grid, centre, radii))
    mask[distance <= 1.0] = True
    return mask


def _monai_hd95(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> float:
    from monai.metrics import compute_hausdorff_distance

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = compute_hausdorff_distance(
            torch.from_numpy(pred[None, None].astype(np.float32)),
            torch.from_numpy(target[None, None].astype(np.float32)),
            percentile=95,
            spacing=spacing,
        )
    return float(value.item())


@pytest.mark.parametrize("n_pairs", [20])
def test_hd95_matches_monai_2d(n_pairs: int) -> None:
    rng = np.random.default_rng(2024)
    spacing = (1.2, 0.8)
    for _ in range(n_pairs):
        pred = _random_blob(rng, (32, 30))
        target = _random_blob(rng, (32, 30))
        assert hausdorff_distance_95(
            pred, target, spacing=spacing
        ) == pytest.approx(_monai_hd95(pred, target, spacing), abs=1e-4)


@pytest.mark.parametrize("n_pairs", [20])
def test_hd95_matches_monai_3d(n_pairs: int) -> None:
    rng = np.random.default_rng(7)
    spacing = (1.5, 0.9, 2.0)
    for _ in range(n_pairs):
        pred = _random_blob(rng, (16, 18, 20))
        target = _random_blob(rng, (16, 18, 20))
        assert hausdorff_distance_95(
            pred, target, spacing=spacing
        ) == pytest.approx(_monai_hd95(pred, target, spacing), abs=1e-4)


def test_boundary_f1_is_symmetric_and_bounded() -> None:
    rng = np.random.default_rng(3)
    pred = rng.random((18, 18)) > 0.5
    target = rng.random((18, 18)) > 0.5
    forward = boundary_f1(pred, target, tolerance_mm=1.5)
    backward = boundary_f1(target, pred, tolerance_mm=1.5)
    assert forward == pytest.approx(backward)
    assert 0.0 <= forward <= 1.0


def test_surface_mask_of_full_array_is_the_whole_border() -> None:
    full = np.ones((3, 3), dtype=bool)
    border = surface_mask(full)
    # the outside of the array is background, so all 8 border voxels are surface
    assert int(border.sum()) == 8
    assert not border[1, 1]
    assert int(surface_mask(np.zeros((3, 3), dtype=bool)).sum()) == 0


def test_empty_mask_behaviour() -> None:
    empty = np.zeros((5, 5), dtype=bool)
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:3, 1:3] = True

    assert hausdorff_distance(empty, empty) == 0.0
    assert hausdorff_distance_95(empty, empty) == 0.0
    assert average_surface_distance(empty, empty) == 0.0
    assert boundary_f1(empty, empty) == 1.0

    assert math.isnan(hausdorff_distance_95(empty, mask))
    assert math.isnan(boundary_f1(mask, empty))
    assert hausdorff_distance_95(empty, mask, empty_value=42.0) == 42.0
    assert boundary_f1(mask, empty, empty_value=0.25) == 0.25


def test_registry_penalize_and_nan_policies() -> None:
    empty = np.zeros((10, 12), dtype=bool)
    mask = np.zeros((10, 12), dtype=bool)
    mask[3:6, 4:8] = True
    spacing = (2.0, 1.5)
    penalty = grid_diagonal_mm(mask.shape, spacing)

    nan_settings = MetricSettings(empty_policy="nan")
    penalised_settings = MetricSettings(empty_policy="penalize")
    nan_metrics = compute_case_metrics(
        empty.astype(float), mask.astype(float), spacing, settings=nan_settings
    )
    penalised = compute_case_metrics(
        empty.astype(float), mask.astype(float), spacing, settings=penalised_settings
    )

    assert math.isnan(nan_metrics["hd95"])
    assert math.isnan(nan_metrics["assd"])
    assert penalised["hd95"] == pytest.approx(penalty)
    assert penalised["assd"] == pytest.approx(penalty)
    assert penalised["boundary_f1"] == 0.0
    assert penalised["dice"] == 0.0
    assert penalised["precision"] == 0.0
    assert penalised["gt_volume_mm3"] == pytest.approx(12 * 2.0 * 1.5)


def test_registry_both_empty_is_perfect_and_volume_free() -> None:
    empty = np.zeros((6, 6), dtype=bool)
    metrics = compute_case_metrics(empty.astype(float), empty.astype(float), spacing=1.0)
    assert metrics["dice"] == 1.0
    assert metrics["hd95"] == 0.0
    assert metrics["boundary_f1"] == 1.0
    assert metrics["pred_volume_mm3"] == 0.0


def test_registry_volume_metrics_and_threshold() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[0:2, 0:2] = True
    probability = np.where(mask, 0.9, 0.1).astype(np.float32)
    metrics = compute_case_metrics(
        probability, mask.astype(np.float32), spacing=(2.0, 3.0), threshold=0.5
    )
    assert metrics["dice"] == 1.0
    assert metrics["pred_volume_mm3"] == pytest.approx(4 * 6.0)
    assert metrics["volume_difference_mm3"] == 0.0


def test_metric_settings_from_config(base_config) -> None:
    settings = MetricSettings.from_config(base_config)
    assert settings.empty_policy == "nan"
    assert settings.hd_percentile == 95.0
    assert settings.boundary_tolerance_mm == 2.0
    assert "hd95" in settings.names and "hd95_pooled" in settings.names
    assert "entropy_mean" in settings.uncertainty_names
    assert "dice" not in settings.uncertainty_names


def test_invalid_settings_and_inputs() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        MetricSettings(names=("dice", "nope"))
    with pytest.raises(ValueError, match="empty_policy"):
        MetricSettings(empty_policy="drop")
    with pytest.raises(ValueError, match="percentile"):
        hausdorff_distance_95(np.ones((3, 3), bool), np.ones((3, 3), bool), percentile=0)
    with pytest.raises(ValueError, match="spacing"):
        hausdorff_distance(np.ones((3, 3), bool), np.ones((3, 3), bool), spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="shapes differ"):
        surface_distances(np.ones((3, 3), bool), np.ones((3, 4), bool))


def test_3d_surfaces() -> None:
    target = np.zeros((12, 12, 12), dtype=bool)
    target[3:8, 3:8, 3:8] = True
    pred = target.copy()
    pred[3:8, 3:8, 9:11] = True  # extra shell above the cube

    assert hausdorff_distance(target, target, spacing=(1.0, 1.0, 2.5)) == 0.0
    hd95 = hausdorff_distance_95(pred, target, spacing=(1.0, 1.0, 2.5))
    assert hd95 > 0.0
    assert boundary_f1(pred, target, spacing=(1.0, 1.0, 2.5), tolerance_mm=1.0) < 1.0


def test_empty_flag_and_default_nan_policy() -> None:
    empty = np.zeros((6, 6), dtype=np.float32)
    mask = np.zeros((6, 6), dtype=np.float32)
    mask[1:3, 1:3] = 1.0

    pred_empty = compute_case_metrics(empty, mask, spacing=1.0)
    gt_empty = compute_case_metrics(mask, empty, spacing=1.0)
    both_empty = compute_case_metrics(empty, empty, spacing=1.0)
    both_nonempty = compute_case_metrics(mask, mask, spacing=1.0)

    assert pred_empty["empty_flag"] == "pred_empty"
    assert gt_empty["empty_flag"] == "gt_empty"
    assert both_empty["empty_flag"] == "both_empty"
    assert both_nonempty["empty_flag"] == "both_nonempty"

    # default policy is `nan`, never a silent penalty
    assert math.isnan(pred_empty["hd95"])
    assert math.isnan(pred_empty["assd"])
    assert math.isnan(pred_empty["boundary_f1"])

    penalised = compute_case_metrics(
        empty, mask, spacing=1.0, settings=MetricSettings(empty_policy="penalize")
    )
    assert penalised["empty_flag"] == "pred_empty"
    assert penalised["hd95"] == pytest.approx(grid_diagonal_mm((6, 6), (1.0, 1.0)))
    assert penalised["boundary_f1"] == 0.0


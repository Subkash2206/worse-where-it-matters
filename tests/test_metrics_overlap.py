"""Overlap metrics: hand-computed values and documented empty-mask behaviour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from melba.metrics import (
    as_binary,
    confusion_counts,
    dice,
    jaccard,
    precision,
    recall,
    volume_mm3,
)


def test_confusion_counts() -> None:
    pred = np.zeros((4, 4), dtype=bool)
    target = np.zeros((4, 4), dtype=bool)
    pred[0, 0] = pred[0, 1] = pred[1, 0] = True
    target[0, 0] = target[0, 1] = target[2, 2] = True
    tp, fp, fn = confusion_counts(pred, target)
    assert (tp, fp, fn) == (2, 1, 1)


def test_hand_computed_scores() -> None:
    pred = np.array([[1, 1, 0, 0]], dtype=bool)
    target = np.array([[1, 0, 0, 1]], dtype=bool)
    # tp=1, fp=1, fn=1
    assert dice(pred, target) == pytest.approx(2 / 4)
    assert jaccard(pred, target) == pytest.approx(1 / 3)
    assert precision(pred, target) == pytest.approx(1 / 2)
    assert recall(pred, target) == pytest.approx(1 / 2)


def test_dice_and_jaccard_relation() -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        pred = rng.random((8, 9)) > 0.5
        target = rng.random((8, 9)) > 0.5
        d = dice(pred, target)
        j = jaccard(pred, target)
        assert d == pytest.approx(2 * j / (1 + j))


def test_empty_masks() -> None:
    empty = np.zeros((3, 3), dtype=bool)
    full = np.ones((3, 3), dtype=bool)
    assert dice(empty, empty) == 1.0
    assert jaccard(empty, empty) == 1.0
    assert dice(empty, full) == 0.0
    assert jaccard(empty, full) == 0.0
    assert math.isnan(precision(empty, full))
    assert math.isnan(recall(full, empty))
    assert recall(full, full) == 1.0


def test_as_binary_thresholds() -> None:
    probabilities = np.array([0.0, 0.49, 0.5, 0.51, 1.0])
    np.testing.assert_array_equal(as_binary(probabilities), [False, False, False, True, True])
    np.testing.assert_array_equal(as_binary(probabilities, 0.1), [False, True, True, True, True])
    labels = np.array([0, 3, 0, 7])
    np.testing.assert_array_equal(as_binary(labels), [False, True, False, True])
    boolean = np.array([True, False])
    assert as_binary(boolean) is boolean


def test_volume_uses_spacing() -> None:
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:3, 1:3, 1:3] = True  # 8 voxels
    assert volume_mm3(mask) == pytest.approx(8.0)
    assert volume_mm3(mask, spacing=(1.0, 2.0, 0.5)) == pytest.approx(8.0)
    assert volume_mm3(mask, spacing=(2.0, 2.0, 2.0)) == pytest.approx(64.0)
    with pytest.raises(ValueError):
        volume_mm3(mask, spacing=(1.0, 1.0))
    with pytest.raises(ValueError):
        volume_mm3(mask, spacing=(1.0, 0.0, 1.0))


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        dice(np.zeros((2, 2), dtype=bool), np.zeros((2, 3), dtype=bool))

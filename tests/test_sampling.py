"""Seeded epoch sampling: reproducible patches per epoch and a fixed val subsample."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.data import FixedSubsetSampler, SeededEpochSampler  # noqa: E402


def test_seeded_epoch_sampler_is_reproducible_and_epoch_aware() -> None:
    sampler = SeededEpochSampler(100, 10, seed=7)
    first = list(sampler)  # epoch 0
    assert first == list(SeededEpochSampler(100, 10, seed=7))
    assert len(first) == 10 and len(set(first)) == 10  # without replacement
    assert all(0 <= index < 100 for index in first)
    sampler.set_epoch(1)
    second = list(sampler)
    assert second != first  # a fresh subset every epoch
    sampler.set_epoch(0)
    assert list(sampler) == first  # ... and the same one again after a resume
    other = list(SeededEpochSampler(100, 10, seed=8))
    assert other != first  # the run seed changes the sequence


def test_seeded_epoch_sampler_handles_more_patches_than_cases() -> None:
    sampler = SeededEpochSampler(5, 12, seed=0)
    indices = list(sampler)
    assert len(indices) == 12
    assert sampler.replacement is True  # 12 > 5 requires replacement
    assert set(indices) <= set(range(5))
    with pytest.raises(ValueError):
        SeededEpochSampler(0, 3)
    with pytest.raises(ValueError):
        SeededEpochSampler(3, 0)


def test_fixed_subset_sampler_is_stable_and_sorted() -> None:
    first = FixedSubsetSampler(250, 40, seed=3)
    assert len(first) == 40
    assert list(first) == sorted(first)
    assert list(first) == list(FixedSubsetSampler(250, 40, seed=3))
    assert list(first) != list(FixedSubsetSampler(250, 40, seed=4))
    capped = FixedSubsetSampler(10, 40, seed=0)  # cannot exceed the pool
    assert len(capped) == 10

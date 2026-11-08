"""Seeded, epoch-aware sampling of a fixed number of patches per epoch.

A 3D BraTS epoch that visits every training case would cost ~10 minutes of GPU time for
751 patches; the reduced protocol caps it at a fixed number of *random* patches per epoch
instead.  The sampler draws a fresh, reproducible subset every epoch (seed, epoch), so two
runs with the same seed see the same sequence of patches and a resumed run continues with
the same stream.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

__all__ = ["SeededEpochSampler"]


class SeededEpochSampler(Sampler[int]):
    """Yield ``n_samples`` indices per epoch, drawn with ``seed`` and the epoch number."""

    def __init__(
        self,
        n_items: int,
        n_samples: int,
        *,
        seed: int = 0,
        epoch: int = 0,
        replacement: bool | None = None,
    ) -> None:
        if int(n_items) < 1:
            raise ValueError(f"n_items must be >= 1, got {n_items}")
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        self.n_items = int(n_items)
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.epoch = int(epoch)
        # sampling without replacement is the default; more patches than cases needs it
        self.replacement = (
            bool(replacement) if replacement is not None else self.n_samples > self.n_items
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_samples

    def indices(self) -> np.ndarray:
        rng = np.random.default_rng([self.seed, self.epoch])
        return rng.choice(
            self.n_items, size=self.n_samples, replace=self.replacement
        ).astype(np.int64)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices().tolist())


class FixedSubsetSampler(Sampler[int]):
    """A fixed set of indices, for a deterministic validation subsample."""

    def __init__(self, n_items: int, n_samples: int, *, seed: int = 0) -> None:
        if int(n_items) < 1:
            raise ValueError(f"n_items must be >= 1, got {n_items}")
        self.indices_ = np.sort(
            np.random.default_rng(int(seed)).choice(
                int(n_items), size=min(int(n_samples), int(n_items)), replace=False
            )
        ).astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices_.size)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices_.tolist())

    @property
    def indices(self) -> Sequence[int]:
        return self.indices_.tolist()

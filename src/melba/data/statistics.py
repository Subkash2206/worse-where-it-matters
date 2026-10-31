"""Training-set intensity statistics for from-scratch models.

ImageNet normalization is only correct for ImageNet-pretrained encoders.  A model
trained from scratch (e.g. ``monai_swinunetr``) is normalized with the per-channel
mean/std of *its own fold's training split*, computed by
``scripts/compute_normalization.py`` and stored in a JSON file keyed by fold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from melba.config import Config

__all__ = ["compute_fold_statistics", "read_statistics", "write_statistics"]


def compute_fold_statistics(
    config: Config, fold: int, *, progress: bool = False
) -> dict[str, Any]:
    """Per-channel mean/std over the ``[0, 1]`` images of one fold's training split.

    The dataset is built with normalization disabled, so the statistics are computed
    on the raw resized images the network would otherwise see, and without
    augmentation (``train=False``) so the numbers are not seed dependent.
    """
    from melba.data import build_dataset  # local import: avoids a package cycle

    unnormalized = config.with_overrides({"dataset.normalization.kind": "none"})
    dataset = build_dataset(unnormalized, "train", train=False, fold=int(fold))
    indices: Iterable[int] = range(len(dataset))
    if progress:
        from tqdm import tqdm

        indices = tqdm(indices, desc=f"fold {fold}", unit="frame")

    total: np.ndarray | None = None
    total_squared: np.ndarray | None = None
    pixels = 0
    for index in indices:
        image = np.asarray(dataset[index]["image"], dtype=np.float64)
        if image.ndim != 3:
            raise ValueError(f"Expected a (C, H, W) image, got shape {image.shape}")
        if total is None:
            total = np.zeros(image.shape[0], dtype=np.float64)
            total_squared = np.zeros(image.shape[0], dtype=np.float64)
        total += image.sum(axis=(1, 2))
        total_squared += np.square(image).sum(axis=(1, 2))
        pixels += image.shape[1] * image.shape[2]

    if total is None or total_squared is None or pixels == 0:
        raise ValueError(f"fold {fold} has no training frames; cannot compute statistics")
    mean = total / pixels
    variance = np.maximum(total_squared / pixels - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std <= 0):
        raise ValueError(f"degenerate (zero-variance) channel in fold {fold}: std={std}")
    return {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "fold": int(fold),
        "n_frames": len(dataset),
        "n_pixels": int(pixels),
        "image_size": list(dataset.image_size),
    }


def read_statistics(path: str | Path) -> dict[str, Any]:
    """Read a statistics JSON file (empty mapping when it does not exist yet)."""
    statistics_path = Path(path)
    if not statistics_path.is_file():
        return {}
    return json.loads(statistics_path.read_text(encoding="utf-8"))


def write_statistics(
    path: str | Path,
    entries: Mapping[str, Mapping[str, Any]],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Merge ``entries`` into the statistics file, replacing folds only with *force*."""
    statistics_path = Path(path)
    payload = dict(read_statistics(statistics_path))
    written: dict[str, Any] = {}
    for key, entry in entries.items():
        if key in payload and not force:
            continue
        payload[key] = dict(entry)
        written[key] = dict(entry)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return written

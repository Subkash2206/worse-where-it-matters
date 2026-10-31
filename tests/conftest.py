"""Shared fixtures: real configs plus a small synthetic CVC-ClinicDB-like dataset.

The toy dataset is written with OpenCV (the distributed CVC TIFFs are not
readable by Pillow) and split at *group* level so the end-to-end tests exercise
the same code paths as the real data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # safety net when pytest's pythonpath ini is unavailable
    sys.path.insert(0, str(SRC))

from melba.config import Config, load_config  # noqa: E402
from melba.data import folded_splits, write_split_csv  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def cvc_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "cvc_clinicdb" / "unet_resnet50.yaml"


@pytest.fixture(scope="session")
def base_config(cvc_config_path: Path) -> Config:
    return load_config(cvc_config_path)


def _write_frame(image_path: Path, mask_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    height, width = 40, 56
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    top, left = (seed * 3) % (height - 14), (seed * 5) % (width - 14)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top : top + 10, left : left + 12] = 255
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask)


@pytest.fixture
def toy_cvc(tmp_path: Path) -> dict[str, Any]:
    """A 12-frame, 6-video toy dataset with 5 valid video-level folds."""
    data_dir = tmp_path / "cvc_toy"
    images_dir = data_dir / "original"
    masks_dir = data_dir / "ground_truth"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    group_map_path = tmp_path / "frame_groups.csv"
    cases: list[str] = []
    group_lines = ["case_id,group_id"]
    for video in range(6):
        for frame in range(2):
            name = f"v{video}_f{frame}"
            _write_frame(images_dir / f"{name}.tif", masks_dir / f"{name}.tif", seed=video * 10 + frame)
            cases.append(name)
            group_lines.append(f"{name},video_{video}")
    group_map_path.write_text("\n".join(group_lines) + "\n", encoding="utf-8")

    group_map = {
        line.split(",")[0]: line.split(",")[1] for line in group_lines[1:] if line.strip()
    }
    folds = folded_splits(cases, group_map, n_folds=5, seed=7)
    split_pattern = tmp_path / "split_fold{fold}.csv"
    for fold, records in folds.items():
        write_split_csv(records, Path(str(split_pattern).format(fold=fold)))

    return {
        "root": data_dir,
        "images_dir": images_dir,
        "masks_dir": masks_dir,
        "group_map": group_map_path,
        "split_pattern": str(split_pattern),
        "cases": cases,
        "folds": folds,
        "records": folds[0],
    }


def toy_config(
    base: Config,
    toy: dict[str, Any],
    tmp_path: Path,
    **overrides: Any,
) -> Config:
    """Point the real CVC config at the toy dataset and at temporary outputs."""
    settings: dict[str, Any] = {
        "device": "cpu",
        "runtime.num_workers": 0,
        "runtime.progress": False,
        "dataset.images_dir": str(toy["images_dir"]),
        "dataset.masks_dir": str(toy["masks_dir"]),
        "dataset.image_size": [64, 64],
        "dataset.split.group_map_file": str(toy["group_map"]),
        "dataset.split.output_pattern": toy["split_pattern"],
        "dataset.split.n_folds": 5,
        "dataset.split.fold": 0,
        "dataset.split.seed": 7,
        "evaluation.fold": 0,
        "model.checkpoint": None,
        "evaluation.output_dir": str(tmp_path / "results"),
        "stats.output_dir": str(tmp_path / "stats"),
        "training.checkpoint_dir": str(tmp_path / "checkpoints"),
    }
    settings.update(overrides)
    return base.with_overrides(settings)


@pytest.fixture
def toy_dataset_config(base_config: Config, toy_cvc: dict[str, Any], tmp_path: Path) -> Config:
    return toy_config(
        base_config,
        toy_cvc,
        tmp_path,
        **{"evaluation.split": "test"},
    )

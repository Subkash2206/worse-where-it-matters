"""Datasets, splits and data loaders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from melba.config import Config, ConfigError
from melba.utils.reproducibility import make_worker_init_fn

from .cvc import (
    AUGMENTATION_GROUPS,
    AugmentationConfig,
    CVCDataset,
    CVCItem,
    discover_cvc_items,
    load_cvc_items,
)
from .brats import (
    BRATS_AUGMENTATION_GROUPS,
    BRATSDataset,
    ET_LABEL,
    MODALITIES,
    BraTSAugmentation,
    BraTSItem,
    discover_brats_cases,
    foreground_crop,
    load_brats_items,
    percentile_normalize,
    read_brats_case,
)
from .cvc_mapping import (
    build_sequence_mapping,
    check_mapping_against_stems,
    mapping_statistics,
    sequence_ranges,
    validate_sequence_mapping,
    write_mapping_csv,
)
from .normalization import (
    DEFAULT_STATS_FILE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NORMALIZATION_KINDS,
    NormalizationSpec,
    load_dataset_statistics,
)
from .statistics import compute_fold_statistics, read_statistics, write_statistics
from .splits import (
    SPLITS,
    SplitError,
    SplitRecord,
    actual_fractions,
    assert_folds_cover_once,
    assert_no_group_leakage,
    contiguous_group_map,
    filter_records,
    folded_splits,
    folds_summary,
    group_folds,
    grouped_split,
    load_group_map,
    read_split_csv,
    split_file_for_fold,
    split_summary,
    write_split_csv,
)

__all__ = [
    "AUGMENTATION_GROUPS",
    "BRATS_AUGMENTATION_GROUPS",
    "BRATSDataset",
    "BraTSAugmentation",
    "BraTSItem",
    "DEFAULT_STATS_FILE",
    "ET_LABEL",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MODALITIES",
    "NORMALIZATION_KINDS",
    "NormalizationSpec",
    "SPLITS",
    "AugmentationConfig",
    "CVCDataset",
    "CVCItem",
    "SplitError",
    "SplitRecord",
    "actual_fractions",
    "assert_folds_cover_once",
    "assert_no_group_leakage",
    "build_dataset",
    "build_dataloader",
    "build_sequence_mapping",
    "check_mapping_against_stems",
    "compute_fold_statistics",
    "contiguous_group_map",
    "discover_brats_cases",
    "discover_cases",
    "discover_cvc_items",
    "filter_records",
    "folded_splits",
    "folds_summary",
    "foreground_crop",
    "group_folds",
    "grouped_split",
    "load_brats_items",
    "load_cvc_items",
    "load_dataset_statistics",
    "load_group_map",
    "mapping_statistics",
    "percentile_normalize",
    "read_brats_case",
    "read_split_csv",
    "sequence_ranges",
    "split_file_for_fold",
    "split_summary",
    "subsample_training_groups",
    "validate_sequence_mapping",
    "write_mapping_csv",
    "write_split_csv",
    "write_statistics",
]


def discover_cases(config: Config) -> list[str]:
    """Case identifiers (file stems) found on disk, independent of any split."""
    name = str(config.require("dataset.name")).strip().lower()
    if name == "cvc_clinicdb":
        return [item.case_id for item in discover_cvc_items(config)]
    raise ConfigError(
        f"Unknown dataset '{name}'. Implemented datasets: cvc_clinicdb (3D BraTS support "
        "is the next milestone)."
    )


def subsample_training_groups(
    items: Sequence[Any],
    fraction: float,
    seed: int,
) -> list[Any]:
    """Keep a seeded subset of *training videos* (never frames) for data-scaling runs.

    The subset is drawn from whole groups so that frames of one video stay together,
    and at least one group is always kept.
    """
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ConfigError(f"training.data_fraction must be in (0, 1], got {fraction}")
    groups = sorted({str(item.group_id) for item in items})
    keep = max(1, int(round(fraction * len(groups))))
    if keep == len(groups):
        return list(items)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(groups))
    selected = {groups[int(index)] for index in order[:keep]}
    return [item for item in items if str(item.group_id) in selected]


def build_dataset(
    config: Config,
    split: str,
    *,
    train: bool = False,
    fold: int | None = None,
) -> Dataset:
    """Build the dataset declared under ``dataset.name`` for one split and fold."""
    name = str(config.require("dataset.name")).strip().lower()
    resolved_fold = int(fold if fold is not None else config.get("dataset.split.fold", 0) or 0)
    if name == "cvc_clinicdb":
        items = load_cvc_items(config, split, fold=resolved_fold)
        if train:
            fraction = float(config.get("training.data_fraction", 1.0) or 1.0)
            items = subsample_training_groups(items, fraction, int(config.require("seed")))
        augmentation = AugmentationConfig.from_config(config) if train else None
        return CVCDataset(
            items,
            image_size=config.require("dataset.image_size"),
            mask_threshold=int(config.get("dataset.mask_threshold", 127)),
            spacing=config.require("dataset.spacing_mm"),
            normalization=NormalizationSpec.from_config(config, fold=resolved_fold),
            augmentation=augmentation,
            seed=int(config.require("seed")),
            fold=resolved_fold,
            train=train,
        )
    if name == "brats2021":
        from .brats import BRATSDataset, BraTSAugmentation, load_brats_items

        items = load_brats_items(config, split, fold=resolved_fold)
        if train:
            fraction = float(config.get("training.data_fraction", 1.0) or 1.0)
            items = subsample_training_groups(items, fraction, int(config.require("seed")))
        section: Mapping[str, Any] = config.section("dataset")
        normalization: Mapping[str, Any] = section.get("normalization") or {}
        kind = str(normalization.get("kind", "percentile")).strip().lower()
        if kind not in {"percentile", "none"}:
            raise ConfigError(
                "BraTS normalization must be 'percentile' (per modality over brain "
                f"voxels) or 'none', got {kind!r}"
            )
        return BRATSDataset(
            items,
            train=train,
            patch_size=config.get("dataset.patch_size", (128, 128, 128)),
            augmentation=BraTSAugmentation.from_config(config) if train else None,
            target=str(section.get("target", "et")),
            low_percentile=float(normalization.get("low", 1.0)),
            high_percentile=float(normalization.get("high", 99.0)),
            foreground_crop_p=float(section.get("foreground_crop_p", 0.9)),
            seed=int(config.require("seed")),
            fold=resolved_fold,
        )
    raise ConfigError(
        f"Unknown dataset '{name}'. Implemented datasets: cvc_clinicdb, brats2021."
    )


def build_dataloader(
    dataset: Dataset,
    config: Config,
    *,
    shuffle: bool = False,
) -> DataLoader:
    """DataLoader with the project's determinism defaults."""
    num_workers = int(config.get("runtime.num_workers", 0))
    seed = int(config.require("seed"))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config.get("training.batch_size", 1)) if shuffle else 1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(config.get("runtime.pin_memory", True)) and torch.cuda.is_available(),
        worker_init_fn=make_worker_init_fn(seed, deterministic=bool(config.get("deterministic", True))),
        generator=generator,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

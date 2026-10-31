"""CVC-ClinicDB (2D polyp segmentation) dataset.

Images are read with OpenCV because the distributed TIFFs are not readable by
Pillow (extra samples per pixel).  Ground-truth masks are binarised after resizing
with nearest-neighbour interpolation, so no interpolation artefacts can create
label values that were never annotated.

Splits are always read from the split CSV produced by
``scripts/make_splits.py`` (video level); this module never splits frames itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from melba.config import Config

from .splits import SplitError, read_split_csv, split_file_for_fold

__all__ = [
    "CVCItem",
    "CVCDataset",
    "AUGMENTATION_GROUPS",
    "AugmentationConfig",
    "discover_cvc_items",
    "load_cvc_items",
]


@dataclass(frozen=True)
class CVCItem:
    """One frame: identifiers plus the resolved image and mask paths."""

    case_id: str
    group_id: str
    split: str
    image_path: Path
    mask_path: Path
    fold: int = 0


#: training augmentation groups (``training.train_aug``)
AUGMENTATION_GROUPS: tuple[str, ...] = (
    "geometric_d4",
    "flips_only",
    "intensity",
    "none",
)


@dataclass(frozen=True)
class AugmentationConfig:
    """Training augmentation settings, grouped so ablations stay comparable."""

    group: str = "none"
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rot90_p: float = 0.5
    brightness: float = 0.0
    contrast: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.group != "none"

    @property
    def applies_geometry(self) -> bool:
        return self.group in {"geometric_d4", "flips_only"}

    @property
    def applies_rot90(self) -> bool:
        return self.group == "geometric_d4"

    @property
    def applies_intensity(self) -> bool:
        return self.group in {"geometric_d4", "flips_only", "intensity"}

    @classmethod
    def from_config(cls, config: Config) -> "AugmentationConfig":
        section: Mapping[str, Any] = config.get("dataset.augmentation") or {}
        fields = {
            key: section[key]
            for key in ("hflip_p", "vflip_p", "rot90_p", "brightness", "contrast")
            if key in section
        }
        group = str(config.get("training.train_aug", section.get("group", "none"))).lower()
        return cls(group=group, **fields)

    def __post_init__(self) -> None:
        if self.group not in AUGMENTATION_GROUPS:
            raise ValueError(
                f"Unknown augmentation group {self.group!r}; expected one of "
                f"{list(AUGMENTATION_GROUPS)}"
            )
        for name in ("hflip_p", "vflip_p", "rot90_p"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name in ("brightness", "contrast"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")


def discover_cvc_items(config: Config) -> list[CVCItem]:
    """List every frame on disk together with its mask (no split information yet)."""
    images_dir = config.require_path("dataset.images_dir", must_exist=True)
    masks_dir = config.require_path("dataset.masks_dir", must_exist=True)
    extension = str(config.get("dataset.image_extension", ".tif")).lower()
    mask_extension = str(config.get("dataset.mask_extension", extension)).lower()

    image_paths = sorted(images_dir.glob(f"*{extension}"))
    if not image_paths:
        raise SplitError(f"No '{extension}' images found in {images_dir}")

    missing_masks = []
    items: list[CVCItem] = []
    for image_path in image_paths:
        mask_path = masks_dir / f"{image_path.stem}{mask_extension}"
        if not mask_path.is_file():
            missing_masks.append(mask_path.name)
            continue
        items.append(
            CVCItem(
                case_id=image_path.stem,
                group_id="",
                split="",
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    if missing_masks:
        preview = ", ".join(sorted(missing_masks)[:5])
        raise SplitError(
            f"{len(missing_masks)} image(s) have no ground-truth mask (first: {preview})"
        )
    return items


def load_cvc_items(config: Config, split: str, *, fold: int = 0) -> list[CVCItem]:
    """Attach video-level split information read from the (per-fold) split CSV."""
    split_file = split_file_for_fold(config, fold)
    if not split_file.is_file():
        raise SplitError(
            f"Split file not found: {split_file}. Run scripts/make_splits.py with this config."
        )
    records = read_split_csv(split_file)
    mismatched = sorted({record.fold for record in records} - {int(fold)})
    if mismatched:
        raise SplitError(
            f"{split_file} contains fold {mismatched} records but fold {fold} was requested"
        )
    assignments = {record.case_id: record for record in records}

    items: list[CVCItem] = []
    on_disk: set[str] = set()
    for item in discover_cvc_items(config):
        on_disk.add(item.case_id)
        record = assignments.get(item.case_id)
        if record is None:
            raise SplitError(
                f"Case '{item.case_id}' is on disk but has no split assignment in {split_file}. "
                "Re-run scripts/make_splits.py so the split matches the data."
            )
        if record.split == split:
            items.append(
                CVCItem(
                    case_id=item.case_id,
                    group_id=record.group_id,
                    split=record.split,
                    image_path=item.image_path,
                    mask_path=item.mask_path,
                    fold=int(fold),
                )
            )

    stale = sorted(set(assignments) - on_disk)
    if stale:
        preview = ", ".join(stale[:5])
        raise SplitError(
            f"{len(stale)} case(s) in {split_file} do not exist on disk (first: {preview}). "
            "Re-run scripts/make_splits.py."
        )
    if not items:
        raise SplitError(f"No cases assigned to split '{split}' in {split_file}")
    return items


class CVCDataset(Dataset):
    """2D frames with deterministic, per-(epoch, index) random augmentation.

    ``__getitem__`` returns a dictionary with tensors ``image`` (C,H,W in [0,1])
    and ``mask`` (1,H,W in {0,1}) plus the metadata needed for the long-format
    result records (``case_id``, ``group_id``, ``split``, ``spacing``).
    """

    def __init__(
        self,
        items: Sequence[CVCItem],
        *,
        image_size: Sequence[int] = (352, 352),
        mask_threshold: int = 127,
        spacing: Sequence[float] = (1.0, 1.0),
        augmentation: AugmentationConfig | None = None,
        seed: int = 0,
        fold: int = 0,
        train: bool = False,
    ) -> None:
        if not items:
            raise ValueError("CVCDataset needs at least one item")
        self.items = list(items)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if len(self.image_size) != 2 or min(self.image_size) < 1:
            raise ValueError(f"image_size must be two positive integers, got {image_size}")
        self.mask_threshold = int(mask_threshold)
        self.spacing = tuple(float(value) for value in spacing)
        if len(self.spacing) != 2:
            raise ValueError(f"CVC-ClinicDB is 2D; spacing needs two values, got {self.spacing}")
        self.augmentation = augmentation or AugmentationConfig()
        if self.augmentation.enabled and not train:
            # Evaluation must never be augmented; make this impossible to forget.
            raise ValueError("Augmentation can only be enabled for training datasets")
        self.seed = int(seed)
        self.fold = int(fold)
        self.train = bool(train)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation stream between epochs (still seed-deterministic)."""
        self._epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, self._epoch, int(index)])

    @staticmethod
    def _read(item: CVCItem) -> tuple[np.ndarray, np.ndarray]:
        image_bgr = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image {item.image_path}")
        mask = cv2.imread(str(item.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask {item.mask_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if mask.shape != image.shape[:2]:
            raise RuntimeError(
                f"Mask {item.mask_path} has shape {mask.shape}, image has {image.shape[:2]}"
            )
        return image, mask

    def _augment(
        self, image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        config = self.augmentation
        if config.applies_geometry:
            if config.applies_rot90 and rng.random() < config.rot90_p:
                k = int(rng.integers(1, 4))
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)
            if rng.random() < config.hflip_p:
                image = image[:, ::-1]
                mask = mask[:, ::-1]
            if rng.random() < config.vflip_p:
                image = image[::-1, :]
                mask = mask[::-1, :]

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        if config.applies_intensity and config.brightness > 0:
            factor = float(rng.uniform(1.0 - config.brightness, 1.0 + config.brightness))
            image = np.clip(image.astype(np.float32) * factor, 0, 255)
        if config.applies_intensity and config.contrast > 0:
            factor = float(rng.uniform(1.0 - config.contrast, 1.0 + config.contrast))
            mean = float(image.mean())
            image = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255)
        return image, mask

    def _resize(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image, mask = self._read(item)
        original_size = image.shape[:2]
        native_mask = (mask > self.mask_threshold).astype(np.float32)
        if self.augmentation.enabled:
            image, mask = self._augment(image, mask, self._rng(index))
        image, mask = self._resize(image, mask)

        image_float = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        mask_float = (mask > self.mask_threshold).astype(np.float32)

        sample = {
            "image": torch.from_numpy(image_float.transpose(2, 0, 1)),
            "mask": torch.from_numpy(mask_float).unsqueeze(0),
            "case_id": item.case_id,
            "group_id": item.group_id,
            "split": item.split,
            "fold": int(item.fold if item.fold else self.fold),
            "spacing": self.spacing,
            "original_size": original_size,
        }
        if not self.train:
            # Metrics are computed on the native grid: the reference mask is not
            # resized, so distance metrics are reported in the spacing declared by
            # the config (network input resolution never leaks into the numbers).
            sample["mask_native"] = torch.from_numpy(native_mask).unsqueeze(0)
        return sample


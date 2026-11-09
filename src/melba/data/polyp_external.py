"""External polyp datasets: Kvasir-SEG, CVC-ColonDB, ETIS-LaribPolypDB.

These are **never** trained on and never written to: the loader reads them where they
live, validates them, and hands frames to the same evaluation runner used for the internal
CVC-ClinicDB grid, so internal and external numbers come from identical code.

Design decisions, and why:

* **Image/mask matching is by file stem, and it fails loudly.** The datasets use different
  extensions (Kvasir-SEG ships ``.jpg`` images *and* ``.jpg`` masks, the other two
  ``.png``) and a silent mis-pairing would produce plausible but meaningless Dice.  Every
  image must find exactly one mask and every mask one image, otherwise the loader raises
  with the first five offenders and the counts.
* **RGB is validated, not assumed.** ``cv2.imread`` returns BGR, and a greyscale file read
  through OpenCV comes back with three identical channels (the exact failure mode that
  made the CVC-ClinicDB TIFFs unusable that way).  Such a frame is rejected here instead
  of being silently scored.
* **Mask binarisation is explicit.** Raw mask values are recorded per dataset;
  ``dataset.mask_threshold: null`` auto-selects ``> 0`` for a 0/1 mask and the configured
  threshold otherwise, and the choice lands in the validation report.
* **Native geometry is kept for the metrics.** Frames are resized to
  ``dataset.image_size`` (352x352) for inference exactly like the CVC training pipeline,
  while ``mask_native`` stays at the original resolution, so Dice and the distance metrics
  are computed on the native grid in the config's spacing (pixels for these datasets).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from melba.config import Config, ConfigError

from .normalization import NormalizationSpec

__all__ = [
    "DEFAULT_IMAGE_EXTENSIONS",
    "ExternalPolypDataset",
    "ExternalPolypItem",
    "binarise_mask",
    "discover_external_items",
    "read_external_case",
    "validate_external_dataset",
]

#: extensions tried when matching images to masks (the config can override)
DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

#: extensions read through tifffile (OpenCV mishandles some distributed TIFFs)
_TIFF_EXTENSIONS = {".tif", ".tiff"}


@dataclass(frozen=True)
class ExternalPolypItem:
    """One external frame: identifiers plus the resolved image and mask paths."""

    case_id: str
    group_id: str
    split: str
    image_path: Path
    mask_path: Path
    fold: int = 0


def _extensions(config: Config) -> tuple[str, ...]:
    raw = config.get("dataset.image_extensions", config.get("dataset.image_extension"))
    if raw is None:
        return DEFAULT_IMAGE_EXTENSIONS
    if isinstance(raw, str):
        raw = [raw]
    values = []
    for value in raw:
        text = str(value).strip().lower()
        values.append(text if text.startswith(".") else f".{text}")
    if not values:
        raise ConfigError("dataset.image_extensions is empty")
    return tuple(values)


def _mask_extensions(config: Config, image_extensions: Sequence[str]) -> tuple[str, ...]:
    raw = config.get("dataset.mask_extensions")
    if raw is None:
        # masks may use a different extension than the image (Kvasir-SEG: both .jpg)
        return tuple(dict.fromkeys([*image_extensions, *DEFAULT_IMAGE_EXTENSIONS]))
    if isinstance(raw, str):
        raw = [raw]
    return tuple(str(value).strip().lower() for value in raw)


def _list_files(directory: Path, extensions: Sequence[str]) -> list[Path]:
    if not directory.is_dir():
        raise ConfigError(f"External dataset directory does not exist: {directory}")
    wanted = {extension.lower() for extension in extensions}
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in wanted and not path.name.startswith(".")
    ]


def discover_external_items(config: Config, *, split: str = "test") -> list[ExternalPolypItem]:
    """Every (image, mask) pair of the external dataset declared by the config.

    Raises
    ------
    ConfigError
        When a directory is missing, when an image has no mask, when several masks compete
        for one stem, or when a mask has no image.  Silent partial matching is the one
        failure this loader must not have.
    """
    images_dir = config.require_path("dataset.images_dir", must_exist=True)
    masks_dir = config.require_path("dataset.masks_dir", must_exist=True)
    image_extensions = _extensions(config)
    mask_extensions = _mask_extensions(config, image_extensions)

    image_paths = _list_files(images_dir, image_extensions)
    if not image_paths:
        raise ConfigError(
            f"No images with extension(s) {list(image_extensions)} in {images_dir}"
        )
    mask_paths = _list_files(masks_dir, mask_extensions)
    if not mask_paths:
        raise ConfigError(f"No masks with extension(s) {list(mask_extensions)} in {masks_dir}")

    masks_by_stem: dict[str, list[Path]] = {}
    for path in mask_paths:
        masks_by_stem.setdefault(path.stem, []).append(path)

    missing: list[str] = []
    ambiguous: list[str] = []
    items: list[ExternalPolypItem] = []
    used: set[str] = set()
    for image_path in image_paths:
        candidates = masks_by_stem.get(image_path.stem, [])
        if not candidates:
            missing.append(image_path.name)
            continue
        if len(candidates) > 1:
            ambiguous.append(f"{image_path.stem} -> {[p.name for p in candidates]}")
            continue
        used.add(image_path.stem)
        items.append(
            ExternalPolypItem(
                case_id=image_path.stem,
                group_id=image_path.stem,  # no video grouping: one image is one unit
                split=split,
                image_path=image_path,
                mask_path=candidates[0],
            )
        )

    problems: list[str] = []
    if missing:
        problems.append(
            f"{len(missing)} image(s) without a mask (first: {', '.join(sorted(missing)[:5])})"
        )
    if ambiguous:
        problems.append(
            f"{len(ambiguous)} stem(s) with several candidate masks (first: "
            f"{'; '.join(ambiguous[:5])})"
        )
    orphan_masks = sorted(set(masks_by_stem) - used)
    if orphan_masks:
        problems.append(
            f"{len(orphan_masks)} mask(s) without an image (first: "
            f"{', '.join(orphan_masks[:5])})"
        )
    if problems:
        raise ConfigError(
            f"External dataset {images_dir} / {masks_dir} does not pair up: "
            + "; ".join(problems)
        )
    if not items:  # pragma: no cover - guarded above
        raise ConfigError("External dataset contains no usable image/mask pairs")
    return items


def _read_image(path: Path) -> np.ndarray:
    """Read one frame as contiguous RGB uint8, rejecting replicated grey channels."""
    if path.suffix.lower() in _TIFF_EXTENSIONS:
        data = np.asarray(tifffile.imread(path))
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ConfigError(f"cv2 could not read {path}")
        data = np.asarray(raw)
        if data.ndim == 3 and data.shape[2] >= 3:
            data = data[..., 2::-1] if data.shape[2] == 3 else data[..., [2, 1, 0, 3]]
    if data.ndim == 2 or (data.ndim == 3 and data.shape[2] == 1):
        raise ConfigError(
            f"{path} was read as a single-channel image of shape {data.shape}; external "
            "polyp frames are RGB, so a grey frame means the colour planes were lost"
        )
    if data.ndim != 3 or data.shape[2] < 3:
        raise ConfigError(f"{path} is not an (H, W, 3) image: shape {data.shape}")
    data = data[..., :3]
    if data.dtype != np.uint8:
        if np.issubdtype(data.dtype, np.integer):
            scale = float(np.iinfo(data.dtype).max)
        else:
            scale = float(np.nanmax(data)) or 1.0
        data = np.clip(np.asarray(data, dtype=np.float64) / scale * 255.0, 0.0, 255.0)
        data = data.round().astype(np.uint8)
    planes = (data[..., 0], data[..., 1], data[..., 2])
    if np.array_equal(planes[0], planes[1]) and np.array_equal(planes[1], planes[2]):
        raise ConfigError(
            f"{path} has three identical channels: the colour planes were replicated "
            "(a greyscale read), so the frame is not a usable RGB image"
        )
    return np.ascontiguousarray(data)


def _read_mask(path: Path) -> np.ndarray:
    """Read one mask as a 2-D integer array (values are *not* binarised here)."""
    if path.suffix.lower() in _TIFF_EXTENSIONS:
        data = np.asarray(tifffile.imread(path))
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ConfigError(f"cv2 could not read {path}")
        data = np.asarray(raw)
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[..., 0]
    if data.ndim == 3:
        # a colour "mask" is only acceptable when its planes agree; then use one
        if np.array_equal(data[..., 0], data[..., 1]) and np.array_equal(
            data[..., 1], data[..., 2]
        ):
            data = data[..., 0]
        else:
            raise ConfigError(
                f"{path} is a colour image with differing channels, not a single-channel mask"
            )
    if data.ndim != 2:
        raise ConfigError(f"{path} must be a single-channel mask (H, W); got {data.shape}")
    if not np.issubdtype(data.dtype, np.integer):
        if np.issubdtype(data.dtype, np.floating) and float(np.nanmax(data)) <= 1.0:
            data = (np.asarray(data) > 0.5).astype(np.uint8) * 255
        else:
            raise ConfigError(f"{path} has non-integer mask dtype {data.dtype}")
    return np.ascontiguousarray(data)


def binarise_mask(mask: np.ndarray, threshold: int | None) -> tuple[np.ndarray, int]:
    """Binarise a raw mask and return ``(binary uint8 0/1, threshold actually used)``.

    ``threshold=None`` means "decide from the values": a mask whose maximum is <= 1 is a
    label map (``> 0``), anything else uses the configured threshold (127 by default).
    """
    values = np.asarray(mask)
    if threshold is None:
        used = 0 if int(values.max()) <= 1 else 127
    else:
        used = int(threshold)
    return (values > used).astype(np.uint8), used


def read_external_case(
    item: ExternalPolypItem, *, threshold: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one case: ``(rgb uint8, raw mask, binary mask)`` at the native resolution."""
    image = _read_image(item.image_path)
    mask = _read_mask(item.mask_path)
    if mask.shape != image.shape[:2]:
        raise ConfigError(
            f"{item.mask_path} has shape {mask.shape} but {item.image_path} has "
            f"{image.shape[:2]}"
        )
    binary, _used = binarise_mask(mask, threshold)
    return image, mask, binary


def validate_external_dataset(
    config: Config, *, split: str = "test", progress: bool = False
) -> dict[str, Any]:
    """Read every case once and describe the dataset (counts, geometry, mask values).

    Never writes anything: this is the report a reader needs before trusting a number
    computed on the dataset - sizes, extensions, mask value sets, empty masks, whether the
    RGB channels are really distinct, and which binarisation threshold was applied.
    """
    items = discover_external_items(config, split=split)
    raw_threshold = config.get("dataset.mask_threshold")
    threshold = None if raw_threshold is None else int(raw_threshold)

    sizes: list[tuple[int, int]] = []
    extensions: dict[str, int] = {}
    mask_extensions: dict[str, int] = {}
    mask_value_counts: dict[str, int] = {}
    empty_masks: list[str] = []
    foreground_pixels: list[int] = []
    thresholds_used: dict[int, int] = {}
    iterator = items
    if progress:  # pragma: no cover - convenience for the CLI
        from tqdm import tqdm

        iterator = tqdm(items, desc=str(config.get("dataset.name", "external")), unit="case")
    for item in iterator:
        image, mask, binary = read_external_case(item, threshold=threshold)
        _binary, used = binarise_mask(mask, threshold)
        thresholds_used[used] = thresholds_used.get(used, 0) + 1
        sizes.append((int(image.shape[0]), int(image.shape[1])))
        extensions[item.image_path.suffix.lower()] = (
            extensions.get(item.image_path.suffix.lower(), 0) + 1
        )
        mask_extensions[item.mask_path.suffix.lower()] = (
            mask_extensions.get(item.mask_path.suffix.lower(), 0) + 1
        )
        for value in np.unique(mask).tolist():
            key = str(int(value))
            mask_value_counts[key] = mask_value_counts.get(key, 0) + 1
        area = int(binary.sum())
        foreground_pixels.append(area)
        if area == 0:
            empty_masks.append(item.case_id)

    array = np.array(sizes, dtype=np.int64)
    areas = np.array(foreground_pixels, dtype=np.int64)
    return {
        "dataset": str(config.get("dataset.name", "external")),
        "images_dir": str(config.require_path("dataset.images_dir")),
        "masks_dir": str(config.require_path("dataset.masks_dir")),
        "read_only": True,
        "n_cases": len(items),
        "image_extensions": dict(sorted(extensions.items())),
        "mask_extensions": dict(sorted(mask_extensions.items())),
        "image_height": {
            "min": int(array[:, 0].min()),
            "median": int(np.median(array[:, 0])),
            "max": int(array[:, 0].max()),
            "distinct": int(len(np.unique(array[:, 0]))),
        },
        "image_width": {
            "min": int(array[:, 1].min()),
            "median": int(np.median(array[:, 1])),
            "max": int(array[:, 1].max()),
            "distinct": int(len(np.unique(array[:, 1]))),
        },
        "rgb_channels_distinct": True,  # _read_image rejects replicated grey frames
        "mask_values": dict(sorted(mask_value_counts.items(), key=lambda kv: int(kv[0]))),
        "mask_threshold_used": {
            str(key): value for key, value in sorted(thresholds_used.items())
        },
        "n_empty_masks": len(empty_masks),
        "empty_masks": empty_masks[:50],
        "foreground_pixels": {
            "min": int(areas.min()),
            "p25": float(np.percentile(areas, 25)),
            "median": float(np.median(areas)),
            "p75": float(np.percentile(areas, 75)),
            "max": int(areas.max()),
        },
        "spacing_mm": list(config.require("dataset.spacing_mm")),
        "inference_size": list(config.require("dataset.image_size")),
    }


class ExternalPolypDataset(Dataset):
    """Evaluation-only dataset over an external polyp set (never augmented, never split).

    ``__getitem__`` mirrors :class:`melba.data.cvc.CVCDataset` in evaluation mode: the frame
    is resized to ``image_size`` and normalised like the training data, while the reference
    mask stays at the native resolution (``mask_native``), so metrics are computed there.
    ``group_id`` is the frame itself: these datasets have no video structure, so every case
    is one independent unit (and the paired statistics must not claim clustering they do
    not have).
    """

    def __init__(
        self,
        items: Sequence[ExternalPolypItem],
        *,
        image_size: Sequence[int] = (352, 352),
        mask_threshold: int | None = None,
        spacing: Sequence[float] = (1.0, 1.0),
        normalization: NormalizationSpec,
        fold: int = 0,
        split: str = "test",
    ) -> None:
        if not items:
            raise ValueError("ExternalPolypDataset needs at least one item")
        self.items = list(items)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if min(self.image_size) < 1:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.mask_threshold = None if mask_threshold is None else int(mask_threshold)
        self.spacing = tuple(float(value) for value in spacing)
        if len(self.spacing) != 2:
            raise ValueError(f"polyp data is 2D; spacing needs two values, got {self.spacing}")
        if not isinstance(normalization, NormalizationSpec):
            raise TypeError("normalization must be a NormalizationSpec")
        self.normalization = normalization
        self.fold = int(fold)
        self.split = str(split)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image, _mask, native_binary = read_external_case(item, threshold=self.mask_threshold)
        original_size = (int(image.shape[0]), int(image.shape[1]))
        height, width = self.image_size
        # exactly the CVC training pipeline: linear for the frame, nearest for the mask
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        image_float = np.ascontiguousarray(resized, dtype=np.float32) / 255.0
        image_chw = np.ascontiguousarray(image_float.transpose(2, 0, 1))
        image_array = np.ascontiguousarray(self.normalization.apply(image_chw))
        native_float = native_binary.astype(np.float32)
        return {
            "image": torch.from_numpy(image_array),
            "mask": torch.from_numpy(native_float).unsqueeze(0),
            # metrics are computed on the native grid, never on the 352x352 network input
            "mask_native": torch.from_numpy(native_float.copy()).unsqueeze(0),
            "case_id": item.case_id,
            "group_id": item.group_id,
            "split": item.split,
            "fold": int(item.fold or self.fold),
            "spacing": self.spacing,
            "normalization": self.normalization.kind,
            "original_size": original_size,
        }




"""BraTS 2021 (3D, four MR modalities) dataset: enhancing-tumour segmentation.

Layout (the official ``BraTS2021_*`` directories, one per patient)::

    <dataset.root>/BraTS2021_00000/BraTS2021_00000_flair.nii.gz
    <dataset.root>/BraTS2021_00000/BraTS2021_00000_t1.nii.gz
    <dataset.root>/BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz
    <dataset.root>/BraTS2021_00000/BraTS2021_00000_t2.nii.gz
    <dataset.root>/BraTS2021_00000/BraTS2021_00000_seg.nii.gz

Decisions, and why:

* **Target** — label 4 (enhancing tumour, ``ET_LABEL``) binarised; every other label is
  background.  The pipeline is the binary mask pipeline (one output channel).
* **Normalization** — per modality, over *brain* voxels only: the 1st/99th percentile of
  the brain intensities is used as a clip range and the clipped volume is z-scored with
  the brain mean/std; background voxels are set to 0.  Computing the percentiles over the
  brain (any modality > 0) instead of the whole array is what stops the large empty air
  region from dominating the statistics.
* **Spacing** — read from the NIfTI affine (``header.get_zooms()``, mm).  It is passed to
  the metric code with every case, so distances are reported in millimetres and never
  assume isotropic voxels.
* **Folds** — case level: every patient is one unit, and ``melba.data.splits.folded_splits``
  gives test = fold ``k`` with validation = the *next* fold (rotating), training = the
  rest, with a fixed seed.  Split CSVs are written by ``scripts/make_brats_splits.py``.
* **Training input** — a foreground-aware random ``patch_size`` crop (default 128^3): with
  probability ``foreground_crop_p`` the crop is centred on a foreground voxel, which keeps
  the tiny ET class in the batch.  Evaluation uses the *whole* volume and
  ``inference.sliding_window`` in the runner.
* **Augmentation** — ``training.train_aug``: ``flips_only`` (independent flips along the
  three axes) or ``geometric_3d`` (flips plus 90-degree rotations in the axial plane), on
  top of an optional intensity jitter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from melba.config import Config

from .splits import SplitError, read_split_csv, split_file_for_fold

__all__ = [
    "BRATS_AUGMENTATION_GROUPS",
    "ET_LABEL",
    "MODALITIES",
    "BraTSAugmentation",
    "BraTSItem",
    "BRATSDataset",
    "discover_brats_cases",
    "foreground_crop",
    "load_brats_items",
    "percentile_normalize",
    "read_brats_case",
]

#: channel order of the network input (BraTS names, fixed so a config cannot shuffle it)
MODALITIES: tuple[str, ...] = ("flair", "t1", "t1ce", "t2")
#: enhancing tumour in the BraTS 2021 label map
ET_LABEL: int = 4
#: supported targets
TARGETS: tuple[str, ...] = ("et", "whole_tumour")
#: voxels of context kept around the brain bounding box in the cache
CACHE_MARGIN: int = 4
#: default cache location (gitignored)
CACHE_DIR = "cache/brats2021_pre"
#: training augmentation groups
BRATS_AUGMENTATION_GROUPS: tuple[str, ...] = (
    "geometric_3d",
    "flips_only",
    "intensity",
    "none",
)


@dataclass(frozen=True)
class BraTSItem:
    """One patient: four modality files plus the segmentation."""

    case_id: str
    group_id: str
    split: str
    directory: Path
    fold: int = 0

    def modality_path(self, modality: str) -> Path:
        return self.directory / f"{self.case_id}_{modality}.nii.gz"

    @property
    def mask_path(self) -> Path:
        return self.directory / f"{self.case_id}_seg.nii.gz"


@dataclass(frozen=True)
class BraTSAugmentation:
    """Training augmentation, grouped so the ablations stay comparable."""

    group: str = "none"
    flip_p: float = 0.5
    rot90_p: float = 0.5
    intensity: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.group != "none"

    @property
    def applies_flips(self) -> bool:
        return self.group in {"geometric_3d", "flips_only"}

    @property
    def applies_rot90(self) -> bool:
        return self.group == "geometric_3d"

    @classmethod
    def from_config(cls, config: Config) -> "BraTSAugmentation":
        section: Mapping[str, Any] = config.get("dataset.augmentation") or {}
        group = str(config.get("training.train_aug", section.get("group", "none"))).lower()
        if group not in BRATS_AUGMENTATION_GROUPS:
            raise ValueError(
                f"Unknown training.train_aug {group!r}; expected one of "
                f"{list(BRATS_AUGMENTATION_GROUPS)}"
            )
        return cls(
            group=group,
            flip_p=float(section.get("flip_p", 0.5)),
            rot90_p=float(section.get("rot90_p", 0.5)),
            intensity=float(section.get("intensity", 0.0)),
        )


def discover_brats_cases(config: Config) -> list[str]:
    """Every patient directory under ``dataset.root`` (``BraTS2021_*`` names)."""
    root = config.resolve(str(config.require("dataset.root")))
    if not root.is_dir():
        raise SplitError(
            f"dataset.root {root} does not exist; point it at the BraTS 2021 training "
            "directory (one BraTS2021_* folder per patient)"
        )
    cases = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / f"{path.name}_seg.nii.gz").is_file()
    )
    if not cases:
        raise SplitError(f"no BraTS case found under {root} (expected BraTS2021_*_seg.nii.gz)")
    return cases


def load_brats_items(config: Config, split: str, fold: int | None = None) -> list[BraTSItem]:
    """The cases of one split, read from the fold's split CSV (never split here)."""
    resolved_fold = int(fold if fold is not None else config.get("dataset.split.fold", 0) or 0)
    path = split_file_for_fold(config, resolved_fold)
    records = read_split_csv(path)
    root = config.resolve(str(config.require("dataset.root")))
    selected = [record for record in records if record.split == split]
    if not selected:
        raise SplitError(f"{path} has no '{split}' rows")
    return [
        BraTSItem(
            case_id=str(record.case_id),
            group_id=str(record.group_id),
            split=str(record.split),
            directory=root / str(record.case_id),
            fold=resolved_fold,
        )
        for record in selected
    ]


def _read_raw_case(
    item: BraTSItem,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Raw ``(4, D, H, W)`` float32 modalities, the label map and the spacing in mm."""
    arrays: list[np.ndarray] = []
    spacing: tuple[float, float, float] | None = None
    for modality in MODALITIES:
        path = item.modality_path(modality)
        if not path.is_file():
            raise FileNotFoundError(f"missing modality file {path}")
        image = nib.load(str(path))
        data = np.asarray(image.get_fdata(dtype=np.float32))
        if spacing is None:
            zooms = image.header.get_zooms()[:3]
            spacing = tuple(float(value) for value in zooms)
        elif tuple(float(value) for value in image.header.get_zooms()[:3]) != spacing:
            raise RuntimeError(
                f"{item.case_id}: the modalities do not share a spacing "
                f"({spacing} vs {image.header.get_zooms()[:3]})"
            )
        arrays.append(data)
    stack = np.stack(arrays, axis=0)  # (4, D, H, W)
    mask_image = nib.load(str(item.mask_path))
    labels = np.asarray(mask_image.get_fdata(dtype=np.float32))
    if labels.shape != stack.shape[1:]:
        raise RuntimeError(
            f"{item.case_id}: segmentation shape {labels.shape} does not match the "
            f"modalities {stack.shape[1:]}"
        )
    return stack, labels, tuple(spacing or (1.0, 1.0, 1.0))


def read_brats_case(
    item: BraTSItem,
    *,
    target: str = "et",
    low: float = 1.0,
    high: float = 99.0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """``(image (4, D, H, W) float32, mask (D, H, W) float32, spacing mm)`` for one case.

    The four modalities are stacked in :data:`MODALITIES` order and normalized per
    modality over the brain; the mask is ``label == ET_LABEL`` for ``target="et"``.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown BraTS target {target!r}; expected one of {list(TARGETS)}")
    stack, labels, spacing = _read_raw_case(item)
    mask = (labels == float(ET_LABEL)) if target == "et" else (labels > 0)
    brain = stack.max(axis=0) > 0
    normalized = np.stack(
        [
            percentile_normalize(stack[index], brain, low=low, high=high)
            for index in range(stack.shape[0])
        ],
        axis=0,
    )
    return normalized.astype(np.float32), mask.astype(np.float32), spacing



def percentile_normalize(
    volume: np.ndarray,
    brain: np.ndarray,
    *,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """Clip to the brain's ``low``/``high`` percentiles and z-score over the brain.

    Background voxels (outside ``brain``) are set to zero, so an empty air region can
    never shift the statistics of a modality.
    """
    values = np.asarray(volume, dtype=np.float32)
    mask = np.asarray(brain, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"volume {values.shape} and brain mask {mask.shape} differ")
    if not mask.any():
        return np.zeros_like(values, dtype=np.float32)
    inside = values[mask]
    lo = float(np.percentile(inside, float(low)))
    hi = float(np.percentile(inside, float(high)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi, lo = float(inside.max()), float(inside.min())
    if hi <= lo:  # a constant volume: nothing to normalize
        return np.zeros_like(values, dtype=np.float32)
    clipped = np.clip(values, lo, hi)
    inside = clipped[mask]
    mean = float(inside.mean())
    std = float(inside.std())
    if std <= 0:
        std = 1.0
    normalized = (clipped - mean) / std
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)



def foreground_crop(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: Sequence[int],
    rng: np.random.Generator,
    *,
    foreground_p: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Random ``patch_size`` crop, centred on foreground with probability ``foreground_p``.

    ET is a few thousand voxels in a 240^3 volume, so a uniformly random crop would be
    empty almost always; centring most crops on a foreground voxel is what makes the
    training signal usable.  Volumes smaller than the patch are padded with zeros.
    """
    patch = tuple(int(value) for value in patch_size)
    if image.ndim != 4 or mask.ndim != 3:
        raise ValueError(
            f"expected (C, D, H, W) and (D, H, W), got {image.shape}/{mask.shape}"
        )
    spatial = image.shape[1:]
    if len(patch) != len(spatial):
        raise ValueError(f"patch {patch} and volume {spatial} have different ranks")
    pads: list[tuple[int, int]] = []
    for size, want in zip(spatial, patch):
        missing = max(0, want - size)
        before = missing // 2
        pads.append((before, missing - before))
    padded_image = image
    padded_mask = mask
    if any(before or after for before, after in pads):
        padded_image = np.pad(image, ((0, 0), *pads), mode="constant", constant_values=0.0)
        padded_mask = np.pad(mask, pads, mode="constant", constant_values=0.0)

    limits = [max(0, padded_image.shape[axis + 1] - patch[axis]) for axis in range(len(patch))]
    if mask.any() and rng.random() < float(foreground_p):
        candidates = np.argwhere(padded_mask > 0)
        centre = candidates[int(rng.integers(0, len(candidates)))]
    else:
        centre = np.array([rng.integers(0, size + 1) for size in limits], dtype=int)
    starts = [
        int(np.clip(int(centre[axis]) - patch[axis] // 2, 0, limits[axis]))
        for axis in range(len(patch))
    ]
    slices = tuple(
        slice(starts[axis], starts[axis] + patch[axis]) for axis in range(len(patch))
    )
    cropped_image = padded_image[(slice(None), *slices)]
    cropped_mask = padded_mask[slices]
    return np.ascontiguousarray(cropped_image), np.ascontiguousarray(cropped_mask)



class BRATSDataset(Dataset):
    """Training patches or whole evaluation volumes, with per-case spacing in mm."""

    def __init__(
        self,
        items: Sequence[BraTSItem],
        *,
        train: bool = False,
        patch_size: Sequence[int] = (128, 128, 128),
        augmentation: BraTSAugmentation | None = None,
        target: str = "et",
        low_percentile: float = 1.0,
        high_percentile: float = 99.0,
        foreground_crop_p: float = 0.9,
        seed: int = 0,
        fold: int = 0,
        use_cache: bool = False,
        cache_dir: str | Path = CACHE_DIR,
    ) -> None:
        if not items:
            raise ValueError("BRATSDataset needs at least one case")
        self.items = list(items)
        self.train = bool(train)
        self.patch_size = tuple(int(value) for value in patch_size)
        if self.train and len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have three entries, got {self.patch_size}")
        self.augmentation = augmentation or BraTSAugmentation()
        self.target = str(target)
        self.low_percentile = float(low_percentile)
        self.high_percentile = float(high_percentile)
        self.foreground_crop_p = float(foreground_crop_p)
        self.seed = int(seed)
        self.fold = int(fold)
        self.use_cache = bool(use_cache)
        self.cache_dir = Path(cache_dir)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        """Reseed the crop/augmentation stream (called once per training epoch)."""
        self.epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, self.epoch, int(index) % (2**32)])

    def _augment(
        self, image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flips along the three axes and, for ``geometric_3d``, axial 90-degree rotations.

        ``image`` is ``(C, D, H, W)`` and ``mask`` is ``(D, H, W)``, so the spatial axes of
        the image are shifted by one relative to the mask.
        """
        config = self.augmentation
        if config.applies_rot90 and rng.random() < config.rot90_p:
            k = int(rng.integers(1, 4))
            image = np.rot90(image, k, axes=(2, 3))  # (H, W) plane of the image
            mask = np.rot90(mask, k, axes=(1, 2))
        if config.applies_flips:
            for axis in range(3):
                if rng.random() < config.flip_p:
                    image = np.flip(image, axis=axis + 1)
                    mask = np.flip(mask, axis=axis)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        if config.enabled and config.intensity > 0:
            factor = float(rng.uniform(1.0 - config.intensity, 1.0 + config.intensity))
            image = image * factor
            if rng.random() < 0.5:  # shift only inside the brain, keep air at zero
                shift = float(rng.uniform(-config.intensity, config.intensity))
                image = image + shift * (image != 0)
        return image.astype(np.float32), mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image, mask, spacing, meta = load_case_for_pipeline(
            item,
            target=self.target,
            low=self.low_percentile,
            high=self.high_percentile,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
        )
        sample: dict[str, Any] = {
            "case_id": item.case_id,
            "group_id": item.group_id,
            "split": item.split,
            "fold": int(item.fold or self.fold),
            "spacing": spacing,
            "original_size": tuple(int(value) for value in meta["original_shape"]),
            "crop_offset": tuple(int(value) for value in meta["offsets"]),
            "source": meta["source"],
        }
        if self.train:
            image, mask = foreground_crop(
                image,
                mask,
                self.patch_size,
                self._rng(index),
                foreground_p=self.foreground_crop_p,
            )
            if self.augmentation.enabled:
                image, mask = self._augment(image, mask, self._rng(index))
            sample["image"] = torch.from_numpy(image)
            sample["mask"] = torch.from_numpy(mask).unsqueeze(0)
            return sample
        sample["image"] = torch.from_numpy(image)
        sample["mask"] = torch.from_numpy(mask).unsqueeze(0)
        # metrics are computed on the untouched native grid, in the case's own mm
        sample["mask_native"] = torch.from_numpy(mask).unsqueeze(0)
        return sample



# ---------------------------------------------------------------------------
# preprocessing cache: brain-cropped, normalized float16 volumes
# ---------------------------------------------------------------------------
def cache_paths(cache_dir: str | Path, case_id: str) -> tuple[Path, Path]:
    """Where the cached image and label arrays of one case live."""
    directory = Path(cache_dir)
    return directory / f"{case_id}_image.npy", directory / f"{case_id}_seg.npy"


def cache_meta_path(cache_dir: str | Path, case_id: str) -> Path:
    """Per-case crop metadata (offsets, original shape, spacing) written beside the arrays."""
    return Path(cache_dir) / f"{case_id}_meta.json"


def cache_index_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "index.json"


def read_cache_index(cache_dir: str | Path) -> dict[str, dict]:
    """The per-case crop metadata (``{}`` when the cache does not exist yet)."""
    path = cache_index_path(cache_dir)
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cases = dict(payload.get("cases") or {})
            if cases:
                return cases
        except (OSError, ValueError):  # pragma: no cover - corrupt index
            pass
    # no (or broken) summary index: rebuild it from the per-case sidecar files
    cases = {}
    for meta_path in sorted(Path(cache_dir).glob("*_meta.json")):
        try:
            entry = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # pragma: no cover - corrupt sidecar
            continue
        case_id = str(entry.get("case_id") or meta_path.stem[: -len("_meta")])
        cases[case_id] = entry
    return cases


def write_cache_index(cache_dir: str | Path, cases: Mapping[str, Mapping]) -> Path:
    path = cache_index_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cases": {key: dict(value) for key, value in sorted(cases.items())}}
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


def place_on_full_grid(
    array: np.ndarray,
    *,
    offsets: Sequence[int],
    original_shape: Sequence[int],
    fill: float = 0.0,
) -> np.ndarray:
    """Copy a cropped ``(C, d, h, w)`` array back into the original volume.

    A metric computed on the cropped grid and the same metric on this padded array are
    identical (a crop only removes voxels that are background in both masks), which is
    what makes the cache safe for evaluation.
    """
    values = np.asarray(array)
    shape = tuple(int(value) for value in original_shape)
    if values.ndim != len(shape) + 1:
        raise ValueError(
            f"expected (C, *spatial) with {len(shape)} spatial axes, got {values.shape}"
        )
    if any(int(size) > limit for size, limit in zip(values.shape[1:], shape)):
        raise ValueError(f"cropped shape {values.shape[1:]} does not fit {shape}")
    full = np.full((values.shape[0], *shape), fill, dtype=values.dtype)
    slices = tuple(
        slice(int(offset), int(offset) + int(size))
        for offset, size in zip(offsets, values.shape[1:])
    )
    full[(slice(None), *slices)] = values
    return full




def preprocess_case(
    item: BraTSItem,
    *,
    cache_dir: str | Path = CACHE_DIR,
    low: float = 1.0,
    high: float = 99.0,
    margin: int = CACHE_MARGIN,
    overwrite: bool = False,
) -> dict:
    """Crop one case to its brain bounding box, normalize and store float16/uint8 arrays.

    The crop offsets and the original shape go into the returned entry (and into the
    cache index), so a prediction made on the cropped grid can be placed back on the
    240x240x155 grid.  The original BraTS directory is only ever read.
    """
    image_path, seg_path = cache_paths(cache_dir, item.case_id)
    if not overwrite and image_path.is_file() and seg_path.is_file():
        meta_path = cache_meta_path(cache_dir, item.case_id)
        if meta_path.is_file():
            try:
                entry = json.loads(meta_path.read_text(encoding="utf-8"))
                return {**entry, "case_id": item.case_id, "skipped": True}
            except (OSError, ValueError):  # pragma: no cover - corrupt sidecar
                pass
    stack, labels, spacing = _read_raw_case(item)
    brain = stack.max(axis=0) > 0
    index = np.argwhere(brain)
    if index.size == 0:  # pragma: no cover - a completely empty volume
        raise RuntimeError(f"{item.case_id}: no brain voxels found")
    starts = np.maximum(index.min(axis=0) - int(margin), 0)
    stops = np.minimum(index.max(axis=0) + int(margin) + 1, np.array(brain.shape))
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))
    cropped_brain = brain[slices]
    normalized = np.stack(
        [
            percentile_normalize(
                stack[channel][slices], cropped_brain, low=low, high=high
            )
            for channel in range(stack.shape[0])
        ],
        axis=0,
    ).astype(np.float16)
    cropped_labels = labels[slices].astype(np.uint8)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_path, normalized)
    np.save(seg_path, cropped_labels)
    entry = {
        "case_id": item.case_id,
        "skipped": False,
        "offsets": [int(value) for value in starts],
        "original_shape": [int(value) for value in brain.shape],
        "shape": [int(value) for value in normalized.shape[1:]],
        "spacing": [float(value) for value in spacing],
        "et_voxels": int((labels == float(ET_LABEL)).sum()),
        "bytes": int(normalized.nbytes + cropped_labels.nbytes),
    }
    meta_path = cache_meta_path(cache_dir, item.case_id)
    meta_path.write_text(json.dumps(entry, indent=1) + "\n", encoding="utf-8")
    return entry


def load_cached_case(
    item: BraTSItem, *, cache_dir: str | Path = CACHE_DIR
) -> dict | None:
    """The cached arrays of one case (``None`` when it has not been preprocessed)."""
    image_path, seg_path = cache_paths(cache_dir, item.case_id)
    meta_path = cache_meta_path(cache_dir, item.case_id)
    if not (image_path.is_file() and seg_path.is_file()):
        return None
    entry: dict | None = None
    if meta_path.is_file():
        try:
            entry = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # pragma: no cover - corrupt sidecar
            entry = None
    if entry is None:  # older caches only have the summary index
        entry = read_cache_index(cache_dir).get(item.case_id)
    if entry is None:
        return None
    return {
        "image": np.load(image_path).astype(np.float32),
        "labels": np.load(seg_path),
        "offsets": tuple(int(value) for value in entry["offsets"]),
        "original_shape": tuple(int(value) for value in entry["original_shape"]),
        "spacing": tuple(float(value) for value in entry["spacing"]),
    }


def load_case_for_pipeline(
    item: BraTSItem,
    *,
    target: str = "et",
    low: float = 1.0,
    high: float = 99.0,
    use_cache: bool = False,
    cache_dir: str | Path = CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], dict]:
    """Like :func:`read_brats_case`, optionally from the brain-cropped cache.

    Returns ``(image, mask, spacing, meta)``; ``meta`` carries the crop ``offsets`` and
    ``original_shape`` so a prediction made on the cropped grid can be placed back on the
    original 240x240x155 grid.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown BraTS target {target!r}; expected one of {list(TARGETS)}")
    cached = load_cached_case(item, cache_dir=cache_dir) if use_cache else None
    if cached is not None:
        labels = cached["labels"].astype(np.float32)
        mask = (labels == float(ET_LABEL)) if target == "et" else (labels > 0)
        return (
            cached["image"],
            mask.astype(np.float32),
            cached["spacing"],
            {
                "offsets": cached["offsets"],
                "original_shape": cached["original_shape"],
                "source": "cache",
            },
        )
    image, mask, spacing = read_brats_case(item, target=target, low=low, high=high)
    shape = tuple(int(value) for value in image.shape[1:])
    return image, mask, spacing, {
        "offsets": (0, 0, 0),
        "original_shape": shape,
        "source": "nifti",
    }

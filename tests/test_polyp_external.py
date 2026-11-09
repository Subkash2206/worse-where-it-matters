"""External polyp loader: pairing, RGB/mask validation, binarisation, dataset, dispatch."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config  # noqa: E402
from melba.data import build_dataset  # noqa: E402
from melba.data.polyp_external import (  # noqa: E402
    ExternalPolypDataset,
    binarise_mask,
    discover_external_items,
    read_external_case,
    validate_external_dataset,
)

CONFIG = _REPO_ROOT / "configs" / "external" / "polyp_external.yaml"


def _write_case(
    root: Path,
    stem: str,
    *,
    size: tuple[int, int] = (40, 52),
    mask_extension: str = ".png",
    image_extension: str = ".png",
    mask_values: tuple[int, int] = (0, 255),
    grey: bool = False,
) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(*size, 3), dtype=np.uint8)
    if grey:  # replicated planes: the OpenCV greyscale failure mode
        image[..., 1] = image[..., 0]
        image[..., 2] = image[..., 0]
    cv2.imwrite(str(images / f"{stem}{image_extension}"), image)
    mask = np.full(size, mask_values[0], dtype=np.uint8)
    mask[8:20, 10:24] = mask_values[1]
    cv2.imwrite(str(masks / f"{stem}{mask_extension}"), mask)


@pytest.fixture()
def external_root(tmp_path: Path) -> Path:
    root = tmp_path / "external"
    for index in range(4):
        _write_case(root, f"case_{index}")
    return root


def _config(root: Path, **overrides):
    settings = {
        "dataset.kind": "external_polyp",
        "dataset.name": "toy_external",
        "dataset.images_dir": str(root / "images"),
        "dataset.masks_dir": str(root / "masks"),
        "dataset.mask_threshold": None,
        "dataset.image_size": [64, 64],
    }
    settings.update(overrides)
    return load_config(CONFIG, settings)


def test_discovery_pairs_images_and_masks(external_root: Path) -> None:
    items = discover_external_items(_config(external_root))
    assert [item.case_id for item in items] == [f"case_{index}" for index in range(4)]
    # one image is one unit: there is no video structure to cluster on
    assert all(item.group_id == item.case_id for item in items)


def test_discovery_fails_loudly_on_unpaired_files(external_root: Path) -> None:
    (external_root / "masks" / "orphan.png").write_bytes(
        (external_root / "masks" / "case_0.png").read_bytes()
    )
    with pytest.raises(ConfigError, match="does not pair up"):
        discover_external_items(_config(external_root))

    (external_root / "masks" / "orphan.png").unlink()
    (external_root / "images" / "unmatched.png").write_bytes(
        (external_root / "images" / "case_0.png").read_bytes()
    )
    with pytest.raises(ConfigError, match="does not pair up"):
        discover_external_items(_config(external_root))


def test_discovery_supports_mixed_extensions(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    _write_case(root, "jpg_case", image_extension=".jpg", mask_extension=".jpg")
    _write_case(root, "png_case", image_extension=".png", mask_extension=".png")
    items = discover_external_items(_config(root))
    assert {item.case_id for item in items} == {"jpg_case", "png_case"}
    assert {item.image_path.suffix for item in items} == {".jpg", ".png"}


def test_grey_replicated_frames_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "grey"
    _write_case(root, "grey_case", grey=True)
    item = discover_external_items(_config(root))[0]
    with pytest.raises(ConfigError, match="identical channels"):
        read_external_case(item)


def test_mask_binarisation_handles_label_and_uint8_masks() -> None:
    labels = np.zeros((4, 4), dtype=np.uint8)
    labels[1:3, 1:3] = 1
    binary, used = binarise_mask(labels, None)
    assert used == 0 and binary.sum() == 4
    uint8 = (labels * 255).astype(np.uint8)
    binary, used = binarise_mask(uint8, None)
    assert used == 127 and binary.sum() == 4
    binary, used = binarise_mask(uint8, 200)
    assert used == 200 and binary.sum() == 4


def test_read_case_rejects_mask_shape_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "mismatch"
    _write_case(root, "case")
    mask = np.zeros((10, 10), dtype=np.uint8)
    cv2.imwrite(str(root / "masks" / "case.png"), mask)
    item = discover_external_items(_config(root))[0]
    with pytest.raises(ConfigError, match="shape"):
        read_external_case(item)


def test_validation_report_describes_the_dataset(external_root: Path) -> None:
    report = validate_external_dataset(_config(external_root))
    assert report["n_cases"] == 4
    assert report["image_extensions"] == {".png": 4}
    assert report["image_height"]["min"] == 40 and report["image_width"]["max"] == 52
    assert sorted(report["mask_values"]) == ["0", "255"]
    assert report["mask_threshold_used"] == {"127": 4}
    assert report["n_empty_masks"] == 0
    assert report["rgb_channels_distinct"] is True
    assert report["foreground_pixels"]["median"] == 12 * 14


def test_dataset_resizes_input_but_keeps_native_mask(external_root: Path) -> None:
    config = _config(external_root)
    dataset = build_dataset(config, "test", fold=0)
    assert len(dataset) == 4
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 64, 64)  # network input
    assert tuple(sample["mask_native"].shape) == (1, 40, 52)  # metrics grid
    assert sample["mask_native"].sum() == 12 * 14
    assert sample["original_size"] == (40, 52)
    assert sample["spacing"] == (1.0, 1.0)
    assert sample["split"] == "test"


def test_external_polyp_dataset_refuses_training(external_root: Path) -> None:
    config = _config(external_root)
    with pytest.raises(ConfigError, match="evaluation-only"):
        build_dataset(config, "train", train=True, fold=0)


def test_dataset_requires_a_normalization_spec(external_root: Path) -> None:
    items = discover_external_items(_config(external_root))
    with pytest.raises(TypeError, match="NormalizationSpec"):
        ExternalPolypDataset(items, normalization="imagenet")  # type: ignore[arg-type]


def test_empty_mask_is_reported_not_hidden(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    _write_case(root, "empty_case", mask_values=(0, 0))
    report = validate_external_dataset(_config(root))
    assert report["n_empty_masks"] == 1
    assert report["empty_masks"] == ["empty_case"]
    assert report["mask_values"] == {"0": 1}


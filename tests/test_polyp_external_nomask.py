"""Reference-free external datasets (HyperKvasir cecum): no masks, empty references.

The polyp-free cohort is the one dataset whose *truth* is "nothing is there", so these tests
pin the two things that could silently make it lie: the loader must build an empty reference
at the frame's own resolution instead of pairing a mask, and the metric/report path must
answer with the reference-free descriptors rather than inventing a Dice on an empty mask.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config  # noqa: E402
from melba.data import build_dataset  # noqa: E402
from melba.data.polyp_external import (  # noqa: E402
    discover_external_items,
    read_external_case,
    validate_external_dataset,
)
from melba.eval.external import external_tables  # noqa: E402
from melba.metrics import MetricSettings, compute_case_metrics  # noqa: E402

CONFIG = _REPO_ROOT / "configs" / "external" / "hyperkvasir_cecum.yaml"
PRESENCE = ("pred_empty", "pred_area_px", "max_prob")


def _write_images(root: Path, count: int = 3, *, size: tuple[int, int] = (40, 52)) -> None:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        frame = rng.integers(0, 255, size=(*size, 3), dtype=np.uint8)
        cv2.imwrite(str(images / f"frame_{index}.jpg"), frame)


@pytest.fixture()
def reference_free_root(tmp_path: Path) -> Path:
    root = tmp_path / "hyperkvasir_cecum"
    _write_images(root)
    return root


def _config(root: Path, **overrides):
    settings = {
        "dataset.name": "toy_reference_free",
        "dataset.images_dir": str(root / "images"),
        "dataset.masks_dir": None,
        "dataset.references": "empty",
        "dataset.image_size": [64, 64],
    }
    settings.update(overrides)
    return load_config(CONFIG, settings)


def test_discovery_lists_images_without_masks(reference_free_root: Path) -> None:
    items = discover_external_items(_config(reference_free_root))
    assert [item.case_id for item in items] == ["frame_0", "frame_1", "frame_2"]
    # no mask file at all: the reference is built in memory, one image is one unit
    assert all(item.mask_path is None for item in items)
    assert all(item.group_id == item.case_id for item in items)


def test_references_empty_refuses_a_mask_directory(reference_free_root: Path) -> None:
    config = _config(
        reference_free_root, **{"dataset.masks_dir": str(reference_free_root / "masks")}
    )
    with pytest.raises(ConfigError, match="must not point at masks"):
        discover_external_items(config)


def test_case_has_an_empty_reference_at_the_native_resolution(
    reference_free_root: Path,
) -> None:
    item = discover_external_items(_config(reference_free_root))[0]
    image, mask, binary = read_external_case(item)
    assert image.shape == (40, 52, 3)
    assert mask.shape == image.shape[:2] and binary.shape == image.shape[:2]
    assert int(mask.sum()) == 0 and int(binary.sum()) == 0


def test_dataset_keeps_the_native_empty_reference(reference_free_root: Path) -> None:
    dataset = build_dataset(_config(reference_free_root), "test", fold=0)
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 64, 64)  # network input
    assert tuple(sample["mask_native"].shape) == (1, 40, 52)  # metrics grid
    assert float(sample["mask_native"].sum()) == 0.0


def test_validation_reports_every_reference_as_empty(reference_free_root: Path) -> None:
    report = validate_external_dataset(_config(reference_free_root))
    assert report["n_cases"] == 3
    assert report["references"] == "empty"
    assert report["masks_dir"] is None
    assert report["n_empty_masks"] == 3
    assert report["mask_extensions"] == {}
    # nothing is claimed about a mask that does not exist
    assert report["mask_threshold_used"] == {}


def test_presence_metrics_describe_the_prediction_only() -> None:
    settings = MetricSettings(names=PRESENCE)
    empty = np.zeros((8, 8), dtype=np.float32)

    quiet = np.full((8, 8), 0.2, dtype=np.float32)
    result = compute_case_metrics(quiet, empty, spacing=(1.0, 1.0), settings=settings)
    assert result["empty_flag"] == "both_empty"
    assert result["pred_empty"] == 1.0
    assert result["pred_area_px"] == 0.0
    assert result["max_prob"] == pytest.approx(0.2, abs=1e-6)

    loud = np.zeros((8, 8), dtype=np.float32)
    loud[2:4, 3:6] = 0.9  # 6 pixels of invented polyp
    result = compute_case_metrics(loud, empty, spacing=(1.0, 1.0), settings=settings)
    assert result["empty_flag"] == "gt_empty"
    assert result["pred_empty"] == 0.0
    assert result["pred_area_px"] == 6.0
    assert result["max_prob"] == pytest.approx(0.9, abs=1e-6)


def _long_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in ("m0", "m1"):
        for case in ("c1", "c2", "c3"):
            for method in ("baseline", "tta4", "tta8"):
                alarm = 0.0 if method == "baseline" else 1.0
                for metric, value in (
                    ("pred_empty", 1.0 - alarm),
                    ("pred_area_px", 120.0 * alarm),
                    ("max_prob", 0.4 + 0.2 * alarm),
                ):
                    rows.append(
                        {
                            "run_id": model,
                            "case_id": case,
                            "group_id": case,
                            "method": method,
                            "metric": metric,
                            "value": value,
                            "empty_flag": "gt_empty" if alarm else "both_empty",
                            "missed_detection": 0,
                        }
                    )
    return pd.DataFrame(rows)


def test_external_tables_drop_the_dice_only_tables() -> None:
    """A reference-free cohort must not get a challenge-Dice or paired quality table."""
    frame = _long_frame()
    tables = external_tables(
        {"m0": frame, "m1": frame},
        metrics=PRESENCE,
        treatments=("tta4", "tta8"),
        paired_metrics=(),
        n_bootstrap=200,
    )
    assert "external_challenge_scoring" not in tables
    assert "external_paired" not in tables
    false_positive = tables["external_false_positive"]
    assert {"baseline", "tta4", "tta8"} <= set(false_positive["method"])
    # the per-checkpoint spread covers the presence metrics, not an absent Dice
    assert set(tables["external_model_spread"]["metric"]) == set(PRESENCE)


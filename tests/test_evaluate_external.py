"""The external-evaluation script: config plumbing and pooled-frame discovery."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in ("src", "scripts"):
    if str(_REPO_ROOT / extra) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / extra))

import evaluate_external as script  # noqa: E402
from melba.config import load_config  # noqa: E402
from melba.eval import per_case_path  # noqa: E402

CONFIG = _REPO_ROOT / "configs" / "external" / "polyp_external.yaml"
MODEL = _REPO_ROOT / "configs" / "cvc_clinicdb" / "unet_resnet50.yaml"


def test_external_config_without_a_dataset_entry_uses_the_yaml_paths(tmp_path) -> None:
    """Regression: _frames_by_model passed entry={} and crashed with KeyError 'images_dir'."""
    external = load_config(CONFIG)
    config = script._external_config(
        external,
        MODEL,
        dataset_name="kvasir_seg",
        checkpoint=Path("unused.pt"),
        fold=2,
        seed=1,
        output_dir=tmp_path,
        save_maps=False,
        limit=None,
    )
    # directories come from the YAML, so the output path can be built without any images
    assert config.require("dataset.images_dir")
    assert config.require("dataset.masks_dir")
    assert config.require("dataset.name") == "kvasir_seg"
    assert config.require("evaluation.fold") == 2
    assert int(config.require("seed")) == 1
    path = per_case_path(config, 2, 1, output_dir=tmp_path)
    assert path.name == "kvasir_seg_test_fold2_seed1_per_case.csv"


def test_external_config_prefers_the_dataset_entry(tmp_path) -> None:
    external = load_config(CONFIG)
    config = script._external_config(
        external,
        MODEL,
        dataset_name="cvc_colondb",
        entry={
            "images_dir": str(tmp_path / "img"),
            "masks_dir": str(tmp_path / "msk"),
        },
        checkpoint=Path("unused.pt"),
        fold=0,
        seed=0,
        output_dir=tmp_path,
        save_maps=False,
        limit=None,
    )
    assert config.require_path("dataset.images_dir") == (tmp_path / "img").resolve()
    assert config.require_path("dataset.masks_dir") == (tmp_path / "msk").resolve()


def test_frames_by_model_reads_what_exists_and_skips_the_rest(tmp_path) -> None:
    """A finished run is picked up, a missing one is skipped instead of raising."""
    external = load_config(CONFIG)
    dataset = "kvasir_seg"
    evaluation = tmp_path / "evaluation"
    steps = []
    for fold, seed in ((0, 0), (0, 1)):
        steps.append(
            {
                "dataset": dataset,
                "model_path": MODEL,
                "model": "cvc_clinicdb/unet_resnet50",
                "fold": fold,
                "seed": seed,
                "checkpoint": Path("unused.pt"),
            }
        )
    config = script._external_config(
        external,
        MODEL,
        dataset_name=dataset,
        checkpoint=Path("unused.pt"),
        fold=0,
        seed=0,
        output_dir=evaluation,
        save_maps=False,
        limit=None,
    )
    present = per_case_path(config, 0, 0, output_dir=evaluation)
    present.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "case_id": "case_1",
                "group_id": "case_1",
                "method": "baseline",
                "metric": "dice",
                "value": 0.5,
                "split": "test",
                "fold": 0,
                "seed": 0,
                "empty_flag": "both_nonempty",
                "missed_detection": 0,
            }
        ]
    ).to_csv(present, index=False)

    frames = script._frames_by_model(external, steps, dataset, evaluation)
    assert list(frames) == ["cvc_clinicdb/unet_resnet50_fold0_seed0"]
    assert frames["cvc_clinicdb/unet_resnet50_fold0_seed0"]["case_id"].tolist() == ["case_1"]

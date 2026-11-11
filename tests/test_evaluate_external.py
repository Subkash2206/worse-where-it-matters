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
    """Complete runs are picked up (new layout first, then the pre-split one), the rest skipped."""
    external = load_config(CONFIG)
    dataset = "kvasir_seg"
    dataset_dir = tmp_path  # the dataset root: <root>/<model>/evaluation and <root>/evaluation
    steps = []
    for fold, seed in ((0, 0), (0, 1), (0, 2)):
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
    methods = ["baseline", "tta2", "tta4", "tta8"]
    common = {
        "dataset": dataset,
        "split": "test",
        "seed": 0,
        "group_id": "case_1",
        "value": 0.5,
        "empty_flag": "both_nonempty",
        "missed_detection": 0,
        "config_path": str(MODEL),
    }

    def write(path: Path, *, folds: tuple[int, ...] = (0,), rows: int = 1) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {**common, "case_id": "case_1", "fold": fold, "seed": seed, "method": method,
             "metric": "dice"}
            for fold, seed in ((0, 0),)
            for method in methods
        ][:rows]
        pd.DataFrame(records).to_csv(path, index=False)

    # (a) the pre-split layout: <root>/evaluation
    write(dataset_dir / "evaluation" / f"{dataset}_test_fold0_seed0_per_case.csv", rows=len(methods))
    # (b) the per-model layout: <root>/<model>/evaluation
    write(
        dataset_dir / MODEL.stem / "evaluation" / f"{dataset}_test_fold0_seed1_per_case.csv",
        rows=len(methods),
    )
    # (c) a truncated run: only the first two methods made it
    write(
        dataset_dir / MODEL.stem / "evaluation" / f"{dataset}_test_fold0_seed2_per_case.csv", rows=2
    )

    frames = script._frames_by_model(
        external, steps, dataset, dataset_dir, n_cases=1, expected_methods=methods
    )
    assert sorted(frames) == [
        "cvc_clinicdb/unet_resnet50_fold0_seed0",
        "cvc_clinicdb/unet_resnet50_fold0_seed1",
    ]
    assert frames["cvc_clinicdb/unet_resnet50_fold0_seed0"]["case_id"].tolist() == ["case_1"] * len(
        methods
    )


def test_per_case_complete_reports_why_a_run_is_incomplete(tmp_path) -> None:
    """A killed run leaves a short CSV; the resume rule must not trust its existence."""
    path = tmp_path / "kvasir_seg_test_fold0_seed0_per_case.csv"
    pd.DataFrame(
        {
            "case_id": ["a", "a", "b", "b"],
            "method": ["baseline", "tta4", "baseline", "tta4"],
            "config_path": [str(MODEL)] * 4,
        }
    ).to_csv(path, index=False)
    assert script.per_case_complete(path, n_cases=2, methods=["baseline", "tta4"])[0]
    complete, reason = script.per_case_complete(
        path, n_cases=3, methods=["baseline", "tta4"]
    )
    assert not complete and "case coverage" in reason
    complete, reason = script.per_case_complete(
        path, n_cases=2, methods=["baseline", "tta4", "tta8"]
    )
    assert not complete and "missing method" in reason
    # the pre-split fallback only accepts a file the requested architecture produced
    complete, reason = script.per_case_complete(
        path, n_cases=2, methods=["baseline", "tta4"], model_stem="swinunetr_2d"
    )
    assert not complete and "produced by" in reason

"""CLI wiring: every script exposes ``--help`` and ``make_splits`` works end-to-end."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["make_splits.py", "train.py", "evaluate.py", "run_stats.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_expose_help(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def _make_splits_args(
    toy_cvc: dict, output_pattern: Path, extra: list[str] | None = None
) -> list[str]:
    args = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "make_splits.py"),
        "--config",
        str(REPO_ROOT / "configs" / "cvc_clinicdb" / "unet_resnet50.yaml"),
        "--set",
        f"dataset.images_dir={toy_cvc['images_dir']}",
        f"dataset.masks_dir={toy_cvc['masks_dir']}",
        f"dataset.split.group_map_file={toy_cvc['group_map']}",
        f"dataset.split.output_pattern={output_pattern}",
        "dataset.split.n_folds=5",
        "dataset.split.seed=11",
    ]
    return args + (extra or [])


def test_make_splits_cli_writes_leak_free_folds(toy_cvc: dict, tmp_path: Path) -> None:
    pattern = tmp_path / "cli_fold{fold}.csv"
    result = subprocess.run(
        _make_splits_args(toy_cvc, pattern, ["--force"]),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    fold_files = sorted(tmp_path.glob("cli_fold*.csv"))
    assert len(fold_files) == 5
    assert (tmp_path / "cvc_clinicdb_folds.json").is_file()

    tested_groups: dict[str, set[int]] = {}
    for fold, path in enumerate(fold_files):
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["split"] for row in rows} == {"train", "val", "test"}
        assert {row["fold"] for row in rows} == {str(fold)}
        groups: dict[str, set[str]] = {}
        for row in rows:
            groups.setdefault(row["group_id"], set()).add(row["split"])
        assert all(len(splits) == 1 for splits in groups.values())
        for row in rows:
            if row["split"] == "test":
                tested_groups.setdefault(row["group_id"], set()).add(fold)

    # every video is a test unit in exactly one fold
    assert len(tested_groups) == 6
    assert all(len(folds) == 1 for folds in tested_groups.values())

    # without --force the script must refuse to clobber existing folds
    second = subprocess.run(
        _make_splits_args(toy_cvc, pattern), capture_output=True, text=True, timeout=300
    )
    assert second.returncode == 2
    assert "already exists" in (second.stdout + second.stderr)


def test_make_splits_cli_reports_missing_group_map(toy_cvc: dict, tmp_path: Path) -> None:
    output = tmp_path / "split.csv"
    args = _make_splits_args(toy_cvc, output)
    args = [
        f"dataset.split.group_map_file={tmp_path / 'missing_groups.csv'}"
        if arg.startswith("dataset.split.group_map_file=")
        else arg
        for arg in args
    ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=180)
    assert result.returncode == 2
    assert "Group mapping CSV not found" in (result.stdout + result.stderr)

"""Grid robustness: per-run child processes, retries and failure isolation."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from melba.training import (
    EXIT_CUDA,
    EXIT_OTHER,
    MAX_RETRIES,
    execute_with_retries,
    is_cuda_error,
    is_cuda_failure,
)


def test_retries_until_success_and_cleans_up_between_attempts() -> None:
    codes = iter([EXIT_CUDA, EXIT_CUDA, 0])
    calls: list[tuple[str, ...]] = []
    sleeps: list[float] = []
    cleanups: list[bool] = []

    def runner(command):
        calls.append(tuple(command))
        return next(codes)

    result = execute_with_retries(
        ["python", "run_one.py"],
        runner=runner,
        sleep=sleeps.append,
        cleanup=lambda: cleanups.append(True),
        retry_sleep=30,
        log=lambda message: None,
    )

    assert result.success is True
    assert result.attempts == 3
    assert result.returncodes == (EXIT_CUDA, EXIT_CUDA, 0)
    assert len(calls) == 3
    assert sleeps == [30, 30]  # GPU left idle between attempts
    assert len(cleanups) == 2


def test_exhausted_retries_are_reported_not_raised() -> None:
    result = execute_with_retries(
        ["python", "run_one.py"],
        runner=lambda command: EXIT_CUDA,
        sleep=lambda seconds: None,
        retry_sleep=0,
        log=lambda message: None,
    )
    assert result.success is False
    assert result.attempts == 1 + MAX_RETRIES
    assert set(result.returncodes) == {EXIT_CUDA}
    assert MAX_RETRIES == 3


def test_is_cuda_failure() -> None:
    assert is_cuda_failure(EXIT_CUDA) is True
    assert is_cuda_failure(EXIT_OTHER) is False
    assert is_cuda_failure(0) is False


def test_is_cuda_error_classification() -> None:
    assert is_cuda_error(RuntimeError("cuDNN error: CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH"))
    assert is_cuda_error(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert is_cuda_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_cuda_error(RuntimeError("CUDA driver version is insufficient"))
    # a path or message that merely contains the letters "cuda" is not a CUDA failure
    assert is_cuda_error(FileExistsError(r"C:\temp\test_non_cuda_failures\blocked")) is False
    assert is_cuda_error(ValueError("bad config: unknown key")) is False


@pytest.fixture
def run_one_config(repo_root, toy_cvc, tmp_path):
    """A file-based toy config (run_one.py takes a path, like the real grid)."""
    payload = {
        "defaults": [str(repo_root / "configs" / "cvc_clinicdb" / "unet_resnet50.yaml")],
        "seed": 0,
        "device": "cpu",
        "deterministic": False,
        "runtime": {"num_workers": 0, "progress": False},
        "experiment": {"name": "toy_run_one"},
        "dataset": {
            "images_dir": str(toy_cvc["images_dir"]),
            "masks_dir": str(toy_cvc["masks_dir"]),
            "image_size": [32, 32],
            "augmentation": {"group": "none"},
            "split": {
                "group_map_file": str(toy_cvc["group_map"]),
                "output_pattern": toy_cvc["split_pattern"],
                "fold": 0,
            },
        },
        "model": {
            "encoder": "resnet18",
            "encoder_weights": None,
            "checkpoint": str(
                tmp_path / "checkpoints" / "toy_run_one" / "fold{fold}_seed{seed}" / "best.pt"
            ),
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "train_aug": "none",
            "checkpoint_dir": str(tmp_path / "checkpoints" / "toy_run_one"),
        },
        "evaluation": {
            "output_dir": str(tmp_path / "results"),
            "timing": False,
            "save_probability_maps": False,
        },
    }
    path = tmp_path / "toy_run_one.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _run_one(repo_root, config_path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_one.py"),
            "--config",
            str(config_path),
            "--fold",
            "0",
            "--seed",
            "0",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_run_one_trains_then_reports_nothing_to_do(repo_root, run_one_config, tmp_path) -> None:
    first = _run_one(repo_root, run_one_config)
    assert first.returncode == 0, first.stderr
    assert "trained 1 epoch(s)" in first.stdout

    run_dir = tmp_path / "checkpoints" / "toy_run_one" / "fold0_seed0"
    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "run_done.json").is_file()
    marker = tmp_path / "results" / "cvc_clinicdb_test_fold0_seed0_complete.json"
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["seed"] == 0

    second = _run_one(repo_root, run_one_config)
    assert second.returncode == 0, second.stderr
    assert "nothing to do" in second.stdout


def test_run_one_maps_non_cuda_failures_to_exit_4(repo_root, run_one_config, tmp_path) -> None:
    # a *file* where the checkpoint directory should be makes run_dir_for fail
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    failed = _run_one(
        repo_root, run_one_config, "--set", f"training.checkpoint_dir={blocker}"
    )
    assert failed.returncode == EXIT_OTHER, failed.stdout + failed.stderr
    assert failed.returncode != EXIT_CUDA


def test_run_grid_status_only_exits_zero(repo_root, run_one_config) -> None:
    status = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_grid.py"),
            "--config",
            str(run_one_config),
            "--folds",
            "0",
            "--seeds",
            "0",
            "--status-only",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert status.returncode == 0, status.stderr
    assert "fold 0 seed 0:" in status.stdout

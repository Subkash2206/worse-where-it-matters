"""Training-loop guarantees: atomic checkpoints, exact resume and grid state."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from melba.training import (
    STATUS_EVALUATE,
    STATUS_RESUME,
    STATUS_SKIP,
    STATUS_TRAIN,
    atomic_torch_save,
    capture_rng_state,
    eval_marker_path,
    expected_eval_paths,
    iter_runs,
    restore_rng_state,
    run_status,
    train,
    write_eval_marker,
)
from melba.utils import seed_everything


@pytest.fixture
def train_config(toy_dataset_config, tmp_path):
    return toy_dataset_config.with_overrides(
        {
            "device": "cpu",
            "model.encoder": "resnet18",
            "model.encoder_weights": None,
            "training.epochs": 1,
            "training.batch_size": 2,
            "dataset.image_size": [32, 32],
            "training.checkpoint_dir": str(tmp_path / "checkpoints"),
            "evaluation.timing": False,
            "seed": 0,
        }
    )


def test_atomic_save_is_readable_and_leaves_no_temp_file(tmp_path) -> None:
    target = tmp_path / "nested" / "last.pt"
    payload = {"epoch": 3, "tensor": torch.zeros(2), "rng": capture_rng_state()}
    atomic_torch_save(payload, target)

    assert target.is_file()
    assert not list(target.parent.glob("*.tmp"))
    loaded = torch.load(target, weights_only=True)  # no pickled numpy objects inside
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["tensor"], payload["tensor"])


def test_rng_state_round_trip() -> None:
    seed_everything(0)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(1)))

    seed_everything(1234)  # move every RNG somewhere else
    restore_rng_state(state)

    assert random.random() == pytest.approx(expected[0], abs=0.0)
    assert float(np.random.random()) == pytest.approx(expected[1], abs=0.0)
    assert float(torch.rand(1)) == pytest.approx(expected[2], abs=0.0)


def test_restore_rng_state_accepts_tensors_on_a_gpu() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU-only test machines
        pytest.skip("CUDA not available")
    seed_everything(0)
    state = capture_rng_state()
    expected = (float(torch.rand(1)), random.random())
    # torch.load(map_location="cuda") moves every tensor in the payload, RNG included
    moved = {
        **state,
        "numpy_keys": state["numpy_keys"].cuda(),
        "torch": state["torch"].cuda(),
        "cuda": [tensor.cuda() for tensor in state["cuda"]],
    }
    seed_everything(7)
    restore_rng_state(moved)
    assert float(torch.rand(1)) == pytest.approx(expected[0], abs=0.0)
    assert random.random() == pytest.approx(expected[1], abs=0.0)


def test_training_writes_last_pt_and_resumes(train_config) -> None:
    first = train(train_config, fold=0, seed=0, epochs=1, resume=True, progress=False)

    assert first.last_checkpoint.is_file()
    assert first.checkpoint.is_file()
    assert first.done_path.is_file()
    assert not list(first.run_dir.glob("*.tmp"))
    state = torch.load(first.last_checkpoint, map_location="cpu", weights_only=True)
    assert state["epoch"] == 1
    assert {"optimizer_state", "scheduler_state", "scaler_state", "rng", "best_value"} <= set(
        state
    )
    assert {"python", "numpy_keys", "torch"} <= set(state["rng"])

    marker = json.loads(first.done_path.read_text(encoding="utf-8"))
    assert marker["fold"] == 0 and marker["seed"] == 0
    assert marker["epochs_run"] == 1
    assert marker["median_epoch_seconds"] > 0

    # A second call continues from epoch 1 instead of restarting
    second = train(train_config, fold=0, seed=0, epochs=2, resume=True, progress=False)
    assert second.resumed_from_epoch == 1
    assert second.epochs_run == 2
    resumed = torch.load(second.last_checkpoint, map_location="cpu", weights_only=True)
    assert resumed["epoch"] == 2

    fresh = train(train_config, fold=0, seed=0, epochs=1, resume=False, progress=False)
    assert fresh.resumed_from_epoch == 0


def test_run_directories_are_per_fold_and_seed(train_config) -> None:
    one = train(train_config, fold=0, seed=0, epochs=1, resume=False, progress=False)
    other_config = train_config.with_overrides({"seed": 2, "dataset.split.fold": 1})
    two = train(other_config, fold=1, seed=2, epochs=1, resume=False, progress=False)
    assert one.run_dir != two.run_dir
    assert one.run_dir.name == "fold0_seed0"
    assert two.run_dir.name == "fold1_seed2"


def test_train_rejects_a_seed_mismatch(train_config) -> None:
    from melba.config import ConfigError

    with pytest.raises(ConfigError, match="seed"):
        train(train_config, fold=0, seed=train_config["seed"] + 1, epochs=1, progress=False)


def test_grid_status_classification(tmp_path) -> None:
    run_dir = tmp_path / "fold0_seed0"
    run_dir.mkdir()
    per_case = tmp_path / "results" / "cvc_clinicdb_test_fold0_seed0_per_case.csv"
    marker = eval_marker_path(per_case)
    assert marker.name == "cvc_clinicdb_test_fold0_seed0_complete.json"

    assert run_status(run_dir, marker) == STATUS_TRAIN
    (run_dir / "last.pt").write_bytes(b"")
    assert run_status(run_dir, marker) == STATUS_RESUME
    (run_dir / "run_done.json").write_text("{}", encoding="utf-8")
    assert run_status(run_dir, marker) == STATUS_EVALUATE
    per_case.parent.mkdir(parents=True)
    write_eval_marker(marker, fold=0, seed=0, per_case_csv=per_case)
    assert run_status(run_dir, marker) == STATUS_SKIP
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["fold"] == 0 and payload["seed"] == 0
    assert payload["per_case_csv"] == str(per_case)


def test_grid_paths_and_order(toy_dataset_config) -> None:
    config = toy_dataset_config.with_overrides({"evaluation.split": "test"})
    per_case, metadata = expected_eval_paths(config, 2, 1)
    assert per_case.name == "cvc_clinicdb_test_fold2_seed1_per_case.csv"
    assert metadata.name == "cvc_clinicdb_test_fold2_seed1_run.json"
    assert iter_runs([0, 1], [0, 2]) == [(0, 0), (0, 2), (1, 0), (1, 2)]

"""Status report: log parsing, ETA arithmetic and the pid-probe helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import status  # noqa: E402
from melba.pidfile import (  # noqa: E402
    LAUNCHER_PROCESS_NAMES,
    pid_alive,
    process_name,
    read_pid_file,
)


def test_format_duration_is_compact() -> None:
    assert status.format_duration(None) == "?"
    assert status.format_duration(-1) == "?"
    assert status.format_duration(0) == "0s"
    assert status.format_duration(45) == "45s"
    assert status.format_duration(90) == "1m 30s"
    assert status.format_duration(3600) == "1h 00m"
    assert status.format_duration(3725) == "1h 02m"


def test_read_pid_file_tolerates_missing_and_broken_files(tmp_path) -> None:
    assert read_pid_file(tmp_path / "missing.pid") == 0
    broken = tmp_path / "broken.pid"
    broken.write_text("not-a-pid", encoding="utf-8")
    assert read_pid_file(broken) == 0
    good = tmp_path / "good.pid"
    good.write_text("4242\n", encoding="utf-8")
    assert read_pid_file(good) == 4242


def test_pid_probe_accepts_the_current_process_only() -> None:
    me = os.getpid()
    assert pid_alive(me)
    assert pid_alive(me, names=LAUNCHER_PROCESS_NAMES)
    assert process_name(me) in {"python", "py", "python3"}
    assert not pid_alive(0)
    assert not pid_alive(-1)
    # a free pid, and a wrong image name, must both be rejected
    assert not pid_alive(999_999, names=LAUNCHER_PROCESS_NAMES)
    assert not pid_alive(me, names=("definitely-not-python",))
    assert process_name(0) is None


TRAINING_LOG = """epoch,train_loss,val_loss,val_dice,learning_rate,monitor,improved,epochs_without_improvement,train_seconds,val_seconds,epoch_seconds
1,0.4,0.3,0.62,1e-04,val_dice,True,0,100.0,60.0,160.0
2,0.3,0.28,0.70,1e-04,val_dice,True,0,10.0,5.0,15.0
3,0.2,0.28,0.68,1e-04,val_dice,False,1,10.0,5.0,15.0
4,0.1,0.2,0.6,1e-04,val_dice,False,2,10.0,5.0
"""


def test_read_training_log_skips_the_truncated_last_line_and_a_bom(tmp_path) -> None:
    path = tmp_path / "training_log.csv"
    path.write_text("\ufeff" + TRAINING_LOG, encoding="utf-8")
    epoch, best, median_epoch = status.read_training_log(path)
    assert epoch == 3  # the half-written 4th row must be ignored, not crash
    assert best == pytest.approx(0.70)
    assert median_epoch == pytest.approx(15.0)
    assert status.read_training_log(tmp_path / "missing.csv") == (None, None, None)


GRID_LOG = """[2026-11-01 00:04:15] grid start: 2 run(s), pid 17880
[2026-11-01 00:04:15] [1/2] fold 0 seed 0: train (C:/ckpt/fold0_seed0)
[train] epoch 001/100: train_loss=0.5 val_loss=0.4 val_dice=0.60 (12.0s: train 8.0s / val 4.0s) *
[train] epoch 002/100: train_loss=0.4 val_loss=0.3 val_dice=0.66 (20.0s: train 15.0s / val 5.0s)
[train] early stopping after 5 epochs without improvement
[train] done: best val_dice=0.66 at epoch 2 | checkpoint C:/ckpt/fold0_seed0/best.pt | 1.5 min
[2026-11-01 00:20:00] ERROR fold 0 seed 1: CUDA error (attempts=3)
[2026-11-01 00:21:00] [2/2] fold 1 seed 0: resume (C:/ckpt/fold1_seed0)
[train] epoch 007/100: train_loss=0.3 val_loss=0.2 val_dice=0.71 (19.0s: train 15.0s / val 4.0s) *
[train] epoch 008/100: train_loss=0.2 val_loss=0.2 val_dice=0.69 (21.0s: train 16.0s / val 5.0s)
"""


def test_parse_grid_log_reports_failures_durations_and_the_current_run(tmp_path) -> None:
    path = tmp_path / "grid.log"
    path.write_text(GRID_LOG, encoding="utf-8")
    info = status.parse_grid_log(path)
    assert info["failures"] == ["fold0/seed1"]
    assert info["durations"] == [pytest.approx(90.0)]
    current = info["current"]
    assert (current["fold"], current["seed"]) == (1, 0)
    assert current["status"] == "resume"
    assert current["epoch"] == 8 and current["total_epochs"] == 100
    assert current["epoch_seconds"] == pytest.approx(21.0)
    assert current["best_val_dice"] == pytest.approx(0.71)  # best of this block only
    empty = status.parse_grid_log(None)
    assert empty["failures"] == [] and empty["current"] == {}



def _state(*, done: int, total: int, run_seconds: float = 600.0) -> status.ExperimentState:
    spec = status.ExperimentSpec("synthetic", "configs/cvc_clinicdb/unet_resnet50.yaml")
    runs = []
    for index in range(total):
        run = status.RunState(0, index, status.STATUS_SKIP if index < done else "train")
        if index < done:
            run.done_seconds = run_seconds + 60.0 * index
            run.best_val_dice = 0.5 + 0.01 * index
            run.epoch = 30
        run.total_epochs = 100
        runs.append(run)
    return status.ExperimentState(spec=spec, available=True, runs=runs)


def test_experiment_eta_uses_the_median_run_and_the_partial_epoch() -> None:
    state = _state(done=3, total=5, run_seconds=600.0)
    assert state.done == 3 and state.total == 5
    assert state.median_run_seconds == pytest.approx(660.0)  # median of 600, 660, 720
    # two pending runs without epoch information -> two median runs
    assert state.eta_seconds == pytest.approx(2 * 660.0)
    # a resumed run with a known epoch only pays for its remaining epochs
    state.runs[3].status = status.STATUS_RESUME
    state.runs[3].epoch = 90
    state.runs[3].epoch_seconds = 10.0
    assert state.eta_seconds == pytest.approx(10 * 10.0 + 660.0)
    # a run that only needs its evaluation costs the fixed evaluation estimate
    state.runs[4].status = status.STATUS_EVALUATE
    assert state.eta_seconds == pytest.approx(100.0 + status.EVAL_SECONDS)
    assert state.best_val_dice[0] == pytest.approx(0.52)
    assert state.best_val_dice[1] == "fold0/seed2"
    assert state.mean_best_val_dice == pytest.approx((0.50 + 0.51 + 0.52) / 3)


def test_failures_are_split_into_unresolved_and_recovered() -> None:
    state = _state(done=2, total=3)
    state.failures = ["fold0/seed0", "fold0/seed2"]
    assert state.recovered_failures == ["fold0/seed0"]  # that run is finished now
    assert state.unresolved_failures == ["fold0/seed2"]  # still not evaluated


def test_queue_eta_counts_the_unknown_steps_separately() -> None:
    states = {"unet": _state(done=3, total=5)}
    queue = status.QueueState(
        name="q",
        pid=0,
        alive=False,
        steps=[
            {
                "name": "unet_grid",
                "kind": "grid",
                "config": "configs/cvc_clinicdb/unet_resnet50.yaml",
            },
            {
                "name": "later_grid",
                "kind": "grid",
                "config": "configs/cvc_clinicdb/unet_resnet50_data025.yaml",
            },
            {"name": "compare", "kind": "compare"},
        ],
        current_step="unet_grid",
    )
    total, estimates, unknown = status.queue_eta(queue, states)
    assert [name for name, _ in estimates] == ["unet_grid", "later_grid", "compare"]
    assert unknown == 1  # data025 has no state at all
    assert total == pytest.approx(2 * 660.0 + status.POST_STEP_SECONDS)
    assert status.queue_eta(None, states) == (None, [], 0)


def test_render_prints_queue_experiments_and_the_tail(tmp_path) -> None:
    states = [_state(done=3, total=5)]
    queue = status.QueueState(
        name="overnight",
        pid=1234,
        alive=True,
        steps=[],
        current_step=None,
        waiting_for="cvc_swin_grid.pid (pid 99)",
    )
    text = status.render(
        states,
        queue,
        [(tmp_path / "fake_grid.pid", 1234, True)],
        tail=2,
    )
    assert "MELBA status" in text
    assert "overnight" in text and "waiting for cvc_swin_grid.pid" in text
    assert "synthetic" in text and "3/5" in text
    assert "best so far" in text and "legend" in text
    assert "active job" in text

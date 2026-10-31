"""Training loop with epoch-level checkpointing, resume and CV-grid helpers."""

from __future__ import annotations

from .grid import (
    STATUS_EVALUATE,
    STATUS_RESUME,
    STATUS_SKIP,
    STATUS_TRAIN,
    eval_marker_path,
    expected_eval_paths,
    iter_runs,
    run_status,
    write_eval_marker,
)
from .trainer import (
    TrainSummary,
    atomic_torch_save,
    build_loss,
    capture_rng_state,
    restore_rng_state,
    run_dir_for,
    train,
)

__all__ = [
    "STATUS_EVALUATE",
    "STATUS_RESUME",
    "STATUS_SKIP",
    "STATUS_TRAIN",
    "TrainSummary",
    "atomic_torch_save",
    "build_loss",
    "capture_rng_state",
    "eval_marker_path",
    "expected_eval_paths",
    "iter_runs",
    "restore_rng_state",
    "run_dir_for",
    "run_status",
    "train",
    "write_eval_marker",
]

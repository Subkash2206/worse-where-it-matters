"""Resume-aware helpers for cross-validation grids.

The grid decides per run whether to skip it, run only its evaluation, resume
training from ``last.pt`` or train from scratch.  Evaluation completion is tracked
by an explicit marker file written *after* the per-case CSV and metadata exist, so
a job killed mid-evaluation is never mistaken for a finished one.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "STATUS_EVALUATE",
    "STATUS_RESUME",
    "STATUS_SKIP",
    "STATUS_TRAIN",
    "eval_marker_path",
    "expected_eval_paths",
    "iter_runs",
    "run_status",
    "write_eval_marker",
]

STATUS_SKIP = "skip"          # trained and evaluated: nothing to do
STATUS_EVALUATE = "evaluate"  # trained, evaluation missing
STATUS_RESUME = "resume"      # interrupted mid-training: last.pt exists
STATUS_TRAIN = "train"        # nothing on disk yet


def eval_marker_path(per_case_csv: str | Path) -> Path:
    """Marker written once an evaluation finished writing all its outputs."""
    per_case = Path(per_case_csv)
    stem = per_case.stem
    if stem.endswith("_per_case"):
        stem = stem[: -len("_per_case")]
    return per_case.with_name(stem + "_complete.json")


def expected_eval_paths(config: Any, fold: int, seed: int) -> tuple[Path, Path]:
    """Where ``run_evaluation`` will write the per-case CSV and run metadata."""
    output = config.resolve(config.require("evaluation.output_dir"))
    split = str(config.get("evaluation.split", "test"))
    dataset_name = str(config.get("dataset.name", "dataset"))
    stem = f"{dataset_name}_{split}_fold{int(fold)}_seed{int(seed)}"
    return output / f"{stem}_per_case.csv", output / f"{stem}_run.json"


def run_status(run_dir: str | Path, eval_marker: str | Path | None = None) -> str:
    """Classify a fold/seed run from its artefacts (pure function of the disk)."""
    run_dir = Path(run_dir)
    trained = (run_dir / "run_done.json").is_file()
    resumable = (run_dir / "last.pt").is_file()
    evaluated = eval_marker is not None and Path(eval_marker).is_file()
    if trained and evaluated:
        return STATUS_SKIP
    if trained:
        return STATUS_EVALUATE
    if resumable:
        return STATUS_RESUME
    return STATUS_TRAIN


def iter_runs(folds: Sequence[int], seeds: Sequence[int]) -> list[tuple[int, int]]:
    """Deterministic (fold, seed) order: fold-major, seed-minor."""
    return [(int(fold), int(seed)) for fold in folds for seed in seeds]


def write_eval_marker(
    marker: str | Path,
    *,
    fold: int,
    seed: int,
    per_case_csv: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Record that one run's evaluation completed."""
    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "fold": int(fold),
        "seed": int(seed),
        "per_case_csv": str(per_case_csv),
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

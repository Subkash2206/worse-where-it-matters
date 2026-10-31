"""Resume-aware helpers for cross-validation grids.

The grid decides per run whether to skip it, run only its evaluation, resume
training from ``last.pt`` or train from scratch.  Evaluation completion is tracked
by an explicit marker file written *after* the per-case CSV and metadata exist, so
a job killed mid-evaluation is never mistaken for a finished one.

Each run is executed in its own child process (``scripts/run_one.py``) by a parent
grid: a CUDA/cuDNN crash therefore destroys only that child's CUDA context, and the
parent retries the run from its ``last.pt`` instead of dying with it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "CUDA_ERROR_PATTERNS",
    "EXIT_CUDA",
    "EXIT_OTHER",
    "EXIT_SUCCESS",
    "MAX_RETRIES",
    "RETRY_SLEEP_SECONDS",
    "STATUS_EVALUATE",
    "STATUS_RESUME",
    "STATUS_SKIP",
    "STATUS_TRAIN",
    "AttemptResult",
    "eval_marker_path",
    "execute_with_retries",
    "expected_eval_paths",
    "is_cuda_error",
    "is_cuda_failure",
    "iter_runs",
    "run_status",
    "write_eval_marker",
]

STATUS_SKIP = "skip"          # trained and evaluated: nothing to do
STATUS_EVALUATE = "evaluate"  # trained, evaluation missing
STATUS_RESUME = "resume"      # interrupted mid-training: last.pt exists
STATUS_TRAIN = "train"        # nothing on disk yet

#: exit codes returned by ``scripts/run_one.py``
EXIT_SUCCESS = 0
EXIT_CUDA = 3   # CUDA/cuDNN RuntimeError: retryable in a fresh process
EXIT_OTHER = 4  # any other failure

#: a failed run is retried this many times before it is marked failed
MAX_RETRIES = 3
#: pause between retries (the GPU is left idle so the driver can reclaim memory)
RETRY_SLEEP_SECONDS = 30


@dataclass(frozen=True)
class AttemptResult:
    """Outcome of executing one run, including every attempt's exit code."""

    success: bool
    attempts: int
    returncodes: tuple[int, ...]


def is_cuda_failure(returncode: int) -> bool:
    """True when a child failed with a CUDA/cuDNN error (retryable)."""
    return int(returncode) == EXIT_CUDA


#: substrings that identify a CUDA/cuDNN failure in an exception message
CUDA_ERROR_PATTERNS: tuple[str, ...] = (
    "cudnn",
    "cublas",
    "cuda error",
    "cuda failure",
    "cuda runtime",
    "cuda driver",
    "cuda out of memory",
    "cuda kernel",
    "device-side assert",
    "outofmemoryerror",
)


def is_cuda_error(error: BaseException) -> bool:
    """Classify an exception as a CUDA/cuDNN failure worth retrying in a new process.

    Only CUDA-specific markers count: a plain ``RuntimeError`` or a filesystem error
    must not be retried three times just because a path happens to contain "cuda".
    """
    if type(error).__name__ in {"OutOfMemoryError", "CudaError"}:
        return True
    message = f"{type(error).__name__}: {error}".lower()
    return any(pattern in message for pattern in CUDA_ERROR_PATTERNS)


def execute_with_retries(
    command: Sequence[str],
    *,
    runner: Callable[[Sequence[str]], int],
    sleep: Callable[[float], Any] = time.sleep,
    cleanup: Callable[[], Any] | None = None,
    max_retries: int = MAX_RETRIES,
    retry_sleep: float = RETRY_SLEEP_SECONDS,
    log: Callable[[str], Any] = print,
) -> AttemptResult:
    """Run ``command`` until it succeeds, at most ``1 + max_retries`` times.

    ``runner`` returns the child's exit code (0 = success).  After a failure the
    optional ``cleanup`` callback runs (freeing GPU memory) and the caller waits
    ``retry_sleep`` seconds before trying again, so the retry starts from ``last.pt``
    in a fresh process.  Exhausting the retries is *not* an exception: the grid marks
    the run failed and continues with the next one.
    """
    returncodes: list[int] = []
    attempts = 1 + max(0, int(max_retries))
    for attempt in range(1, attempts + 1):
        code = int(runner(command))
        returncodes.append(code)
        if code == EXIT_SUCCESS:
            return AttemptResult(True, attempt, tuple(returncodes))
        log(f"attempt {attempt}/{attempts} failed with exit code {code}")
        if attempt == attempts:
            break
        if cleanup is not None:
            cleanup()
        sleep(float(retry_sleep))
    return AttemptResult(False, len(returncodes), tuple(returncodes))


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

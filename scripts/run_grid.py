"""Orchestrate a fold x seed grid, one child process per run, with retries.

For every (fold, seed) the parent inspects the disk and then either skips the run or
spawns ``scripts/run_one.py`` for it:

* trained **and** evaluated -> skip;
* trained, evaluation missing -> the child runs the evaluation only;
* ``last.pt`` present -> the child resumes training from that epoch;
* otherwise -> the child trains from scratch.

Running each run in its own process means a CUDA/cuDNN crash (e.g.
``CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH``) only kills that child, so it cannot
poison the next run's CUDA context.  After a failure the parent frees GPU memory,
waits 30 s and retries the run from its ``last.pt``, up to three times.  A run that
still fails is logged, marked failed, and the grid **continues** with the next run;
the final summary lists done / skipped / failed runs.

Usage
-----
    python scripts/run_grid.py --config configs/cvc_clinicdb/unet_resnet50.yaml --folds 0 1 2 3 4 --seeds 0 1 2 --epochs 100 --patience 20
    python scripts/run_grid.py --config ... --status-only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config  # noqa: E402
from melba.training import (  # noqa: E402
    MAX_RETRIES,
    RETRY_SLEEP_SECONDS,
    STATUS_SKIP,
    AttemptResult,
    eval_marker_path,
    execute_with_retries,
    expected_eval_paths,
    is_cuda_failure,
    iter_runs,
    run_dir_for,
    run_status,
)


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{_stamp()}] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="experiment YAML in configs/")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument(
        "--patience", type=int, default=None, help="override training.early_stopping_patience"
    )
    parser.add_argument("--retries", type=int, default=MAX_RETRIES, help="retries per run")
    parser.add_argument(
        "--retry-sleep", type=float, default=RETRY_SLEEP_SECONDS, help="seconds between retries"
    )
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--pid-file", default=None, help="default: logs/cvc_unet_grid.pid")
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="print the state of every run and exit without training",
    )
    return parser.parse_args(argv)


def free_gpu_memory() -> None:
    """Release cached CUDA allocations between attempts (the child already exited)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover - cleanup must never mask the failure
        pass


def run_child(command: list[str]) -> int:
    """Execute one ``run_one.py`` child, inheriting this process's stdout/stderr."""
    return int(subprocess.run(command, cwd=str(_REPO_ROOT)).returncode)


def run_report(run_dir: Path, marker: Path) -> dict[str, object]:
    """Training/evaluation facts for the log, read back from the child's artefacts."""
    report: dict[str, object] = {}
    done_path = run_dir / "run_done.json"
    if done_path.is_file():
        report["training"] = json.loads(done_path.read_text(encoding="utf-8"))
    if marker.is_file():
        report["evaluation"] = json.loads(marker.read_text(encoding="utf-8"))
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: list[str] = list(args.overrides)
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    if args.patience is not None:
        overrides.append(f"training.early_stopping_patience={args.patience}")
    try:
        base_config = load_config(args.config, overrides)
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    runs = iter_runs(args.folds, args.seeds)
    if not runs:
        print("ERROR: no runs selected", file=sys.stderr)
        return 2

    def state_of(fold: int, seed: int) -> tuple[str, Path, Path]:
        config = base_config.with_overrides(
            {"seed": int(seed), "dataset.split.fold": int(fold), "evaluation.fold": int(fold)}
        )
        run_dir = run_dir_for(config, fold, seed)
        per_case, _ = expected_eval_paths(config, fold, seed)
        return run_status(run_dir, eval_marker_path(per_case)), run_dir, eval_marker_path(per_case)

    if args.status_only:
        log(f"status for {len(runs)} run(s) of {args.config}")
        for fold, seed in runs:
            status, run_dir, _ = state_of(fold, seed)
            log(f"fold {fold} seed {seed}: {status:<8} ({run_dir})")
        return 0

    pid_file = (
        Path(args.pid_file) if args.pid_file else _REPO_ROOT / "logs" / "cvc_unet_grid.pid"
    )
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    started_all = time.perf_counter()
    durations: list[float] = []
    skipped: list[tuple[int, int]] = []
    finished: list[tuple[int, int]] = []
    failures: list[dict[str, object]] = []
    log(
        f"grid start: {len(runs)} run(s), config {args.config}, pid {os.getpid()}, "
        f"one child process per run, {args.retries} retr(y/ies), {args.retry_sleep:g}s backoff"
    )


    try:
        for index, (fold, seed) in enumerate(runs, start=1):
            try:
                status, run_dir, marker = state_of(fold, seed)
                log(f"[{index}/{len(runs)}] fold {fold} seed {seed}: {status} ({run_dir})")
                if status == STATUS_SKIP:
                    skipped.append((fold, seed))
                    log(f"[{index}/{len(runs)}] fold {fold} seed {seed}: already done")
                    continue

                command = [
                    sys.executable,
                    "-u",
                    str(_REPO_ROOT / "scripts" / "run_one.py"),
                    "--config",
                    str(args.config),
                    "--fold",
                    str(fold),
                    "--seed",
                    str(seed),
                ]
                if overrides:
                    command += ["--set", *overrides]

                run_started = time.perf_counter()
                attempt = execute_with_retries(
                    command,
                    runner=run_child,
                    cleanup=free_gpu_memory,
                    max_retries=args.retries,
                    retry_sleep=args.retry_sleep,
                    log=lambda message: log(f"fold {fold} seed {seed}: {message}"),
                )
                elapsed = time.perf_counter() - run_started
                post_status, _, _ = state_of(fold, seed)

                if attempt.success and post_status != STATUS_SKIP:
                    attempt = AttemptResult(False, attempt.attempts, attempt.returncodes)
                    log(
                        f"fold {fold} seed {seed}: child exited 0 but the run is "
                        f"'{post_status}', treating it as failed"
                    )

                if not attempt.success:
                    failures.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "attempts": attempt.attempts,
                            "returncodes": list(attempt.returncodes),
                            "cuda": any(is_cuda_failure(code) for code in attempt.returncodes),
                            "status": post_status,
                            "seconds": elapsed,
                        }
                    )
                    log(
                        f"fold {fold} seed {seed}: FAILED after {attempt.attempts} attempt(s) "
                        f"(codes {list(attempt.returncodes)}); continuing with the next run"
                    )
                    continue

                report = run_report(run_dir, marker)
                training = report.get("training") or {}
                best = training.get("best_value")
                best_text = f"{best:.4f}" if isinstance(best, (int, float)) else "n/a"
                durations.append(elapsed)
                finished.append((fold, seed))
                remaining = len(runs) - index
                mean = sum(durations) / len(durations)
                log(
                    f"fold {fold} seed {seed}: finished in {elapsed / 60:.1f} min "
                    f"({attempt.attempts} attempt(s)); best "
                    f"{training.get('monitor', 'metric')}={best_text}; mean "
                    f"{mean / 60:.1f} min/run; projected remaining "
                    f"{mean * remaining / 3600:.2f} h for {remaining} run(s)"
                )
            except Exception as error:  # one broken run must never stop the grid
                failures.append({"fold": fold, "seed": seed, "error": repr(error)})
                log(f"ERROR fold {fold} seed {seed}: {error}")
                traceback.print_exc()
    except KeyboardInterrupt:
        log("interrupted: every run resumes from its last.pt when the grid restarts")
        return 130
    finally:
        pid_file.unlink(missing_ok=True)
        free_gpu_memory()


    total = time.perf_counter() - started_all
    log(
        f"grid finished in {total / 3600:.2f} h: {len(finished)} done, "
        f"{len(skipped)} skipped, {len(failures)} failed"
    )
    if finished:
        log("done: " + ", ".join(f"fold{fold}/seed{seed}" for fold, seed in finished))
    if skipped:
        log("skipped: " + ", ".join(f"fold{fold}/seed{seed}" for fold, seed in skipped))
    for failure in failures:
        log(
            "failed: "
            f"fold{failure.get('fold')}/seed{failure.get('seed')} "
            f"attempts={failure.get('attempts', '?')} "
            f"codes={failure.get('returncodes', failure.get('error', '?'))}"
        )

    summary_path = _REPO_ROOT / "logs" / "cvc_unet_grid_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "config": str(args.config),
                "runs": len(runs),
                "finished": finished,
                "skipped": skipped,
                "failures": failures,
                "total_seconds": total,
                "retries_per_run": args.retries,
                "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"summary written to {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


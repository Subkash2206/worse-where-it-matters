"""Run a fold x seed cross-validation grid, resuming whatever is incomplete.

For every (fold, seed) pair the grid inspects the disk before doing any work:

* trained **and** evaluated -> skip;
* trained, evaluation missing -> run only the evaluation;
* ``last.pt`` present -> resume training from that epoch;
* otherwise -> train from scratch.

Training happens in this single process, so stopping the grid (one PID, written to
``logs/cvc_unet_grid.pid``) stops training at the next epoch boundary and the run can
be resumed by launching the same command again.

Usage
-----
    python scripts/run_grid.py --config configs/cvc_clinicdb/unet_resnet50.yaml --folds 0 1 2 3 4 --seeds 0 1 2 --epochs 100 --patience 20
    python scripts/run_grid.py --config ... --status-only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config  # noqa: E402
from melba.eval import run_evaluation  # noqa: E402
from melba.training import (  # noqa: E402
    STATUS_EVALUATE,
    STATUS_RESUME,
    STATUS_SKIP,
    eval_marker_path,
    expected_eval_paths,
    iter_runs,
    run_dir_for,
    run_status,
    train,
    write_eval_marker,
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
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--pid-file", default=None, help="default: logs/cvc_unet_grid.pid")
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="print the state of every run and exit without training",
    )
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm bars")
    return parser.parse_args(argv)


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

    def config_for(fold: int, seed: int):
        return base_config.with_overrides(
            {"seed": int(seed), "dataset.split.fold": int(fold), "evaluation.fold": int(fold)}
        )

    def state_of(fold: int, seed: int):
        config = config_for(fold, seed)
        run_dir = run_dir_for(config, fold, seed)
        per_case, _ = expected_eval_paths(config, fold, seed)
        marker = eval_marker_path(per_case)
        return run_status(run_dir, marker), run_dir

    if args.status_only:
        log(f"status for {len(runs)} run(s) of {args.config}")
        for fold, seed in runs:
            status, run_dir = state_of(fold, seed)
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
    log(f"grid start: {len(runs)} run(s), config {args.config}, pid {os.getpid()}")

    try:
        for index, (fold, seed) in enumerate(runs, start=1):
            status, run_dir = state_of(fold, seed)
            log(f"[{index}/{len(runs)}] fold {fold} seed {seed}: {status} ({run_dir})")
            if status == STATUS_SKIP:
                skipped.append((fold, seed))
                log(f"[{index}/{len(runs)}] fold {fold} seed {seed}: already trained and evaluated")
                continue

            run_started = time.perf_counter()
            try:
                if status == STATUS_EVALUATE:
                    log(f"fold {fold} seed {seed}: training complete, running evaluation only")
                else:
                    summary = train(
                        config_for(fold, seed),
                        fold=fold,
                        seed=seed,
                        epochs=args.epochs,
                        resume=(status == STATUS_RESUME),
                        progress=not args.no_progress,
                    )
                    log(
                        f"fold {fold} seed {seed}: training done after {summary.epochs_run} "
                        f"epoch(s), best {summary.monitor}={summary.best_value:.4f} "
                        f"(epoch {summary.best_epoch}), "
                        f"{summary.median_epoch_seconds:.1f}s median epoch"
                    )

                result = run_evaluation(
                    config_for(fold, seed), fold=fold, progress=not args.no_progress
                )
                per_case, _ = expected_eval_paths(config_for(fold, seed), fold, seed)
                write_eval_marker(
                    eval_marker_path(per_case),
                    fold=fold,
                    seed=seed,
                    per_case_csv=result.per_case_csv,
                    extra={"methods": list(result.methods), "n_cases": result.n_cases},
                )
                elapsed = time.perf_counter() - run_started
                durations.append(elapsed)
                finished.append((fold, seed))
                remaining = len(runs) - index
                mean = statistics.fmean(durations)
                log(
                    f"fold {fold} seed {seed}: finished in {elapsed / 60:.1f} min; "
                    f"mean {mean / 60:.1f} min/run; projected remaining "
                    f"{mean * remaining / 3600:.2f} h for {remaining} run(s)"
                )
            except Exception as error:  # one broken run must not stop the grid
                failures.append({"fold": fold, "seed": seed, "error": repr(error)})
                log(f"ERROR fold {fold} seed {seed}: {error}")
                traceback.print_exc()
    except KeyboardInterrupt:
        log("interrupted: the current run resumes from last.pt when the grid restarts")
        return 130
    finally:
        pid_file.unlink(missing_ok=True)

    total = time.perf_counter() - started_all
    log(
        f"grid finished in {total / 3600:.2f} h: {len(finished)} run(s) trained+evaluated, "
        f"{len(skipped)} skipped, {len(failures)} failed"
    )
    for failure in failures:
        log(f"failed: fold {failure['fold']} seed {failure['seed']}: {failure['error']}")
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
                "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


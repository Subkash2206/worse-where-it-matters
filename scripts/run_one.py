"""Run exactly one fold/seed job in its own process: train if needed, then evaluate.

``scripts/run_grid.py`` spawns one of these per run.  Running in a child process means
a CUDA/cuDNN failure (e.g. ``CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH``) cannot poison
the CUDA context of the next run: the child's context is destroyed with the process,
and the parent retries the run from ``last.pt``.

Exit codes (used by the parent to decide whether to retry):

===  ===================================================================
0    success (or nothing to do: training complete + evaluation present)
2    configuration error (not retryable in practice, but it is reported)
3    CUDA / cuDNN failure -> retryable in a fresh process
4    any other failure
===  ===================================================================

Usage
-----
    python scripts/run_one.py --config configs/cvc_clinicdb/unet_resnet50.yaml \
        --fold 2 --seed 0 --epochs 100 --patience 20
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.eval import run_evaluation  # noqa: E402
from melba.training import (  # noqa: E402
    EXIT_CUDA,
    EXIT_OTHER,
    STATUS_EVALUATE,
    STATUS_RESUME,
    STATUS_SKIP,
    eval_marker_path,
    expected_eval_paths,
    is_cuda_error,
    run_dir_for,
    run_status,
    train,
    write_eval_marker,
)

EXIT_CONFIG = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="experiment YAML in configs/")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument(
        "--patience", type=int, default=None, help="override training.early_stopping_patience"
    )
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace):
    """Config for this run: CLI overrides plus the run's own fold/seed."""
    overrides = parse_set_args(args.overrides)
    overrides.update(
        {
            "seed": int(args.seed),
            "dataset.split.fold": int(args.fold),
            "evaluation.fold": int(args.fold),
        }
    )
    if args.epochs is not None:
        overrides["training.epochs"] = int(args.epochs)
    if args.patience is not None:
        overrides["training.early_stopping_patience"] = int(args.patience)
    return load_config(args.config, overrides)


def free_gpu_memory() -> None:
    """Drop cached allocations before the process exits (the parent also waits)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover - never let cleanup mask the real failure
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = build_config(args)
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return EXIT_CONFIG

    try:
        run_dir = run_dir_for(config, args.fold, args.seed)
        per_case, _ = expected_eval_paths(config, args.fold, args.seed)
        marker = eval_marker_path(per_case)
        status = run_status(run_dir, marker)
        print(f"[run_one] fold {args.fold} seed {args.seed}: {status} ({run_dir})", flush=True)
        if status == STATUS_SKIP:
            print("[run_one] training and evaluation already complete; nothing to do", flush=True)
            return 0
        if status != STATUS_EVALUATE:
            summary = train(
                config,
                fold=args.fold,
                seed=args.seed,
                epochs=args.epochs,
                resume=(status == STATUS_RESUME),
                progress=False,
            )
            print(
                f"[run_one] trained {summary.epochs_run} epoch(s), best "
                f"{summary.monitor}={summary.best_value:.4f} (epoch {summary.best_epoch})",
                flush=True,
            )
        result = run_evaluation(config, fold=args.fold, progress=False)
        write_eval_marker(
            marker,
            fold=args.fold,
            seed=args.seed,
            per_case_csv=result.per_case_csv,
            extra={"methods": list(result.methods), "n_cases": result.n_cases},
        )
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return EXIT_CONFIG
    except Exception as error:  # noqa: BLE001 - the exit code tells the parent what to do
        traceback.print_exc()
        free_gpu_memory()
        return EXIT_CUDA if is_cuda_error(error) else EXIT_OTHER

    print(f"[run_one] fold {args.fold} seed {args.seed}: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

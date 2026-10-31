"""Train a segmentation model with validation-only checkpoint selection.

Usage
-----
    python scripts/train.py --config configs/cvc_clinicdb/unet_resnet50.yaml

The training split is used for optimisation, the validation split for model
selection, and the test split is never touched here (``scripts/evaluate.py``
refuses to report the selection split by default).

Runs are resumable: every epoch atomically writes ``last.pt`` next to ``best.pt``
(model, optimizer, scheduler, AMP scaler, epoch, early-stopping state and all RNG
states), so starting the same command again continues where it stopped.  Use
``--fresh`` to ignore an existing ``last.pt`` and start over.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config  # noqa: E402
from melba.data import SplitError  # noqa: E402
from melba.training import train  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="experiment YAML in configs/")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument(
        "--fold", type=int, default=None, help="fold index (default dataset.split.fold)"
    )
    parser.add_argument("--seed", type=int, default=None, help="override seed")
    parser.add_argument(
        "--fresh", action="store_true", help="ignore an existing last.pt and train from scratch"
    )
    parser.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: list[str] = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    if args.fold is not None:
        overrides.append(f"dataset.split.fold={args.fold}")
    try:
        config = load_config(args.config, overrides)
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    try:
        summary = train(
            config,
            fold=args.fold,
            seed=args.seed,
            epochs=args.epochs,
            resume=not args.fresh,
            progress=not args.no_progress,
        )
    except (ConfigError, SplitError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"epochs run   : {summary.epochs_run}")
    print(f"best {summary.monitor:<8}: {summary.best_value:.4f} (epoch {summary.best_epoch})")
    print(f"checkpoint   : {summary.checkpoint}")
    print(f"last state   : {summary.last_checkpoint}")
    print(f"training log : {summary.log_path}")
    print(f"run marker   : {summary.done_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

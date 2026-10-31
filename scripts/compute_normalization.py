"""Compute per-fold training-set mean/std for ``dataset.normalization.kind: dataset``.

ImageNet statistics are wrong for encoders trained from scratch, so a from-scratch
model (e.g. ``monai_swinunetr``) is normalized with the statistics of its own
training split.  For every requested fold this script iterates the *training* frames
of that fold (never validation or test frames), accumulates per-channel sums, and
writes one entry per fold into the statistics JSON:

    {
      "fold0": {"mean": [...], "std": [...], "n_frames": 376, ...},
      "fold1": {...}
    }

Usage
-----
    python scripts/compute_normalization.py --config configs/cvc_clinicdb/swinunetr_2d.yaml
    python scripts/compute_normalization.py --config ... --folds 0 1 --force

Existing folds are kept unless ``--force`` is given, so the script is cheap to re-run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC := _REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import path
    sys.path.insert(0, str(_SRC))

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.data import (  # noqa: E402
    DEFAULT_STATS_FILE,
    SplitError,
    compute_fold_statistics,
    read_statistics,
    write_statistics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="experiment YAML in configs/")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="folds (default: all)")
    parser.add_argument(
        "--output",
        default=None,
        help=f"statistics JSON (default: dataset.normalization.stats_file or {DEFAULT_STATS_FILE})",
    )
    parser.add_argument("--force", action="store_true", help="recompute folds already present")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, parse_set_args(args.overrides))
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    n_folds = int(config.get("dataset.split.n_folds", 5) or 1)
    folds = [int(fold) for fold in (args.folds if args.folds else range(n_folds))]
    stats_file = (
        args.output
        or (config.get("dataset.normalization") or {}).get("stats_file")
        or DEFAULT_STATS_FILE
    )
    output_path = config.resolve(str(stats_file))

    existing = read_statistics(output_path)
    pending = {
        f"fold{fold}": fold
        for fold in folds
        if args.force or f"fold{fold}" not in existing
    }
    if not pending:
        print(f"nothing to do: {output_path} already holds folds {sorted(existing)}")
        return 0

    entries: dict[str, dict] = {}
    try:
        for key, fold in pending.items():
            statistics = compute_fold_statistics(config, fold, progress=not args.no_progress)
            entries[key] = statistics
            print(
                f"{key}: {statistics['n_frames']} training frames | "
                f"mean={[round(value, 4) for value in statistics['mean']]} | "
                f"std={[round(value, 4) for value in statistics['std']]}"
            )
    except (ConfigError, SplitError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    written = write_statistics(output_path, entries, force=args.force)
    print(f"written {sorted(written)} to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

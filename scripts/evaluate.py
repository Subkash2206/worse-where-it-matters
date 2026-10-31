"""Evaluate every configured TTA method on one split.

Usage
-----
    python scripts/evaluate.py --config configs/cvc_clinicdb/unet_resnet50.yaml

Writes, under ``evaluation.output_dir``:

* ``<dataset>_<split>_per_case.csv`` — long format, one row per case/method/metric;
* ``<dataset>_<split>_summary.csv`` — descriptive statistics with case counts;
* ``<dataset>_<split>_run.json``   — config, seed, environment, checkpoint hash.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.data import SplitError  # noqa: E402
from melba.eval import evaluate  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="experiment YAML in configs/")
    parser.add_argument("--split", default=None, choices=["val", "test"], help="override evaluation.split")
    parser.add_argument("--fold", type=int, default=None, help="fold index (default evaluation.fold)")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    parser.add_argument("--output-dir", default=None, help="override evaluation.output_dir")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config overrides, e.g. model.checkpoint=path/to.pt",
    )
    parser.add_argument(
        "--allow-selection-split",
        action="store_true",
        help="permit reporting the split used for checkpoint selection (development only)",
    )
    parser.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, parse_set_args(args.overrides))
    try:
        result = evaluate(
            config,
            split=args.split,
            limit=args.limit,
            output_dir=args.output_dir,
            allow_selection_split=args.allow_selection_split or None,
            progress=not args.no_progress,
            fold=args.fold,
        )
    except (ConfigError, SplitError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"run_id       : {result.run_id}")
    print(f"split        : {result.split} (fold {result.fold}, {result.n_cases} cases)")
    print(f"methods      : {', '.join(result.methods)}")
    print(f"per-case CSV : {result.per_case_csv}")
    if result.summary_csv:
        print(f"summary CSV  : {result.summary_csv}")
    print(f"run metadata : {result.metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Paired statistical analysis of long-format per-case result files.

Usage
-----
    # single fold
    python scripts/run_stats.py --config configs/cvc_clinicdb/unet_resnet50.yaml \
        --per-case results/.../cvc_clinicdb_test_fold0_per_case.csv --treatment tta8

    # pooled over all 5 cross-validation folds (one video = one test unit)
    python scripts/run_stats.py --config configs/cvc_clinicdb/unet_resnet50.yaml \
        --per-case results/.../cvc_clinicdb_test_fold{0,1,2,3,4}_per_case.csv --treatment tta8

The primary test is the Wilcoxon signed-rank test over per-group values (video
level), with a seeded bootstrap CI, rank-biserial effect size, Holm correction and
the paired t-test kept only as a labelled secondary result.  When several per-case
files are given they are pooled across folds, and the pool is validated: every case
and every video must appear in exactly one fold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.eval.records import read_long_csv  # noqa: E402
from melba.stats import analyze_long, analyze_pooled_folds  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--per-case",
        required=True,
        nargs="+",
        help="one long-format CSV per fold; two or more are pooled across folds",
    )
    parser.add_argument("--treatment", required=True, help="TTA method to compare against the control")
    parser.add_argument("--control", default="baseline", help="control method (default: baseline)")
    parser.add_argument("--config", default=None, help="experiment YAML providing stats.* defaults")
    parser.add_argument("--unit", default=None, choices=["group", "case"], help="pairing unit")
    parser.add_argument("--output", default=None, help="output CSV path")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats_config: dict[str, object] = {
        "unit": "group",
        "primary_test": "wilcoxon",
        "alternative": "two-sided",
        "n_bootstrap": 10000,
        "ci": 0.95,
        "bootstrap_seed": 0,
        "correction": "holm",
        "output_dir": "results/stats",
    }
    config = None
    if args.config:
        try:
            config = load_config(args.config, parse_set_args(args.overrides))
        except ConfigError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        stats_config.update(config.section("stats"))

    unit = args.unit or str(stats_config.get("unit", "group"))
    analysis_kwargs = {
        "treatment": args.treatment,
        "control": args.control,
        "unit": unit,
        "primary_test": str(stats_config.get("primary_test", "wilcoxon")),
        "alternative": str(stats_config.get("alternative", "two-sided")),
        "n_bootstrap": int(stats_config.get("n_bootstrap", 10000)),
        "ci": float(stats_config.get("ci", 0.95)),
        "seed": int(stats_config.get("bootstrap_seed", 0)),
        "correction": str(stats_config.get("correction", "holm")),
    }
    try:
        frames = [read_long_csv(path) for path in args.per_case]
        pooled = len(frames) > 1
        if pooled:
            table = analyze_pooled_folds(frames, **analysis_kwargs)
        else:
            table = analyze_long(frames[0], **analysis_kwargs)
    except (ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = (_REPO_ROOT / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        base = stats_config.get("output_dir", "results/stats")
        output_dir = config.resolve(base) if config is not None else _REPO_ROOT / str(base)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = "pooled_" if pooled else ""
        output_path = output_dir / f"{prefix}{args.treatment}_vs_{args.control}_{unit}.csv"
    table.to_csv(output_path, index=False)

    display = table[
        [
            "metric",
            "n_units",
            "delta_mean",
            "ci_low",
            "ci_high",
            "effect_size",
            "p_value",
            "p_value_adjusted",
            "preferred",
        ]
    ]
    print(f"treatment: {args.treatment} | control: {args.control} | unit: {unit}")
    if pooled:
        print(
            f"pooled across {len(frames)} folds: "
            f"{int(table['n_folds'].iloc[0])} folds, "
            f"{int(table['n_groups_pooled'].iloc[0])} videos, "
            f"{int(table['n_cases_pooled'].iloc[0])} cases"
        )
    print(f"primary test: {table['test'].iloc[0]} (paired t-test kept as secondary column)")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nwritten to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

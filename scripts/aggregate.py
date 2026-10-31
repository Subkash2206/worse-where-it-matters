"""Pool the whole cross-validation grid: seeds, folds, paired stats and seed ensembles.

Produces (under ``stats.output_dir/aggregate`` by default):

* ``pooled_by_seed`` - per method and metric, the mean of the pooled 29-video results
  per seed, the standard deviation across seeds and each seed's value;
* ``paired_by_seed`` - video-level paired tests (Wilcoxon, bootstrap CI, rank-biserial,
  Holm) pooled over the 5 folds, computed inside every seed;
* ``paired_seed_averaged`` - the same tests after averaging metric values per case over
  seeds, so each video stays one independent test unit;
* ``detection_by_seed`` / ``detection_seed_averaged`` - missed-detection rates with
  cluster-bootstrap CIs, transition tables and exact McNemar tests, pooled over folds;
* ``ensemble_per_case`` / ``ensemble_comparison`` / ``ensemble_detection`` - the seed
  ensemble of the baseline maps (``ensemble3``) and of the tta4 maps
  (``ensemble3_tta4``), evaluated with the same metric code as the runs and compared
  with ``baseline``/``tta4`` on the same metrics, missed detections included.

Usage
-----
    python scripts/aggregate.py --config configs/cvc_clinicdb/unet_resnet50.yaml
    python scripts/aggregate.py --config ... --seeds 0 --output-dir results/_agg --no-ensemble
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.eval import aggregate_all, discover_runs  # noqa: E402
from melba.utils import git_revision  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="experiment YAML in configs/")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="folds (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--per-case-dir", default=None, help="directory with the per-case CSVs")
    parser.add_argument("--maps-dir", default=None, help="root of the probability maps")
    parser.add_argument("--output-dir", default=None, help="default: stats.output_dir/aggregate")
    parser.add_argument("--unit", default=None, choices=["group", "case"], help="pairing unit")
    parser.add_argument("--treatments", nargs="+", default=["tta2", "tta4", "tta8"])
    parser.add_argument("--control", default="baseline")
    parser.add_argument(
        "--n-bootstrap", type=int, default=None, help="override stats.n_bootstrap"
    )
    parser.add_argument(
        "--no-ensemble", action="store_true", help="skip the seed ensembles (maps not needed)"
    )
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def _print_table(title: str, table, columns=None, max_rows: int = 40) -> None:
    if table is None or table.empty:
        print(f"\n{title}: (empty)")
        return
    printable = table if columns is None else table[[c for c in columns if c in table.columns]]
    print(f"\n{title}")
    print(
        printable.head(max_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, parse_set_args(args.overrides))
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    stats = config.section("stats")
    unit = args.unit or str(stats.get("unit", "group"))
    n_bootstrap = int(args.n_bootstrap or stats.get("n_bootstrap", 10000))
    ci = float(stats.get("ci", 0.95))
    correction = str(stats.get("correction", "holm"))
    bootstrap_seed = int(stats.get("bootstrap_seed", 0))
    primary_test = str(stats.get("primary_test", "wilcoxon"))
    alternative = str(stats.get("alternative", "two-sided"))

    runs = discover_runs(
        config,
        folds=args.folds,
        seeds=args.seeds,
        output_dir=args.per_case_dir,
        map_root=args.maps_dir,
    )
    if not runs:
        print(
            "ERROR: no per-case results found for the requested folds/seeds; "
            "run scripts/run_grid.py first",
            file=sys.stderr,
        )
        return 2
    folds_found = sorted({run.fold for run in runs})
    seeds_found = sorted({run.seed for run in runs})
    print(f"{len(runs)} run(s) found: folds {folds_found}, seeds {seeds_found}")
    if args.output_dir:
        output_dir: Path | None = Path(args.output_dir)
    else:
        base = config.resolve(str(stats.get("output_dir", "results/stats")))
        output_dir = base / "aggregate"

    try:
        result = aggregate_all(
            config,
            runs=runs,
            unit=unit,
            treatments=args.treatments,
            control=args.control,
            primary_test=primary_test,
            alternative=alternative,
            n_bootstrap=n_bootstrap,
            ci=ci,
            bootstrap_seed=bootstrap_seed,
            correction=correction,
            ensemble=not args.no_ensemble,
            expected_seeds=tuple(int(seed) for seed in args.seeds),
            output_dir=output_dir,
        )
    except (ValueError, ConfigError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    tables = result.tables
    _print_table(
        f"pooled per-video results (mean +/- std across {len(seeds_found)} seed(s))",
        tables.get("pooled_by_seed"),
        columns=["method", "metric", "n_units", "n_cases", "n_seeds", "mean", "std"],
    )
    _print_table(
        "paired tests, seeds averaged per case first (unit = video)",
        tables.get("paired_seed_averaged"),
        columns=[
            "treatment",
            "control",
            "metric",
            "n_units",
            "n_cases_used",
            "restriction",
            "delta_mean",
            "ci_low",
            "ci_high",
            "p_value",
            "p_value_adjusted",
            "preferred",
        ],
    )
    _print_table(
        "missed detections, pooled over folds (seeds averaged per case)",
        tables.get("detection_seed_averaged"),
        columns=[
            "method",
            "n_cases",
            "n_missed",
            "missed_rate",
            "ci_low",
            "ci_high",
            "nonempty_to_empty",
            "empty_to_nonempty",
            "mcnemar_p",
            "mcnemar_p_adjusted",
        ],
    )
    _print_table(
        "seed ensemble vs baseline / tta4",
        tables.get("ensemble_comparison"),
        columns=[
            "treatment",
            "control",
            "metric",
            "n_units",
            "n_cases_used",
            "delta_mean",
            "ci_low",
            "ci_high",
            "p_value",
            "p_value_adjusted",
            "preferred",
        ],
    )
    _print_table(
        "seed ensemble missed detections",
        tables.get("ensemble_detection"),
        columns=[
            "method",
            "n_cases",
            "n_missed",
            "missed_rate",
            "ci_low",
            "ci_high",
            "nonempty_to_empty",
            "empty_to_nonempty",
            "mcnemar_p",
        ],
    )
    for note in result.notes:
        print(f"note: {note}")

    print(f"\ngit revision: {git_revision()}")
    if result.paths:
        print("written:")
        for name, path in sorted(result.paths.items()):
            print(f"  {name:<28} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


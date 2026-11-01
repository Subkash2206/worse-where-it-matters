"""Compare two experiments (same folds/seeds) side by side.

Produces under ``results/compare_models/``:

* ``methods.csv`` — baseline / tta4 / tta8 / ensemble3, Dice / BF1 / HD95, per model;
* ``deltas.csv`` — paired tests vs each model's own baseline (video level);
* ``detection.csv`` — missed detections, transitions and exact McNemar per model;
* ``tails.csv`` — HD95 percentiles and failure rates per model;
* ``residual_error.csv`` — Spearman(baseline error, TTA delta) per model and pooled over
  both models, with video-cluster bootstrap CIs (the residual-error test);
* ``comparison.md`` — all of the above rendered as markdown.

Usage
-----
    python scripts/compare_models.py \
        --configs configs/cvc_clinicdb/unet_resnet50.yaml configs/cvc_clinicdb/swinunetr_2d.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.eval import (  # noqa: E402
    TailThresholds,
    deltas_table,
    detection_table,
    load_ground_truth,
    load_model,
    method_table,
    missed_detection_table,
    residual_error_table,
    tail_table_for_models,
    write_markdown,
)
from melba.eval.aggregate import write_tables  # noqa: E402
from melba.utils import git_revision  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--configs", nargs="+", required=True, help="two (or more) experiment YAMLs")
    parser.add_argument("--labels", nargs="+", default=None, help="short names for the models")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--control", default="baseline")
    parser.add_argument("--treatment", default="tta4")
    parser.add_argument("--output-dir", default=None, help="default: results/compare_models")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def _print(table, columns=None, max_rows: int = 40, title: str = "") -> None:
    if table is None or table.empty:
        print(f"\n{title}: (empty)")
        return
    if title:
        print(f"\n{title}")
    view = table if columns is None else table[[c for c in columns if c in table.columns]]
    print(view.head(max_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.configs) < 2:
        print("ERROR: --configs needs at least two experiment configs", file=sys.stderr)
        return 2
    if args.labels is not None and len(args.labels) != len(args.configs):
        print("ERROR: --labels must match --configs in length", file=sys.stderr)
        return 2

    overrides = parse_set_args(args.overrides)
    configs = []
    for path in args.configs:
        try:
            configs.append(load_config(path, overrides))
        except ConfigError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
    labels = args.labels or [
        str(config.get("experiment.name", f"model{index}"))
        for index, config in enumerate(configs)
    ]
    folds = args.folds or list(range(int(configs[0].get("dataset.split.n_folds", 5) or 1)))

    # the reference masks and the split files are identical for every model
    try:
        ground_truth = load_ground_truth(configs[0], folds)
    except (ConfigError, FileNotFoundError) as error:
        print(f"ERROR: could not load the reference masks: {error}", file=sys.stderr)
        return 2

    models = []
    for config, label in zip(configs, labels):
        try:
            models.append(
                load_model(
                    config,
                    name=label,
                    seeds=args.seeds,
                    folds=folds,
                    ensemble=True,
                    ground_truth=ground_truth,
                )
            )
        except (ValueError, FileNotFoundError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2

    output_dir = (
        Path(args.output_dir) if args.output_dir else _REPO_ROOT / "results" / "compare_models"
    )
    tables = {
        "methods": method_table(models),
        "deltas": deltas_table(models, control=args.control, n_bootstrap=args.n_bootstrap),
        "detection": missed_detection_table(
            models, control=args.control, n_bootstrap=args.n_bootstrap
        ),
        "tails": tail_table_for_models(
            models,
            control=args.control,
            thresholds=TailThresholds(),
            n_bootstrap=args.n_bootstrap,
        ),
        "residual_error": residual_error_table(
            models,
            treatment=args.treatment,
            control=args.control,
            n_bootstrap=args.n_bootstrap,
        ),
    }


    _print(tables["methods"], title="(1) methods side by side (video level, mean ± std)")
    _print(
        tables["deltas"],
        columns=[
            "model", "treatment", "metric", "n_units", "n_cases_used", "delta_mean",
            "ci_low", "ci_high", "p_value", "p_value_adjusted", "preferred",
        ],
        title=f"(2) paired vs {args.control} (video level, Holm)",
    )
    _print(
        tables["detection"],
        columns=[
            "model", "method", "n_cases", "n_missed", "missed_rate", "ci_low", "ci_high",
            "nonempty_to_empty", "empty_to_nonempty", "mcnemar_p",
        ],
        title="(3) missed detections",
    )
    _print(
        tables["tails"],
        columns=[
            "model", "method", "hd95_p95", "hd95_failure_rate", "dice_failure_rate",
            "delta_hd95_failure_rate_estimate", "delta_hd95_failure_rate_ci_low",
            "delta_hd95_failure_rate_ci_high",
        ],
        title="(4) tails: HD95 p95 and failure rates",
    )
    _print(
        tables["residual_error"],
        columns=[
            "scope", "metric", "restriction", "n_cases", "n_clusters", "delta_mean",
            "spearman_rho", "ci_low", "ci_high", "spearman_p", "p_bootstrap",
        ],
        title=(
            f"(5) residual-error test: Spearman(baseline error, "
            f"{args.treatment} - {args.control})"
        ),
    )

    paths = write_tables(
        tables,
        output_dir,
        manifest={
            "configs": [
                str(config.path) if config.path else str(path)
                for config, path in zip(configs, args.configs)
            ],
            "labels": labels,
            "folds": folds,
            "seeds": args.seeds,
            "control": args.control,
            "treatment": args.treatment,
            "n_bootstrap": args.n_bootstrap,
            "git_revision": git_revision(),
        },
    )

    sections = [
        (
            "(1) Methods side by side",
            tables["methods"],
            "Per-video values pooled over the folds, mean ± std over the videos.",
        ),
        (
            f"(2) Paired comparisons vs {args.control}",
            tables["deltas"],
            "Wilcoxon signed-rank, bootstrap CI of the mean delta, rank-biserial effect "
            "size, Holm correction; distance metrics on both-non-empty cases.",
        ),
        (
            "(3) Missed detections",
            tables["detection"],
            "Cluster bootstrap CI of the missed rate, transitions vs the model's own "
            "baseline and exact McNemar.",
        ),
        (
            "(4) Tail metrics",
            tables["tails"],
            "HD95 percentiles and failure rates (HD95 > 50 px, Dice < 0.5).",
        ),
        (
            "(5) Residual-error test",
            tables["residual_error"],
            "Spearman between the baseline error and the TTA delta per case: a negative "
            "rho for HD95 (positive for Dice) means TTA helps most where the baseline is "
            "weakest. `pooled_models` repeats it over both models; CIs resample videos.",
        ),
    ]
    summary = write_markdown(
        str(output_dir / "comparison.md"),
        title="Cross-model comparison: " + " vs ".join(labels),
        preamble=(
            f"{len(models)} models, folds {folds}, seeds {args.seeds}; statistics identical "
            "to scripts/aggregate.py (video-level pairing, Holm correction)."
        ),
        sections=sections,
    )

    print(f"\ngit revision: {git_revision()}")
    print("written:")
    for name, path in sorted(paths.items()):
        print(f"  {name:<16} {path}")
    print(f"  {'comparison.md':<16} {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


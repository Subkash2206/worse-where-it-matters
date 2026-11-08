"""Data-fraction sweep comparison: 100 % vs 25 % vs 50 % of the training videos.

Each variant trained on a seeded subset of whole training videos and evaluated on the same
folds, so the models can be paired video by video — the question is whether the TTA gain
survives less training data.

Writes under ``results/compare_data_sweep/``:

* ``methods.csv`` — baseline / tta2 / tta4 / tta8, Dice / BF1 / HD95 / ASSD, per dataset
  fraction (video level, mean ± std);
* ``deltas.csv`` — the TTA gains of every fraction against its own baseline, so the
  fractions are comparable without a cross-model pairing;
* ``gain_difference.csv`` — paired by video, absolute **and** relative (log-ratio for
  distances, fraction of the remaining error for Dice) between the fractions;
* ``residual_error.csv`` — Oldham's residual-error test + paired-swap null per fraction;
* ``comparison.md``.

Usage
-----
    python scripts/compare_data_sweep.py          # the three configs below, seed 0
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
    deltas_table,
    gain_difference_table,
    load_ground_truth,
    load_model,
    method_table,
    residual_error_table,
    write_markdown,
)
from melba.eval.aggregate import write_tables  # noqa: E402
from melba.utils import git_revision  # noqa: E402

import pandas as pd  # noqa: E402

#: fraction label -> experiment config (all trained with the same seed and folds)
DEFAULT_CONFIGS: tuple[str, ...] = (
    "configs/cvc_clinicdb/unet_resnet50.yaml",
    "configs/cvc_clinicdb/unet_resnet50_data050.yaml",
    "configs/cvc_clinicdb/unet_resnet50_data025.yaml",
)
DEFAULT_LABELS: tuple[str, ...] = ("100pct", "50pct", "25pct")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="default [0]: the fraction variants were trained with seed 0 only",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels = args.labels or list(DEFAULT_LABELS)[: len(args.configs)]
    if len(labels) != len(args.configs):
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
    folds = args.folds or list(range(int(configs[0].get("dataset.split.n_folds", 5) or 1)))
    try:
        ground_truth = load_ground_truth(configs[0], folds)
    except (ConfigError, FileNotFoundError) as error:
        print(f"ERROR: could not load the reference masks: {error}", file=sys.stderr)
        return 2

    models = []
    for config in configs:
        try:
            models.append(
                load_model(
                    config,
                    name=labels[len(models)],
                    seeds=args.seeds,
                    folds=folds,
                    ensemble=False,  # one seed per fraction: an ensemble would be a copy
                    ground_truth=ground_truth,
                )
            )
        except (ValueError, FileNotFoundError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _REPO_ROOT / "results" / "compare_data_sweep"
    )
    # the table compares every model with the first (100 %); the remaining ordered pairs
    # (50 % vs 25 %) are appended so every fraction pair is tested directly
    gain_parts = [gain_difference_table(models, n_bootstrap=args.n_bootstrap)]
    for index in range(1, len(models)):
        for other in range(index + 1, len(models)):
            gain_parts.append(
                gain_difference_table(
                    [models[index], models[other]], n_bootstrap=args.n_bootstrap
                )
            )
    gain_parts = [table for table in gain_parts if table is not None and not table.empty]
    gain_all = (
        pd.concat(gain_parts, ignore_index=True) if gain_parts else pd.DataFrame()
    )
    tables = {
        "methods": method_table(models),
        "deltas": deltas_table(models, n_bootstrap=args.n_bootstrap),
        "gain_difference": gain_all,
        "residual_error": residual_error_table(
            models, n_bootstrap=args.n_bootstrap, n_permutations=args.n_permutations
        ),
    }

    print("\n(1) methods side by side (video level, mean over videos)")
    print(tables["methods"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n(2) TTA gains of each fraction vs its own baseline (dice / hd95 / assd)")
    print(
        tables["deltas"]
        .loc[lambda frame: frame["metric"].isin(["dice", "hd95", "assd"])]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print("\n(3) between-fraction gain difference (absolute and relative scales)")
    print(gain_all.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n(4) residual-error test per fraction (Oldham + paired-swap null)")
    print(
        tables["residual_error"].to_string(index=False, float_format=lambda v: f"{v:.4f}")
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
            "n_bootstrap": args.n_bootstrap,
            "n_permutations": args.n_permutations,
            "git_revision": git_revision(),
        },
    )
    sections = [
        ("(1) Methods side by side", tables["methods"], "Video level, mean ± std."),
        (
            "(2) TTA gains of each fraction vs its own baseline",
            tables["deltas"],
            "Wilcoxon signed-rank, video bootstrap CI of the mean delta, Holm correction.",
        ),
        (
            "(3) Between-fraction gain difference",
            gain_all,
            "Paired by video. `scale=absolute` is metric units; `scale=relative` is "
            "scale-free: `log(treatment / control)` for distances and "
            "`(treatment - control) / (1 - control)` for Dice.",
        ),
        (
            "(4) Residual-error test per fraction",
            tables["residual_error"],
            "Oldham's `Spearman((control + treatment)/2, delta)` with a video-cluster "
            "bootstrap CI and a paired-swap permutation null; `rho_naive` is the "
            "regression-to-the-mean biased version, kept for comparison.",
        ),
    ]
    summary = write_markdown(
        str(output_dir / "comparison.md"),
        title="Data-fraction sweep: " + " vs ".join(labels),
        preamble=(
            f"{len(models)} fractions ({', '.join(labels)}), folds {folds}, seeds "
            f"{args.seeds}; the variants differ only in `training.data_fraction` (whole "
            "training videos, seeded)."
        ),
        sections=sections,
    )
    print("\nwritten:")
    for name, path in sorted(paths.items()):
        print(f"  {name:<28} {path}")
    print(f"  {'comparison.md':<28} {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

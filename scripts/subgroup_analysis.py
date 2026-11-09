"""Lesion-size subgroup analysis over an existing cross-validation grid (CPU only).

Reads the per-case CSVs an internal grid already produced (no GPU, no training), attaches the
reference lesion size of every case, and writes the tables a Clinical Translation &
Validation review asks for: metric and miss-rate per size bin, the paired TTA delta per bin,
and a test of whether the TTA gain differs across bins at all.

Usage
-----
    python scripts/subgroup_analysis.py --models cvc_clinicdb/unet_resnet50 \
        cvc_clinicdb/swinunetr_2d --folds 0 1 2 3 4 --seeds 0 1 2
    python scripts/subgroup_analysis.py --models ... --n-bootstrap 2000      # quick look
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from melba.config import ConfigError, load_config  # noqa: E402
from melba.eval import (  # noqa: E402
    discover_runs,
    frames_by_seed,
    seed_averaged_frame,
    string_ids,
    write_markdown,
)
from melba.eval.aggregate import write_tables  # noqa: E402
from melba.eval.subgroups import (  # noqa: E402
    SIZE_BIN_LABELS,
    add_size_bins,
    empty_reference_subgroup_table,
    lesion_size_table,
    subgroup_delta_table,
    subgroup_interaction_table,
    subgroup_summary_table,
    tertile_cutoffs,
)
from melba.utils import git_revision  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cvc_clinicdb/unet_resnet50", "cvc_clinicdb/swinunetr_2d"],
        help="internal configs (relative to configs/, without .yaml)",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--metrics", nargs="+", default=["dice", "hd95"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def _analyse_model(
    name: str, config: Any, sizes: pd.DataFrame, args: argparse.Namespace
) -> dict[str, pd.DataFrame]:
    """Every subgroup table for one model (seeds averaged per case first)."""
    runs = discover_runs(config, folds=list(args.folds), seeds=list(args.seeds))
    if not runs:
        raise ConfigError(f"no per-case results found for {name}")
    by_seed = frames_by_seed(runs)
    averaged = string_ids(
        seed_averaged_frame(by_seed).drop(columns=["n_seeds"], errors="ignore")
    )
    resolved = tertile_cutoffs(sizes["gt_pixels"])
    enriched, cutoffs = add_size_bins(averaged, sizes, cutoffs=resolved)
    tables: dict[str, pd.DataFrame] = {
        "subgroup_summary": subgroup_summary_table(
            enriched, metrics=tuple(args.metrics), n_bootstrap=args.n_bootstrap
        ),
        "subgroup_delta": subgroup_delta_table(
            enriched, metrics=tuple(args.metrics), n_bootstrap=args.n_bootstrap
        ),
        "subgroup_interaction": subgroup_interaction_table(
            enriched, metrics=tuple(args.metrics), n_bootstrap=args.n_bootstrap
        ),
        "subgroup_empty_reference": empty_reference_subgroup_table(
            enriched, n_bootstrap=args.n_bootstrap
        ),
    }
    for table in tables.values():
        if table is not None and not table.empty:
            table.insert(0, "model", name)
            table["cutoff_low_px"] = cutoffs[0]
            table["cutoff_high_px"] = cutoffs[1]
    return tables


def _concat(tables: dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    pieces = [table for name, table in tables.items() if name == key and not table.empty]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _write_markdown(output_dir: Path, tables: dict[str, pd.DataFrame], sizes: pd.DataFrame) -> None:
    summary = _concat(tables, "subgroup_summary")
    deltas = _concat(tables, "subgroup_delta")
    interaction = _concat(tables, "subgroup_interaction")
    empty = _concat(tables, "subgroup_empty_reference")
    delta_columns = [
        column
        for column in (
            "model", "size_bin", "treatment", "metric", "n_units", "n_cases_used",
            "delta_mean", "ci_low", "ci_high", "p_value", "p_value_adjusted", "preferred",
        )
        if column in deltas.columns
    ]
    sections = [
        (
            "Reference lesion size per case",
            sizes.groupby("fold")["gt_pixels"].describe().reset_index(),
            "`gt_pixels` is the reference foreground area at the native resolution (mm^3 is "
            "in the CSV when the dataset declares a spacing).",
        ),
        (
            "Metric and miss rate per subgroup (seeds averaged per case)",
            summary,
            "Tertiles of the reference area (`small`/`medium`/`large`); CIs resample groups "
            "(videos), so frames of one video are not counted as independent.",
        ),
        (
            "Paired TTA delta per size bin",
            deltas[delta_columns] if delta_columns else deltas,
            "The same paired comparison as the main tables, computed inside each bin: a "
            "negative HD95 delta means the treatment is better there.",
        ),
        (
            "Interaction: does the TTA gain differ across size bins?",
            interaction,
            "`kruskal_p` tests the per-case paired deltas across the three bins; `delta_gap` "
            "is `large - small` with a bootstrap CI over groups. A CI excluding 0 means the "
            "TTA effect genuinely depends on lesion size.",
        ),
        (
            "Empty-reference subgroup",
            empty,
            "`n_cases = 0` for the polyp collections: every reference contains a polyp, so the "
            "subgroup does not exist there (it is populated for BraTS ET).",
        ),
    ]
    write_markdown(
        str(output_dir / "subgroup_analysis.md"),
        title="MELBA subgroup analysis - lesion size",
        preamble=(
            "CPU-only re-analysis of the per-case CSVs an existing cross-validation grid "
            "already wrote. Size bins are tertiles of the reference area computed per dataset; "
            "the cutoffs are repeated in every table (`cutoff_low_px`, `cutoff_high_px`)."
        ),
        sections=sections,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = (
        Path(args.output_dir) if args.output_dir else _REPO_ROOT / "results" / "subgroups"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    collected: dict[str, pd.DataFrame] = {}
    sizes_written = False
    sizes_for_markdown: pd.DataFrame = pd.DataFrame()
    for model in args.models:
        config_path = _REPO_ROOT / "configs" / f"{model}.yaml"
        if not config_path.is_file():
            print(f"ERROR: config not found: {config_path}", file=sys.stderr)
            return 2
        config = load_config(config_path)
        sizes = lesion_size_table(config, folds=list(args.folds))
        if not sizes_written:
            sizes.to_csv(output_root / "lesion_sizes.csv", index=False)
            sizes_for_markdown = sizes
            sizes_written = True
        try:
            tables = _analyse_model(model, config, sizes, args)
        except ConfigError as error:
            print(f"ERROR [{model}]: {error}", file=sys.stderr)
            return 2
        print(f"\n{model}")
        for key, table in tables.items():
            if table is None or table.empty:
                continue
            collected[key] = pd.concat(
                [collected.get(key, pd.DataFrame()), table], ignore_index=True
            )
            print(f"  {key}: {len(table)} rows")
    if not collected:
        print("ERROR: no subgroup tables produced", file=sys.stderr)
        return 2
    write_tables(
        collected,
        output_root,
        manifest={
            "models": list(args.models),
            "folds": list(args.folds),
            "seeds": list(args.seeds),
            "metrics": list(args.metrics),
            "n_bootstrap": args.n_bootstrap,
            "size_bins": list(SIZE_BIN_LABELS),
            "git_revision": git_revision(),
        },
    )
    _write_markdown(output_root, collected, sizes_for_markdown)
    print(f"\nwritten to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


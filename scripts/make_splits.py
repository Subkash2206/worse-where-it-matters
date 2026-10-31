"""Create video-level, cross-validated splits for the dataset of a config.

Usage
-----
    python scripts/make_splits.py --config configs/cvc_clinicdb/unet_resnet50.yaml

Writes one CSV per fold (``dataset.split.output_pattern``) plus ``folds.json`` with
the provenance.  The frame -> video mapping is data (``group_map_file``); the script
**fails loudly** when its case ids do not match the image filename stems exactly,
because a silently incomplete mapping is how frame-level leakage starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.data import (  # noqa: E402
    SplitError,
    actual_fractions,
    contiguous_group_map,
    discover_cases,
    folded_splits,
    folds_summary,
    load_group_map,
    write_split_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="experiment YAML in configs/")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config overrides, e.g. dataset.split.seed=7",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing fold files")
    return parser.parse_args(argv)


def validate_mapping(disk_cases: list[str], group_map: dict[str, str], mapping_path: Path) -> None:
    """Every image stem must be mapped, and every mapped case must exist on disk."""
    on_disk = set(disk_cases)
    mapped = set(group_map)
    missing = sorted(on_disk - mapped)
    unknown = sorted(mapped - on_disk)
    if missing or unknown:
        lines = [
            f"{mapping_path} does not match the dataset files.",
            f"  images on disk : {len(on_disk)}",
            f"  mapped case ids: {len(mapped)}",
            f"  missing mappings ({len(missing)}): {missing[:20]}",
            f"  mapped but not on disk ({len(unknown)}): {unknown[:20]}",
            "The mapping must contain every image filename stem (case_id) exactly once.",
        ]
        raise SplitError("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, parse_set_args(args.overrides))

    try:
        cases = discover_cases(config)
        strategy = str(config.get("dataset.split.strategy", "group_map")).strip().lower()
        if strategy == "group_map":
            mapping_path = config.require_path("dataset.split.group_map_file")
            group_map = load_group_map(
                mapping_path,
                case_column=str(config.get("dataset.split.frame_column", "case_id")),
                group_column=str(config.get("dataset.split.group_column", "group_id")),
            )
            validate_mapping(cases, group_map, mapping_path)
        elif strategy == "contiguous_blocks":
            mapping_path = Path("<contiguous_blocks proxy>")
            group_map = contiguous_group_map(
                cases, int(config.require("dataset.split.block_size"))
            )
            print(
                "WARNING: 'contiguous_blocks' groups consecutive frames and is only a proxy "
                "for video identity. State this limitation in the paper."
            )
        else:
            raise SplitError(
                f"Unknown dataset.split.strategy {strategy!r}; use 'group_map' or "
                "'contiguous_blocks'"
            )

        n_folds = int(config.get("dataset.split.n_folds", 5))
        seed = int(config.require("dataset.split.seed"))
        folds = folded_splits(cases, group_map, n_folds=n_folds, seed=seed)

        pattern = config.require("dataset.split.output_pattern")
        written: dict[int, str] = {}
        for fold, records in folds.items():
            target = config.resolve(str(pattern).format(fold=fold))
            write_split_csv(records, target, overwrite=bool(args.force))
            written[fold] = str(target)
    except (ConfigError, SplitError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    summary = folds_summary(folds)
    total_cases = len(cases)
    total_groups = len(set(group_map.values()))
    print(
        f"Wrote {n_folds} folds for {total_cases} cases / {total_groups} groups "
        f"(seed {seed}, strategy {strategy})"
    )
    for fold, per_split in summary.items():
        fractions = actual_fractions(folds[fold])
        print(
            f"  fold {fold}: "
            + " | ".join(
                f"{split} {per_split[split]['groups']}g/{per_split[split]['cases']}c "
                f"({fractions[split]:.1%})"
                for split in ("train", "val", "test")
            )
            + f"  -> {written[fold]}"
        )
    print("  every group is a test unit in exactly one fold (verified)")

    metadata_path = (
        config.resolve(str(pattern).format(fold=0)).parent / "cvc_clinicdb_folds.json"
    )
    metadata = {
        "config_path": str(config.path) if config.path else None,
        "config_hash": config.fingerprint,
        "strategy": strategy,
        "group_map_file": str(mapping_path),
        "n_folds": n_folds,
        "seed": seed,
        "n_cases": total_cases,
        "n_groups": total_groups,
        "folds": {int(fold): summary[fold] for fold in summary},
        "files": {int(fold): written[fold] for fold in written},
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Provenance written to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

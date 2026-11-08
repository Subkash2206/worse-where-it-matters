"""Write the case-level BraTS 2021 fold CSVs and list the empty-ET cases per fold.

For fold ``k`` the test set is fold ``k``, the validation set is fold ``k+1`` (rotating)
and the training set is everything else; the assignment is seeded, so the folds are stable
across runs.  Cases with no enhancing tumour are *not* dropped: they are reported (and
written to ``data/splits/brats2021_empty_et.json``) because an empty reference makes Dice
undefined for that case and the aggregate has to know why.

Usage
-----
    python scripts/make_brats_splits.py --config configs/brats2021/segresnet.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.data import (  # noqa: E402
    ET_LABEL,
    discover_brats_cases,
    folded_splits,
    split_file_for_fold,
    write_split_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="BraTS experiment YAML")
    parser.add_argument("--force", action="store_true", help="overwrite existing split CSVs")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def et_voxels(config, case_id: str) -> int:
    """Number of enhancing-tumour voxels in one case (0 for an empty-ET patient)."""
    root = config.resolve(str(config.require("dataset.root")))
    path = root / case_id / f"{case_id}_seg.nii.gz"
    labels = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    return int((labels == float(ET_LABEL)).sum())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, parse_set_args(args.overrides))
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    try:
        cases = discover_brats_cases(config)
    except Exception as error:  # noqa: BLE001 - reported, not raised
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    n_folds = int(config.get("dataset.split.n_folds", 5) or 5)
    seed = int(config.get("dataset.split.seed", 2024) or 2024)
    # case level CV: every patient is its own group, so no leakage is possible
    group_map = {case: case for case in cases}
    folds = folded_splits(cases, group_map, n_folds=n_folds, seed=seed)

    empty_et = {case: et_voxels(config, case) for case in cases}
    empty_cases = sorted(case for case, count in empty_et.items() if count == 0)
    print(f"{len(cases)} case(s), {n_folds} folds, split seed {seed}")
    if empty_cases:
        print(f"{len(empty_cases)} case(s) without enhancing tumour: {empty_cases}")

    written: list[Path] = []
    for fold, records in sorted(folds.items()):
        path = split_file_for_fold(config, fold)
        try:
            write_split_csv(records, path, overwrite=args.force)
        except Exception as error:  # noqa: BLE001 - a clobbered split must not be silent
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        written.append(path)
        test = [record.case_id for record in records if record.split == "test"]
        val = [record.case_id for record in records if record.split == "val"]
        train = [record.case_id for record in records if record.split == "train"]
        test_empty = [case for case in test if empty_et[case] == 0]
        val_empty = [case for case in val if empty_et[case] == 0]
        print(
            f"  fold {fold}: train {len(train)} / val {len(val)} / test {len(test)}"
            f" | empty-ET in test {test_empty} | in val {val_empty}"
        )

    report = {
        "cases": len(cases),
        "n_folds": n_folds,
        "split_seed": seed,
        "et_voxels": empty_et,
        "empty_et_cases": empty_cases,
        "folds": {
            str(fold): {
                split: sorted(
                    record.case_id for record in records if record.split == split
                )
                for split in ("train", "val", "test")
            }
            for fold, records in sorted(folds.items())
        },
    }
    report_path = _REPO_ROOT / "data" / "splits" / "brats2021_empty_et.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten {len(written)} split file(s):")
    for path in written:
        print(f"  {path}")
    print(f"  {report_path}  (empty-ET cases and the fold composition)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

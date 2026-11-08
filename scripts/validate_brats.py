"""Validate a BraTS 2021 directory (read-only) and summarise what the pipeline will see.

For every ``BraTS2021_*`` folder it checks that the four modalities and the segmentation
exist, that they share one shape and one voxel spacing (the spacing is what the metrics
report, so it must be right), and what labels the segmentation actually contains.  Cases
with no enhancing tumour (label 4) are listed: an empty reference makes Dice undefined for
that case, so the aggregate has to know about them up front.

Nothing below ``dataset.root`` is ever written; the report goes to
``data/splits/brats2021_validation.json``.

Usage
-----
    python scripts/validate_brats.py --config configs/brats2021/segresnet.yaml
    python scripts/validate_brats.py --config ... --limit 20 --no-report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.data import ET_LABEL, MODALITIES, SplitError, discover_brats_cases  # noqa: E402

#: what the official BraTS 2021 training data uses
EXPECTED_SHAPE = (240, 240, 155)
EXPECTED_LABELS = (0, 1, 2, 4)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="BraTS experiment YAML")
    parser.add_argument("--limit", type=int, default=None, help="check only the first N cases")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-report", action="store_true", help="do not write the JSON report")
    parser.add_argument("--expected-shape", type=int, nargs=3, default=list(EXPECTED_SHAPE))
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def check_case(directory: Path, expected_shape: tuple[int, int, int]) -> dict:
    """Everything the pipeline depends on for one patient (header-only where possible)."""
    case = directory.name
    problems: list[str] = []
    result: dict = {"case_id": case, "problems": problems}
    headers: dict[str, tuple[tuple[int, ...], tuple[float, ...]]] = {}
    for modality in MODALITIES:
        path = directory / f"{case}_{modality}.nii.gz"
        if not path.is_file():
            problems.append(f"missing {modality}")
            continue
        header = nib.load(str(path)).header
        headers[modality] = (
            tuple(int(value) for value in header.get_data_shape()[:3]),
            tuple(round(float(value), 4) for value in header.get_zooms()[:3]),
        )
    seg_path = directory / f"{case}_seg.nii.gz"
    labels: dict[str, int] = {}
    if not seg_path.is_file():
        problems.append("missing seg")
    else:
        seg = nib.load(str(seg_path))
        headers["seg"] = (
            tuple(int(value) for value in seg.header.get_data_shape()[:3]),
            tuple(round(float(value), 4) for value in seg.header.get_zooms()[:3]),
        )
        data = np.asanyarray(seg.dataobj)
        values, counts = np.unique(data, return_counts=True)
        labels = {str(int(value)): int(count) for value, count in zip(values, counts)}
        unexpected = sorted(set(int(value) for value in values) - set(EXPECTED_LABELS))
        if unexpected:
            problems.append(f"unexpected labels {unexpected}")
        result["et_voxels"] = int(labels.get(str(ET_LABEL), 0))

    shapes = {key: value[0] for key, value in headers.items()}
    spacings = {key: value[1] for key, value in headers.items()}
    result["shape"] = shapes.get("flair")
    result["spacing"] = spacings.get("flair")
    if len(set(shapes.values())) > 1:
        problems.append(f"shapes differ between files: {shapes}")
    elif shapes.get("flair") not in (None, tuple(expected_shape)) and problems == []:
        problems.append(f"unexpected shape {shapes['flair']} (expected {tuple(expected_shape)})")
    if len(set(spacings.values())) > 1:
        problems.append(f"spacing differs between files: {spacings}")
    result["labels"] = labels
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, parse_set_args(args.overrides))
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    root = config.resolve(str(config.require("dataset.root")))
    expected_shape = tuple(int(value) for value in args.expected_shape)
    try:
        cases = discover_brats_cases(config)
    except (SplitError, ConfigError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.limit:
        cases = cases[: int(args.limit)]
    print(f"root   : {root}")
    print(f"cases  : {len(cases)} (checking {len(cases)})")

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        results = list(pool.map(lambda case: check_case(root / case, expected_shape), cases))

    bad = [item for item in results if item["problems"]]
    shapes = Counter(str(item.get("shape")) for item in results)
    spacings = Counter(str(item.get("spacing")) for item in results)
    labels: Counter = Counter()
    for item in results:
        for value, count in (item.get("labels") or {}).items():
            labels[value] += int(count)
    empty_et = sorted(item["case_id"] for item in results if not item.get("et_voxels"))
    et_voxels = [item.get("et_voxels", 0) for item in results if item.get("et_voxels")]

    print(f"shapes : {dict(shapes)}")
    print(f"spacing: {dict(spacings)}")
    print(f"labels : {dict(labels)}  (voxels over all checked cases)")
    print(f"empty ET: {len(empty_et)} case(s)" + (f" -> {empty_et}" if empty_et else ""))
    if et_voxels:
        print(
            "ET voxels per case: min {} / median {:.0f} / max {}".format(
                min(et_voxels), float(np.median(et_voxels)), max(et_voxels)
            )
        )
    if bad:
        print(f"BAD CASES: {len(bad)}")
        for item in bad[:20]:
            print(f"  {item['case_id']}: {'; '.join(item['problems'])}")
    else:
        print("BAD CASES: none")

    if not args.no_report:
        report = {
            "root": str(root),
            "cases_checked": len(results),
            "expected_shape": list(expected_shape),
            "shapes": {key: count for key, count in shapes.items()},
            "spacings": {key: count for key, count in spacings.items()},
            "label_voxels": {key: count for key, count in sorted(labels.items())},
            "empty_et_cases": empty_et,
            "bad_cases": bad,
        }
        path = _REPO_ROOT / "data" / "splits" / "brats2021_validation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report : {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())


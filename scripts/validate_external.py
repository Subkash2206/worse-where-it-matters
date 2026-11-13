"""Validate the external polyp datasets (read-only) and write a report.

Reads every image/mask pair of Kvasir-SEG, CVC-ColonDB and ETIS-LaribPolypDB through the
loader the evaluation uses, and reports what a reader must know before trusting a number:
counts, extensions, native sizes, the raw mask value sets, empty masks, and which
binarisation threshold was applied.  Nothing is written inside the dataset directories.

Usage
-----
    python scripts/validate_external.py --config configs/external/polyp_external.yaml
    python scripts/validate_external.py --config ... --datasets kvasir_seg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.data.polyp_external import validate_external_dataset  # noqa: E402
from melba.eval import write_markdown  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="external evaluation YAML")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="subset of external.datasets keys (default: all)",
    )
    parser.add_argument("--output-dir", default=None, help="default: results/external_validation")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def _summary_row(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": report["dataset"],
        "n_cases": report["n_cases"],
        "image_ext": " ".join(f"{k}x{v}" for k, v in report["image_extensions"].items()),
        "mask_ext": " ".join(f"{k}x{v}" for k, v in report["mask_extensions"].items()),
        "height_min": report["image_height"]["min"],
        "height_median": report["image_height"]["median"],
        "height_max": report["image_height"]["max"],
        "width_min": report["image_width"]["min"],
        "width_median": report["image_width"]["median"],
        "width_max": report["image_width"]["max"],
        "distinct_sizes": (
            f"{report['image_height']['distinct']}x{report['image_width']['distinct']}"
        ),
        "mask_values": ", ".join(report["mask_values"]),
        "threshold_used": ", ".join(
            f"{value} ({count})" for value, count in report["mask_threshold_used"].items()
        ),
        "rgb_distinct": report["rgb_channels_distinct"],
        "n_empty_masks": report["n_empty_masks"],
        "lesion_px_median": report["foreground_pixels"]["median"],
        "lesion_px_min": report["foreground_pixels"]["min"],
        "lesion_px_max": report["foreground_pixels"]["max"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, parse_set_args(args.overrides))
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    datasets = config.section("external").get("datasets") or {}
    if not datasets:
        print("ERROR: the config declares no external.datasets", file=sys.stderr)
        return 2
    selected = list(args.datasets) if args.datasets else list(datasets)
    unknown = sorted(set(selected) - set(datasets))
    if unknown:
        print(f"ERROR: unknown external dataset(s) {unknown}", file=sys.stderr)
        return 2

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _REPO_ROOT / "results" / "external_validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict[str, Any]] = {}
    for name in selected:
        entry = datasets[name]
        dataset_config = config.with_overrides(
            {
                "dataset.name": name,
                "dataset.images_dir": entry["images_dir"],
                "dataset.masks_dir": entry.get("masks_dir"),
                "dataset.references": entry.get(
                    "references", config.get("dataset.references", "mask")
                ),
            }
        )
        try:
            report = validate_external_dataset(dataset_config)
        except ConfigError as error:
            print(f"ERROR [{name}]: {error}", file=sys.stderr)
            return 2
        report["label"] = str(entry.get("label", name))
        reports[name] = report
        print(
            f"{name}: {report['n_cases']} cases | sizes "
            f"{report['image_height']['min']}-{report['image_height']['max']} x "
            f"{report['image_width']['min']}-{report['image_width']['max']} | "
            f"mask values {sorted(report['mask_values'], key=int)} | "
            f"threshold {report['mask_threshold_used']} | "
            f"{report['n_empty_masks']} empty mask(s)"
        )

    (output_dir / "external_validation.json").write_text(
        json.dumps({"datasets": reports}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = pd.DataFrame([_summary_row(report) for report in reports.values()])
    table.to_csv(output_dir / "external_validation.csv", index=False)

    sections: list[tuple[str, Any, str]] = [
        (
            "External dataset validation",
            table,
            "Read-only audit: every image is paired with exactly one mask by file stem (the "
            "loader raises otherwise), frames must be genuinely RGB (three identical "
            "channels are rejected), and masks are binarised with the threshold shown here "
            "(`> 0` for a 0/1 label map, `mask_threshold` otherwise). `lesion_px_*` is the "
            "reference polyp area in native pixels before the 352x352 resize.",
        )
    ]
    for name, report in reports.items():
        detail = pd.DataFrame(
            [
                {"field": "label", "value": report["label"]},
                {"field": "images_dir", "value": report["images_dir"]},
                {
                    "field": "masks_dir",
                    "value": report["masks_dir"] or "(none: references are empty)",
                },
                {"field": "references", "value": report.get("references", "mask")},
                {"field": "n_cases", "value": report["n_cases"]},
                {"field": "inference_size", "value": report["inference_size"]},
                {"field": "spacing_mm", "value": report["spacing_mm"]},
                {"field": "mask_values", "value": sorted(report["mask_values"], key=int)},
                {"field": "empty_masks", "value": ", ".join(report["empty_masks"]) or "none"},
                {
                    "field": "lesion_px min/p25/median/p75/max",
                    "value": list(report["foreground_pixels"].values()),
                },
            ]
        )
        sections.append((f"{name} - details", detail, ""))
    write_markdown(
        str(output_dir / "external_validation.md"),
        title="MELBA external polyp datasets - validation",
        preamble=(
            "All three collections are used read-only and never trained on. Mask values are "
            "reported as found, not assumed."
        ),
        sections=sections,
    )
    print("written:")
    for name in ("external_validation.json", "external_validation.csv", "external_validation.md"):
        print(f"  {output_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


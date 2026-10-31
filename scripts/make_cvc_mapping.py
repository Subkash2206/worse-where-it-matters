"""Build the CVC-ClinicDB frame -> sequence mapping CSV.

Usage
-----
    python scripts/make_cvc_mapping.py --config configs/cvc_clinicdb/unet_resnet50.yaml

The frame -> sequence table itself comes from the YAML config
(``dataset.split.sequence_end_frames``, from Bernal et al., 2015).  The script
expands it into ``dataset.split.group_map_file`` (``case_id,group_id``), validates
that frames ``1..612`` appear exactly once in 29 sequences, and checks the mapping
against the actual image filename stems.  It fails loudly on any mismatch, so a
wrong or incomplete table can never silently leak frames between splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.data import (  # noqa: E402
    build_sequence_mapping,
    check_mapping_against_stems,
    mapping_statistics,
    validate_sequence_mapping,
    write_mapping_csv,
)
from melba.data.splits import SplitError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="experiment YAML in configs/")
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing mapping file")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip comparing the mapping with the image filename stems on disk",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, parse_set_args(args.overrides))

    try:
        end_frames = [int(value) for value in config.require("dataset.split.sequence_end_frames")]
        mapping = build_sequence_mapping(end_frames)
        validate_sequence_mapping(mapping, n_groups=len(end_frames))
        stats = mapping_statistics(mapping)

        if not args.no_verify:
            images_dir = config.require_path("dataset.images_dir", must_exist=True)
            extension = str(config.get("dataset.image_extension", ".tif")).lower()
            stems = {path.stem for path in images_dir.glob(f"*{extension}")}
            if not stems:
                raise SplitError(f"No '{extension}' images found in {images_dir}")
            missing, unknown = check_mapping_against_stems(mapping, stems)
            if missing or unknown:
                raise SplitError(
                    "Frame mapping does not match the image files:\n"
                    f"  images on disk : {len(stems)}\n"
                    f"  mapped frames  : {len(mapping)}\n"
                    f"  images without a mapping ({len(missing)}): {missing[:20]}\n"
                    f"  mapped but missing on disk ({len(unknown)}): {unknown[:20]}"
                )
            print(f"Verified: all {len(stems)} image filename stems match the mapping exactly.")

        target = config.require_path("dataset.split.group_map_file")
        write_mapping_csv(mapping, target, overwrite=bool(args.force))
    except (ConfigError, SplitError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"Wrote {stats['n_frames']} rows / {stats['n_groups']} sequences to {target}")
    print(
        f"  frames {stats['first_frame']}..{stats['last_frame']} each exactly once | "
        f"smallest {stats['smallest_group']} ({stats['smallest_group_frames']} frames) | "
        f"largest {stats['largest_group']} ({stats['largest_group_frames']} frames)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

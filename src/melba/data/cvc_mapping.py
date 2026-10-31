"""CVC-ClinicDB frame -> sequence (video) mapping.

CVC-ClinicDB (Bernal et al., 2015) ships 612 frames taken from 29 colonoscopy
sequences.  The filenames are global frame numbers (``1.tif`` ... ``612.tif``), so
the only way to recover the source sequence is the official end-frame table:
sequence ``k`` covers frames ``previous_end + 1 ... end_k``.

Keeping the table in the YAML config (``dataset.split.sequence_end_frames``) and
building the mapping here makes the split reproducible and auditable instead of
relying on a hand-maintained CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .splits import SplitError, case_sort_key

__all__ = [
    "build_sequence_mapping",
    "check_mapping_against_stems",
    "mapping_statistics",
    "sequence_ranges",
    "validate_sequence_mapping",
    "write_mapping_csv",
]


def sequence_ranges(end_frames: Sequence[int]) -> list[tuple[int, int, int]]:
    """Return ``(sequence_index, first_frame, last_frame)`` for each sequence."""
    ranges: list[tuple[int, int, int]] = []
    previous_end = 0
    for index, end in enumerate(end_frames, start=1):
        end = int(end)
        if end <= previous_end:
            raise SplitError(
                f"sequence_end_frames must be strictly increasing, got {end} after {previous_end}"
            )
        ranges.append((index, previous_end + 1, end))
        previous_end = end
    if not ranges:
        raise SplitError("sequence_end_frames is empty")
    return ranges


def build_sequence_mapping(end_frames: Sequence[int]) -> dict[str, str]:
    """Map every frame number (as its filename stem) to ``seq01`` ... ``seqNN``."""
    mapping: dict[str, str] = {}
    for index, first, last in sequence_ranges(end_frames):
        group = f"seq{index:02d}"
        for frame in range(first, last + 1):
            mapping[str(frame)] = group
    return mapping


def validate_sequence_mapping(mapping: Mapping[str, str], *, n_groups: int = 29) -> None:
    """Check completeness (frames 1..N once each), group count and group sizes."""
    if not mapping:
        raise SplitError("Frame mapping is empty")
    frames = sorted(int(case) for case in mapping)
    expected = list(range(1, frames[-1] + 1))
    if frames != expected:
        missing = sorted(set(expected) - set(frames))
        raise SplitError(f"Frame mapping is not contiguous from 1; missing frames: {missing[:20]}")
    if len(mapping) != len(frames):
        raise SplitError("Frame mapping contains duplicate case ids")

    groups = sorted(set(mapping.values()))
    if len(groups) != int(n_groups):
        raise SplitError(f"Expected {n_groups} sequences, found {len(groups)}: {groups[:5]}...")
    sizes: dict[str, int] = {group: 0 for group in groups}
    for group in mapping.values():
        sizes[group] += 1
    if min(sizes.values()) < 1:
        raise SplitError("A sequence has no frames")


def mapping_statistics(mapping: Mapping[str, str]) -> dict[str, object]:
    """Summary used for logging and the validation report."""
    sizes: dict[str, int] = {}
    for group in mapping.values():
        sizes[group] = sizes.get(group, 0) + 1
    return {
        "n_frames": len(mapping),
        "n_groups": len(sizes),
        "first_frame": min(int(case) for case in mapping),
        "last_frame": max(int(case) for case in mapping),
        "smallest_group": min(sizes, key=lambda group: sizes[group]),
        "smallest_group_frames": min(sizes.values()),
        "largest_group": max(sizes, key=lambda group: sizes[group]),
        "largest_group_frames": max(sizes.values()),
        "group_sizes": dict(sorted(sizes.items())),
    }


def check_mapping_against_stems(
    mapping: Mapping[str, str],
    stems: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Return ``(missing_from_mapping, mapped_but_not_on_disk)``."""
    on_disk = {str(stem) for stem in stems}
    mapped = set(mapping)
    missing = sorted(on_disk - mapped, key=case_sort_key)
    unknown = sorted(mapped - on_disk, key=case_sort_key)
    return missing, unknown


def write_mapping_csv(
    mapping: Mapping[str, str],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the ``case_id,group_id`` mapping (numeric order, UTF-8, LF)."""
    csv_path = Path(path)
    if csv_path.exists() and not overwrite:
        raise SplitError(
            f"Mapping file already exists: {csv_path}. Pass overwrite/--force to replace it."
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(mapping, key=case_sort_key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["case_id", "group_id"])
        for case in ordered:
            writer.writerow([case, mapping[case]])
    return csv_path

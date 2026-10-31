"""Group-level (video / patient) dataset splitting.

The previous code base split CVC-ClinicDB at *frame* level, so near-duplicate
frames from the same colonoscopy video ended up in both training and reported
splits.  Everything here works on *groups* (a video, a patient, a study) and the
result is validated before it is written: a group may only ever appear in one
split.

The frame -> group mapping is data, not code: it is supplied as a CSV
(``case_id,group_id``) whose path comes from the YAML config, because
CVC-ClinicDB filenames do not encode the source video.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "SPLITS",
    "SplitError",
    "SplitRecord",
    "actual_fractions",
    "assert_folds_cover_once",
    "assert_no_group_leakage",
    "contiguous_group_map",
    "filter_records",
    "folded_splits",
    "folds_summary",
    "group_folds",
    "grouped_split",
    "load_group_map",
    "read_split_csv",
    "split_file_for_fold",
    "split_summary",
    "validate_records",
    "write_split_csv",
]

SPLITS: tuple[str, ...] = ("train", "val", "test")


class SplitError(ValueError):
    """Raised when a split would leak cases between splits or is inconsistent."""


@dataclass(frozen=True)
class SplitRecord:
    """One case (frame/volume) and the split/fold it belongs to."""

    case_id: str
    group_id: str
    split: str
    fold: int = 0

    def as_row(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "group_id": self.group_id,
            "split": self.split,
            "fold": str(int(self.fold)),
        }


def load_group_map(
    path: str | Path,
    *,
    case_column: str = "case_id",
    group_column: str = "group_id",
) -> dict[str, str]:
    """Read a ``case_id -> group_id`` mapping CSV.

    A missing file is a hard error: silently falling back to frame-level splitting
    is exactly the leakage bug this project must not repeat.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise SplitError(
            f"Group mapping CSV not found: {csv_path}\n"
            "CVC-ClinicDB frame numbers do not encode the source video, so the "
            "frame -> video mapping must be supplied as data. Create a CSV with "
            f"the columns '{case_column},{group_column}' (see docs/splits.md)."
        )

    mapping: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {case_column, group_column} - set(reader.fieldnames or [])
        if missing:
            raise SplitError(
                f"{csv_path} is missing column(s) {sorted(missing)}; "
                f"found {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            case = str(row[case_column]).strip()
            group = str(row[group_column]).strip()
            if not case or not group:
                raise SplitError(f"{csv_path}:{line_number}: empty case_id or group_id")
            if case in mapping and mapping[case] != group:
                raise SplitError(
                    f"{csv_path}:{line_number}: case '{case}' is mapped to both "
                    f"'{mapping[case]}' and '{group}'"
                )
            mapping[case] = group
    if not mapping:
        raise SplitError(f"Group mapping CSV is empty: {csv_path}")
    return mapping


def contiguous_group_map(
    case_ids: Sequence[str],
    block_size: int,
    *,
    prefix: str = "block",
) -> dict[str, str]:
    """Group consecutive cases in blocks of ``block_size``.

    This is only a *proxy* for video identity (CVC-ClinicDB frames are stored in
    acquisition order) and must be declared as such in any experiment that uses
    it.  Prefer an explicit mapping whenever the video identity is known.
    """
    if int(block_size) < 1:
        raise SplitError(f"block_size must be >= 1, got {block_size}")
    ordered = sorted({str(case) for case in case_ids}, key=case_sort_key)
    return {
        case: f"{prefix}_{index // int(block_size):03d}" for index, case in enumerate(ordered)
    }


def case_sort_key(case_id: str):
    """Sort ``1.tif``, ``2.tif``, ``10.tif`` numerically (fallback: lexical)."""
    stem = Path(str(case_id)).stem
    try:
        return (0, int(stem), "")
    except ValueError:
        return (1, 0, stem)


def _allocate_groups(n_groups: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of ``n_groups`` groups to splits."""
    targets = {name: float(fractions[name]) * n_groups for name in SPLITS}
    counts = {name: int(np.floor(targets[name])) for name in SPLITS}
    remaining = n_groups - sum(counts.values())
    by_remainder = sorted(
        SPLITS, key=lambda name: (targets[name] - counts[name], -SPLITS.index(name)), reverse=True
    )
    for name in by_remainder[:remaining]:
        counts[name] += 1

    # Every split must receive at least one group, otherwise it cannot be
    # evaluated; borrow from the largest split (train in practice).
    for name in ("val", "test", "train"):
        if counts[name] == 0:
            donor = max(SPLITS, key=lambda candidate: counts[candidate])
            if counts[donor] <= 1:
                raise SplitError(
                    f"Cannot allocate at least one group per split from {n_groups} groups"
                )
            counts[donor] -= 1
            counts[name] += 1
    return counts


def grouped_split(
    case_ids: Iterable[str],
    group_map: Mapping[str, str],
    fractions: Mapping[str, float] | None = None,
    seed: int = 0,
) -> list[SplitRecord]:
    """Assign whole groups to train/val/test.

    The algorithm is deterministic for a given ``seed``:

    1. group ids are sorted, then permuted with ``numpy.random.default_rng(seed)``;
    2. the number of groups per split follows a largest-remainder allocation;
    3. groups are handed out in permuted order (train, then val, then test).

    Parameters
    ----------
    case_ids:
        All cases that must be assigned (typically the files found on disk).
    group_map:
        ``case_id -> group_id`` mapping (see :func:`load_group_map`).
    fractions:
        Target fractions of *groups* (not frames) per split.
    seed:
        Split seed, independent from the training seed.
    """
    fractions = fractions or {"train": 0.7, "val": 0.15, "test": 0.15}
    missing_keys = set(SPLITS) - set(fractions)
    if missing_keys:
        raise SplitError(f"fractions is missing split(s) {sorted(missing_keys)}")
    total = float(sum(float(fractions[name]) for name in SPLITS))
    if abs(total - 1.0) > 1e-6:
        raise SplitError(f"fractions must sum to 1.0, got {total}")

    cases = sorted({str(case) for case in case_ids}, key=case_sort_key)
    if not cases:
        raise SplitError("No cases to split")

    unmapped = [case for case in cases if str(case) not in group_map]
    if unmapped:
        preview = ", ".join(unmapped[:5])
        raise SplitError(
            f"{len(unmapped)} case(s) have no group mapping (first: {preview}). "
            "Provide a complete frame -> video mapping CSV; do not fall back to "
            "frame-level splitting."
        )

    groups: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        groups[str(group_map[case])].append(case)

    group_ids = sorted(groups)
    if len(group_ids) < len(SPLITS):
        raise SplitError(
            f"Only {len(group_ids)} group(s) available but {len(SPLITS)} splits are required"
        )

    counts = _allocate_groups(len(group_ids), fractions)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(group_ids))

    assignment: dict[str, str] = {}
    cursor = 0
    for split in SPLITS:
        for index in order[cursor : cursor + counts[split]]:
            assignment[group_ids[int(index)]] = split
        cursor += counts[split]

    records = [
        SplitRecord(
            case_id=case,
            group_id=str(group_map[case]),
            split=assignment[str(group_map[case])],
        )
        for case in cases
    ]
    validate_records(records)
    assert_no_group_leakage(records)
    return records


def assert_no_group_leakage(records: Iterable[SplitRecord]) -> None:
    """Raise :class:`SplitError` if any group spans more than one split."""
    seen: dict[str, set[str]] = defaultdict(set)
    for record in records:
        seen[record.group_id].add(record.split)
    leaks = {group: sorted(splits) for group, splits in seen.items() if len(splits) > 1}
    if leaks:
        preview = dict(list(sorted(leaks.items()))[:5])
        raise SplitError(
            f"{len(leaks)} group(s) appear in more than one split: {preview}. "
            "This leaks near-duplicate frames between train and evaluation."
        )


def validate_records(records: Iterable[SplitRecord]) -> None:
    """Check split names, identifiers, fold ids and duplicate cases."""
    seen_cases: dict[str, str] = {}
    for record in records:
        if record.split not in SPLITS:
            raise SplitError(f"Unknown split {record.split!r} for case {record.case_id!r}")
        if not record.case_id or not record.group_id:
            raise SplitError("Split records need non-empty case_id and group_id")
        if isinstance(record.fold, bool) or not isinstance(record.fold, int) or record.fold < 0:
            raise SplitError(f"Fold must be a non-negative integer, got {record.fold!r}")
        previous = seen_cases.get(record.case_id)
        if previous is not None and previous != record.split:
            raise SplitError(
                f"Case {record.case_id!r} assigned to both {previous!r} and {record.split!r}"
            )
        seen_cases.setdefault(record.case_id, record.split)


def split_summary(records: Iterable[SplitRecord]) -> dict[str, dict[str, int]]:
    """Number of cases and groups per split."""
    summary: dict[str, dict[str, int]] = {split: {"cases": 0, "groups": 0} for split in SPLITS}
    groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    for record in records:
        summary[record.split]["cases"] += 1
        groups[record.split].add(record.group_id)
    for split in SPLITS:
        summary[split]["groups"] = len(groups[split])
    return summary


def actual_fractions(records: Iterable[SplitRecord]) -> dict[str, float]:
    """Fraction of *cases* per split (for reporting next to the group targets)."""
    counts = {split: 0 for split in SPLITS}
    total = 0
    for record in records:
        counts[record.split] += 1
        total += 1
    if total == 0:
        return {split: float("nan") for split in SPLITS}
    frac = {split: counts[split] / total for split in SPLITS}
    return frac


def filter_records(records: Iterable[SplitRecord], split: str) -> list[SplitRecord]:
    """Return only the records of one split, in sorted case order."""
    if split not in SPLITS:
        raise SplitError(f"Unknown split {split!r}; expected one of {list(SPLITS)}")
    selected = [record for record in records if record.split == split]
    return sorted(selected, key=lambda record: case_sort_key(record.case_id))


def write_split_csv(
    records: Sequence[SplitRecord],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write ``case_id,group_id,split`` rows; refuses to clobber by default."""
    csv_path = Path(path)
    if csv_path.exists() and not overwrite:
        raise SplitError(
            f"Split file already exists: {csv_path}. Pass overwrite=True/--force to replace it."
        )
    validate_records(records)
    assert_no_group_leakage(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        records,
        key=lambda record: (
            int(record.fold),
            SPLITS.index(record.split),
            case_sort_key(record.case_id),
        ),
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "group_id", "split", "fold"])
        writer.writeheader()
        for record in ordered:
            writer.writerow(record.as_row())
    return csv_path


def read_split_csv(path: str | Path) -> list[SplitRecord]:
    """Read and validate a split CSV written by :func:`write_split_csv`."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise SplitError(f"Split file not found: {csv_path}. Run scripts/make_splits.py first.")
    records: list[SplitRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "group_id", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SplitError(f"{csv_path} is missing column(s) {sorted(missing)}")
        for row in reader:
            fold_text = str(row.get("fold", "") or "0").strip()
            try:
                fold = int(fold_text)
            except ValueError as error:
                raise SplitError(f"{csv_path}: invalid fold value {fold_text!r}") from error
            records.append(
                SplitRecord(
                    case_id=str(row["case_id"]).strip(),
                    group_id=str(row["group_id"]).strip(),
                    split=str(row["split"]).strip(),
                    fold=fold,
                )
            )
    if not records:
        raise SplitError(f"Split file is empty: {csv_path}")
    validate_records(records)
    assert_no_group_leakage(records)
    return records


def group_folds(
    group_ids: Iterable[str],
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, int]:
    """Assign every group to exactly one fold (deterministic given ``seed``)."""
    n_folds = int(n_folds)
    if n_folds < 3:
        raise SplitError(
            f"n_folds must be at least 3 so that train/val/test all exist, got {n_folds}"
        )
    identifiers = sorted({str(group) for group in group_ids})
    if len(identifiers) < n_folds:
        raise SplitError(f"Cannot build {n_folds} folds from {len(identifiers)} group(s)")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(identifiers))
    base, remainder = divmod(len(identifiers), n_folds)
    sizes = [base + (1 if fold < remainder else 0) for fold in range(n_folds)]

    assignment: dict[str, int] = {}
    cursor = 0
    for fold, size in enumerate(sizes):
        for index in order[cursor : cursor + size]:
            assignment[identifiers[int(index)]] = fold
        cursor += size
    return assignment


def folded_splits(
    case_ids: Iterable[str],
    group_map: Mapping[str, str],
    n_folds: int = 5,
    seed: int = 0,
) -> dict[int, list[SplitRecord]]:
    """Build ``n_folds`` train/val/test splits in which each group is tested once.

    For fold ``k`` the test set is fold ``k``, the validation set is the next fold
    (rotating) and the training set is everything else.  Every group is therefore a
    test unit in exactly one fold, which makes the per-case results poolable across
    folds (see :func:`melba.stats.pool_folds`).
    """
    n_folds = int(n_folds)
    cases = sorted({str(case) for case in case_ids}, key=case_sort_key)
    if not cases:
        raise SplitError("No cases to split")
    unmapped = [case for case in cases if str(case) not in group_map]
    if unmapped:
        preview = ", ".join(unmapped[:5])
        raise SplitError(
            f"{len(unmapped)} case(s) have no group mapping (first: {preview}). "
            "Provide a complete frame -> video mapping CSV; do not fall back to "
            "frame-level splitting."
        )

    assignment = group_folds({str(group_map[case]) for case in cases}, n_folds, seed)
    folds: dict[int, list[SplitRecord]] = {}
    for fold in range(n_folds):
        test_fold = fold
        val_fold = (fold + 1) % n_folds
        records: list[SplitRecord] = []
        for case in cases:
            group = str(group_map[case])
            group_fold = assignment[group]
            if group_fold == test_fold:
                split = "test"
            elif group_fold == val_fold:
                split = "val"
            else:
                split = "train"
            records.append(SplitRecord(case_id=case, group_id=group, split=split, fold=fold))
        validate_records(records)
        assert_no_group_leakage(records)
        folds[fold] = records

    assert_folds_cover_once([record for records in folds.values() for record in records])
    return folds


def assert_folds_cover_once(records: Iterable[SplitRecord], *, split: str = "test") -> None:
    """Every case and every group must appear exactly once in the given split."""
    records = list(records)
    selected = [record for record in records if record.split == split]
    if not selected:
        raise SplitError(f"No '{split}' records found across folds")

    case_counts: dict[str, int] = {}
    group_folds: dict[str, set[int]] = {}
    for record in selected:
        case_counts[record.case_id] = case_counts.get(record.case_id, 0) + 1
        group_folds.setdefault(record.group_id, set()).add(int(record.fold))
    duplicated_cases = sorted(case for case, count in case_counts.items() if count > 1)
    # a group owns several frames, so it must be counted once per *fold*, not once
    # per record
    duplicated_groups = sorted(group for group, folds in group_folds.items() if len(folds) > 1)
    if duplicated_cases or duplicated_groups:
        raise SplitError(
            f"'{split}' contains duplicate cases ({duplicated_cases[:5]}) or groups "
            f"({duplicated_groups[:5]}) across folds; each must be evaluated exactly once"
        )

    all_cases = {record.case_id for record in records}
    missing = sorted(all_cases - set(case_counts))
    if missing:
        raise SplitError(
            f"{len(missing)} case(s) are never part of '{split}' (first: {missing[:5]})"
        )
    missing_groups = sorted({record.group_id for record in records} - set(group_folds))
    if missing_groups:
        raise SplitError(
            f"{len(missing_groups)} group(s) are never part of '{split}' "
            f"(first: {missing_groups[:5]})"
        )


def folds_summary(
    folds: Mapping[int, Sequence[SplitRecord]],
) -> dict[int, dict[str, dict[str, int]]]:
    """Per-fold case/group counts for every split."""
    return {int(fold): split_summary(records) for fold, records in sorted(folds.items())}


def split_file_for_fold(config: Any, fold: int) -> Path:
    """Resolve the split CSV for one fold from ``dataset.split.output_pattern``."""
    fold = int(fold)
    if fold < 0:
        raise SplitError(f"fold must be non-negative, got {fold}")
    pattern = config.get("dataset.split.output_pattern")
    if pattern:
        return config.resolve(str(pattern).format(fold=fold))
    return config.require_path("dataset.split.output_file")




"""Group-level splitting: determinism, no leakage, complete assignment."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from melba.data import (
    SplitError,
    SplitRecord,
    actual_fractions,
    assert_folds_cover_once,
    assert_no_group_leakage,
    contiguous_group_map,
    filter_records,
    folded_splits,
    group_folds,
    grouped_split,
    load_group_map,
    read_split_csv,
    split_file_for_fold,
    split_summary,
    write_split_csv,
)


def _synthetic(n_groups: int = 10, cases_per_group: int = 3):
    cases: list[str] = []
    group_map: dict[str, str] = {}
    for group in range(n_groups):
        for case in range(cases_per_group):
            case_id = f"case_{group:02d}_{case}"
            cases.append(case_id)
            group_map[case_id] = f"video_{group:02d}"
    return cases, group_map


FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


def test_split_is_deterministic_for_a_seed() -> None:
    cases, group_map = _synthetic()
    first = grouped_split(cases, group_map, FRACTIONS, seed=42)
    second = grouped_split(list(reversed(cases)), group_map, FRACTIONS, seed=42)
    assert first == second
    assert grouped_split(cases, group_map, FRACTIONS, seed=43) != first


def test_no_group_leakage_and_all_cases_assigned() -> None:
    cases, group_map = _synthetic()
    records = grouped_split(cases, group_map, FRACTIONS, seed=0)
    assert sorted(record.case_id for record in records) == sorted(cases)
    assert_no_group_leakage(records)

    groups_per_split = {split: set() for split in ("train", "val", "test")}
    for record in records:
        groups_per_split[record.split].add(record.group_id)
    assert not (groups_per_split["train"] & groups_per_split["val"])
    assert not (groups_per_split["train"] & groups_per_split["test"])
    assert not (groups_per_split["val"] & groups_per_split["test"])
    assert all(groups_per_split[split] for split in groups_per_split)


def test_group_counts_follow_fractions() -> None:
    cases, group_map = _synthetic(n_groups=20, cases_per_group=1)
    records = grouped_split(cases, group_map, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=5)
    summary = split_summary(records)
    assert summary["train"]["groups"] == 14
    assert summary["val"]["groups"] == 3
    assert summary["test"]["groups"] == 3
    assert summary["train"]["cases"] == 14
    assert actual_fractions(records)["train"] == pytest.approx(14 / 20)


def test_every_split_gets_at_least_one_group() -> None:
    cases, group_map = _synthetic(n_groups=3, cases_per_group=2)
    records = grouped_split(cases, group_map, {"train": 0.98, "val": 0.01, "test": 0.01}, seed=0)
    summary = split_summary(records)
    assert all(summary[split]["groups"] == 1 for split in ("train", "val", "test"))


def test_unmapped_cases_are_rejected() -> None:
    cases, group_map = _synthetic(n_groups=5, cases_per_group=1)
    incomplete = dict(group_map)
    del incomplete[cases[0]]
    with pytest.raises(SplitError, match="no group mapping"):
        grouped_split(cases, incomplete, FRACTIONS, seed=0)


def test_too_few_groups_are_rejected() -> None:
    with pytest.raises(SplitError, match="splits are required"):
        grouped_split(["a", "b"], {"a": "video", "b": "video"}, FRACTIONS, seed=0)


def test_bad_fractions_are_rejected() -> None:
    cases, group_map = _synthetic()
    with pytest.raises(SplitError, match="sum to 1.0"):
        grouped_split(cases, group_map, {"train": 0.8, "val": 0.2, "test": 0.2}, seed=0)
    with pytest.raises(SplitError, match="missing split"):
        grouped_split(cases, group_map, {"train": 0.8, "val": 0.2}, seed=0)


def test_leakage_detection() -> None:
    records = [SplitRecord("a", "video_1", "train"), SplitRecord("b", "video_1", "test")]
    with pytest.raises(SplitError, match="more than one split"):
        assert_no_group_leakage(records)


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    cases, group_map = _synthetic()
    records = grouped_split(cases, group_map, FRACTIONS, seed=3)
    path = tmp_path / "nested" / "split.csv"
    written = write_split_csv(records, path)
    loaded = read_split_csv(written)
    assert sorted(loaded, key=lambda record: (record.split, record.case_id)) == sorted(
        records, key=lambda record: (record.split, record.case_id)
    )
    assert_no_group_leakage(loaded)

    with pytest.raises(SplitError, match="already exists"):
        write_split_csv(records, written)
    write_split_csv(records, written, overwrite=True)


def test_read_detects_leakage_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "leaky.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "group_id", "split"])
        writer.writerow(["1", "video_1", "train"])
        writer.writerow(["2", "video_1", "test"])
    with pytest.raises(SplitError, match="more than one split"):
        read_split_csv(path)


def test_load_group_map(tmp_path: Path) -> None:
    path = tmp_path / "groups.csv"
    path.write_text("frame,video\n1.tif,v1\n2.tif,v1\n3.tif,v2\n", encoding="utf-8")
    mapping = load_group_map(path, case_column="frame", group_column="video")
    assert mapping == {"1.tif": "v1", "2.tif": "v1", "3.tif": "v2"}

    with pytest.raises(SplitError, match="not found"):
        load_group_map(tmp_path / "missing.csv")

    bad = tmp_path / "bad.csv"
    bad.write_text("frame,video\n1.tif,v1\n1.tif,v2\n", encoding="utf-8")
    with pytest.raises(SplitError, match="mapped to both"):
        load_group_map(bad, case_column="frame", group_column="video")


def test_contiguous_group_map_groups_sorted_numerically() -> None:
    cases = [f"{index}.tif" for index in range(10, 0, -1)]
    mapping = contiguous_group_map(cases, block_size=4)
    assert mapping["1.tif"] == "block_000"
    assert mapping["4.tif"] == "block_000"
    assert mapping["5.tif"] == "block_001"
    assert mapping["10.tif"] == "block_002"
    with pytest.raises(SplitError):
        contiguous_group_map(cases, block_size=0)


def test_filter_records() -> None:
    cases, group_map = _synthetic()
    records = grouped_split(cases, group_map, FRACTIONS, seed=0)
    test_records = filter_records(records, "test")
    assert test_records
    assert all(record.split == "test" for record in test_records)
    assert [record.case_id for record in test_records] == sorted(
        record.case_id for record in test_records
    )
    with pytest.raises(SplitError):
        filter_records(records, "holdout")


def test_group_folds_are_balanced_and_complete() -> None:
    cases, group_map = _synthetic(n_groups=29, cases_per_group=2)
    assignment = group_folds(group_map.values(), n_folds=5, seed=3)
    assert len(assignment) == 29
    sizes = sorted(
        sum(1 for fold in assignment.values() if fold == index) for index in range(5)
    )
    assert sizes == [5, 6, 6, 6, 6]
    assert assignment == group_folds(group_map.values(), n_folds=5, seed=3)
    assert assignment != group_folds(group_map.values(), n_folds=5, seed=4)


def test_folded_splits_cover_each_video_once_as_test() -> None:
    cases, group_map = _synthetic(n_groups=29, cases_per_group=2)
    folds = folded_splits(cases, group_map, n_folds=5, seed=3)
    assert sorted(folds) == [0, 1, 2, 3, 4]

    tested_groups: dict[str, set[int]] = {}
    for fold, records in folds.items():
        assert_no_group_leakage(records)
        assert {record.fold for record in records} == {fold}
        assert {record.split for record in records} == {"train", "val", "test"}
        assert sorted(record.case_id for record in records) == sorted(cases)
        for record in records:
            if record.split == "test":
                tested_groups.setdefault(record.group_id, set()).add(fold)

    assert len(tested_groups) == 29
    assert all(len(group_folds) == 1 for group_folds in tested_groups.values())
    # rotation: the validation groups of fold k are the test groups of fold k+1
    for fold in range(5):
        val_groups = {r.group_id for r in folds[fold] if r.split == "val"}
        next_test = {r.group_id for r in folds[(fold + 1) % 5] if r.split == "test"}
        assert val_groups == next_test


def test_folded_splits_write_and_reload(tmp_path: Path) -> None:
    cases, group_map = _synthetic(n_groups=6, cases_per_group=2)
    folds = folded_splits(cases, group_map, n_folds=3, seed=0)
    for fold, records in folds.items():
        path = write_split_csv(records, tmp_path / f"fold{fold}.csv")
        loaded = read_split_csv(path)
        assert {record.fold for record in loaded} == {fold}
        assert_no_group_leakage(loaded)


def test_folded_split_requirements() -> None:
    cases, group_map = _synthetic(n_groups=6, cases_per_group=1)
    with pytest.raises(SplitError, match="at least 3"):
        folded_splits(cases, group_map, n_folds=2, seed=0)
    with pytest.raises(SplitError, match="Cannot build"):
        folded_splits(cases, group_map, n_folds=10, seed=0)


def test_assert_folds_cover_once_detects_duplicates() -> None:
    records = [
        SplitRecord("a", "video_1", "test", fold=0),
        SplitRecord("b", "video_1", "test", fold=1),
    ]
    with pytest.raises(SplitError, match="duplicate"):
        assert_folds_cover_once(records)
    with pytest.raises(SplitError, match="No 'test' records"):
        assert_folds_cover_once([SplitRecord("a", "video_1", "train", fold=0)])


def test_split_file_for_fold(base_config) -> None:
    path = split_file_for_fold(base_config, 3)
    assert path.name == "cvc_clinicdb_fold3.csv"
    with pytest.raises(SplitError):
        split_file_for_fold(base_config, -1)


"""CVC-ClinicDB frame -> sequence mapping built from the official table."""

from __future__ import annotations

from pathlib import Path

import pytest

from melba.data import (
    build_sequence_mapping,
    check_mapping_against_stems,
    mapping_statistics,
    sequence_ranges,
    validate_sequence_mapping,
    write_mapping_csv,
)
from melba.data.splits import SplitError


def official_end_frames(base_config) -> list[int]:
    return [int(value) for value in base_config["dataset.split.sequence_end_frames"]]


def test_sequence_ranges_cover_frames_contiguously(base_config) -> None:
    end_frames = official_end_frames(base_config)
    ranges = sequence_ranges(end_frames)
    assert len(ranges) == 29
    assert ranges[0] == (1, 1, 25)
    assert ranges[9] == (10, 200, 205)  # smallest sequence: 6 frames
    assert ranges[-1] == (29, 592, 612)
    for (_, _, previous_end), (_, start, _) in zip(ranges, ranges[1:]):
        assert start == previous_end + 1


def test_mapping_has_612_frames_in_29_sequences(base_config) -> None:
    end_frames = official_end_frames(base_config)
    mapping = build_sequence_mapping(end_frames)
    validate_sequence_mapping(mapping, n_groups=29)

    stats = mapping_statistics(mapping)
    assert stats["n_frames"] == 612
    assert stats["n_groups"] == 29
    assert stats["first_frame"] == 1
    assert stats["last_frame"] == 612
    assert stats["smallest_group"] == "seq10"
    assert stats["smallest_group_frames"] == 6
    assert stats["largest_group"] == "seq08"
    assert stats["largest_group_frames"] == 26
    assert sorted(mapping) == sorted(str(frame) for frame in range(1, 613))
    assert mapping["1"] == "seq01"
    assert mapping["25"] == "seq01"
    assert mapping["26"] == "seq02"
    assert mapping["612"] == "seq29"


def test_invalid_tables_are_rejected() -> None:
    with pytest.raises(SplitError, match="increasing"):
        sequence_ranges([10, 10])
    with pytest.raises(SplitError, match="empty"):
        build_sequence_mapping([])
    mapping = build_sequence_mapping([3, 5])
    with pytest.raises(SplitError, match="Expected 29 sequences"):
        validate_sequence_mapping(mapping)
    with pytest.raises(SplitError, match="not contiguous"):
        validate_sequence_mapping({str(frame): "seq01" for frame in (1, 3)})


def test_mapping_against_stems(tmp_path: Path, base_config) -> None:
    mapping = build_sequence_mapping(official_end_frames(base_config))
    stems = [str(frame) for frame in range(1, 613)]
    assert check_mapping_against_stems(mapping, stems) == ([], [])

    missing, unknown = check_mapping_against_stems(mapping, stems[:-1] + ["613"])
    assert missing == ["613"]  # on disk but not mapped
    assert unknown == ["612"]  # mapped but missing on disk

    path = write_mapping_csv(mapping, tmp_path / "nested" / "groups.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "case_id,group_id"
    assert len(lines) == 613
    assert lines[1] == "1,seq01"
    assert lines[-1] == "612,seq29"
    with pytest.raises(SplitError, match="already exists"):
        write_mapping_csv(mapping, path)

"""Long-format result records: schema, streaming writer, summary, provenance."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from melba.eval import (
    RESULT_COLUMNS,
    LongResultWriter,
    RecordError,
    make_run_id,
    read_long_csv,
    summarize_long,
    write_json,
)


def _row(**overrides):
    row = {column: "" for column in RESULT_COLUMNS}
    row.update(
        {
            "run_id": "exp-abc-seed0",
            "created_utc": "2026-01-01T00:00:00Z",
            "config_path": "configs/x.yaml",
            "config_hash": "abc",
            "seed": 0,
            "dataset": "cvc_clinicdb",
            "model": "smp_unet",
            "split": "test",
            "fold": 0,
            "method": "baseline",
            "aggregator": "mean",
            "n_views": 1,
            "view_names": "identity",
            "threshold": 0.5,
            "spacing_mm": "1,1",
            "case_id": "case_1",
            "group_id": "video_1",
            "empty_flag": "both_nonempty",
            "missed_detection": 0,
            "metric": "dice",
            "value": 0.9,
        }
    )
    row.update(overrides)
    return row


def test_writer_and_reader_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "per_case.csv"
    with LongResultWriter(path) as writer:
        writer.write(_row(metric="dice", value=0.9))
        writer.write(_row(metric="hd95", value=float("nan")))
        writer.write(_row(case_id="case_2", metric="dice", value=0.8))
        assert writer.n_rows == 3

    frame = pd.read_csv(path)
    assert frame.columns.tolist() == list(RESULT_COLUMNS)
    assert len(frame) == 3
    assert math.isnan(frame.loc[frame["metric"] == "hd95", "value"].iloc[0])

    loaded = read_long_csv(path)
    summary = summarize_long(loaded)
    assert {"method", "metric", "n", "mean", "std", "median", "q25", "q75", "n_cases"} <= set(
        summary.columns
    )
    assert sorted(summary["n_cases"].tolist()) == [1, 2]


def test_writer_rejects_bad_records(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    with LongResultWriter(path) as writer:
        with pytest.raises(RecordError, match="unknown column"):
            writer.write({**_row(), "extra": 1})
        incomplete = _row()
        del incomplete["metric"]
        with pytest.raises(RecordError, match="missing column"):
            writer.write(incomplete)


def test_append_mode(tmp_path: Path) -> None:
    path = tmp_path / "append.csv"
    with LongResultWriter(path) as writer:
        writer.write(_row())
    with LongResultWriter(path, append=True) as writer:
        writer.write(_row(case_id="case_2"))
    assert len(pd.read_csv(path)) == 2


def test_read_long_csv_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_long_csv(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"case_id": ["a"], "value": [1.0]}).to_csv(bad, index=False)
    with pytest.raises(RecordError, match="missing"):
        read_long_csv(bad)


def test_make_run_id_is_deterministic(base_config) -> None:
    first = make_run_id(base_config)
    assert first == make_run_id(base_config)
    assert base_config["experiment.name"] in first
    assert base_config.fingerprint in first
    assert first.endswith(f"seed{base_config['seed']}")
    assert make_run_id(base_config.with_overrides({"seed": 5})).endswith("seed5")


def test_write_json(tmp_path: Path) -> None:
    path = write_json(tmp_path / "nested" / "run.json", {"b": 1, "a": [1, 2]})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"a": [1, 2], "b": 1}

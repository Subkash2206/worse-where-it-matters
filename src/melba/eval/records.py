"""Long-format per-case result records.

Every evaluation writes one row per ``(case, method, metric)`` — never only
aggregated means.  Summary files are written *in addition* and always carry the
case count so that a mean can never be mistaken for the evidence.

Schema::

    run_id, created_utc, config_path, config_hash, seed, dataset, model, split,
    method, aggregator, n_views, view_names, threshold, spacing_mm, case_id,
    group_id, metric, value
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "RESULT_COLUMNS",
    "LongResultWriter",
    "RecordError",
    "make_run_id",
    "read_long_csv",
    "summarize_long",
    "utc_now",
    "write_json",
]

RESULT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "created_utc",
    "config_path",
    "config_hash",
    "seed",
    "dataset",
    "model",
    "split",
    "fold",
    "method",
    "aggregator",
    "n_views",
    "view_names",
    "threshold",
    "spacing_mm",
    "case_id",
    "group_id",
    "empty_flag",
    "metric",
    "value",
)


class RecordError(ValueError):
    """Raised when a result record does not match the long-format schema."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(config: Any) -> str:
    """Deterministic identifier for a configuration + seed combination."""
    return f"{config.require('experiment.name')}-{config.fingerprint}-seed{int(config.require('seed'))}"


def _format(value: Any) -> Any:
    """CSV-safe representation (``nan`` becomes an empty field)."""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return repr(value) if value not in (float("inf"), float("-inf")) else str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return value


class LongResultWriter:
    """Streaming writer for the long-format results CSV."""

    def __init__(
        self,
        path: str | Path,
        *,
        columns: Sequence[str] = RESULT_COLUMNS,
        append: bool = False,
    ) -> None:
        self.path = Path(path)
        self.columns = tuple(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.path.exists() and self.path.stat().st_size > 0
        mode = "a" if append and existing else "w"
        self._handle = self.path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(self.columns))
        if mode == "w":
            self._writer.writeheader()
        self._rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        unknown = sorted(set(row) - set(self.columns))
        if unknown:
            raise RecordError(f"Record contains unknown column(s): {unknown}")
        missing = [column for column in self.columns if column not in row]
        if missing:
            raise RecordError(f"Record is missing column(s): {missing}")
        self._writer.writerow({key: _format(row[key]) for key in self.columns})
        self._rows += 1

    def write_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            self.write(row)

    @property
    def n_rows(self) -> int:
        return self._rows

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> "LongResultWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def read_long_csv(path: str | Path) -> pd.DataFrame:
    """Read a long-format CSV, validating that the required columns exist."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Result CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    missing = [column for column in ("case_id", "method", "metric", "value") if column not in frame]
    if missing:
        raise RecordError(f"{csv_path} is not a long-format result file; missing {missing}")
    return frame


def summarize_long(
    frame: pd.DataFrame,
    *,
    by: Sequence[str] = ("method", "metric"),
    value: str = "value",
) -> pd.DataFrame:
    """Descriptive statistics for the summary CSV (per-case file stays untouched)."""
    missing = [column for column in (*by, value) if column not in frame]
    if missing:
        raise RecordError(f"Cannot summarise: missing column(s) {missing}")
    grouped = frame.groupby(list(by), dropna=False)[value]
    summary = grouped.agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        q25=lambda series: series.quantile(0.25),
        q75=lambda series: series.quantile(0.75),
        min="min",
        max="max",
    ).reset_index()
    counts = frame.groupby(list(by), dropna=False)["case_id"].nunique().reset_index(name="n_cases")
    return summary.merge(counts, on=list(by), how="left")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write provenance metadata next to the results."""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return json_path

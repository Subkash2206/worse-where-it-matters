"""Deployment terms: what TTA-4/TTA-8 buy and cost per 1000 frames (or patients).

For every dataset and model this script converts the aggregated tables into per-1000 numbers, so the
benefit and the price sit in one row:

* **large boundary failures fixed** - the drop in the HD95 failure rate (``tail_metrics.csv``,
  ``delta_hd95_failure_rate_estimate``) times 1000; negative means TTA creates failures;
* **misses added** - the net non-empty -> empty transitions per case (``detection`` tables
  internally, the majority-of-checkpoints counts externally) times 1000;
* **false positives** - only BraTS has empty references, so only BraTS fills these two columns, and
  they are **raw counts on the 33 lesion-free patients** rather than a rate: how many of those 33
  patients still receive a false-positive prediction under the method
  (``false_positive_seed_averaged.csv``, ``n_false_positive``), and the net fp -> correct minus
  correct -> fp count TTA moves;
* **absolute latency** - forward passes per image/volume (baseline 1, TTA-4 4, TTA-8 8) times the
  measured latency per pass.  The per-run JSON records that timing exists but not the values (they
  went to the console), so the two measured constants of this project are used: 45 ms per pass for
  the 2-D polyp models at 352x352 and 1750 ms per pass for the 3-D SegResNet at 128^3.  Latency is
  reported **in ms per frame for the 2-D models and in s per volume for BraTS**, and the
  ``latency_is_estimated`` column states that it is an estimate (the per-pass constant times the
  number of passes), not a measured end-to-end wall clock.

Usage
-----
    python scripts/deployment_terms.py            # writes results/deployment_terms.{csv,md}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = _REPO_ROOT / "results"
OUTPUT = RESULTS / "deployment_terms"
LATENCY_MS_PER_PASS = {"2d": 45.0, "3d": 1750.0}
EXTRA_PASSES = {"tta4": 3, "tta8": 7}  # the identity view is the baseline pass


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _tail_rows(directory: Path, key: str, kind: str) -> list[dict[str, Any]]:
    """Large-boundary failures fixed, misses added and the lesion-free false positives.

    The baseline row is the reference of the table: zero extra passes, no transition, no failure
    removed - but it does carry its own false-positive count on the lesion-free patients, which is
    what the TTA rows are compared against.
    """
    rows: list[dict[str, Any]] = []
    tail = _read(directory / "tail_metrics.csv")
    detection = _read(directory / "detection_seed_averaged.csv")
    false_positive = _read(directory / "false_positive_seed_averaged.csv")
    for method in ("baseline", "tta4", "tta8"):
        fixed = added = 0.0 if method == "baseline" else np.nan
        false_positives_removed = np.nan
        if method != "baseline":
            if not tail.empty and "delta_hd95_failure_rate_estimate" in tail:
                row = tail.loc[tail["method"] == method]
                if not row.empty:
                    fixed = -float(row["delta_hd95_failure_rate_estimate"].iloc[0]) * 1000.0
            if not detection.empty and method in set(detection["method"]):
                row = detection.loc[detection["method"] == method].iloc[0]
                net = float(row.get("nonempty_to_empty", np.nan)) - float(
                    row.get("empty_to_nonempty", np.nan)
                )
                added = net * 1000.0 / max(float(row["n_cases"]), 1.0)
        # raw counts on the lesion-free patients (BraTS is the only collection with empty references)
        lesion_free = np.nan
        false_positives = np.nan
        if not false_positive.empty and method in set(false_positive["method"]):
            row = false_positive.loc[false_positive["method"] == method].iloc[0]
            lesion_free = float(row.get("n_cases", np.nan))
            false_positives = float(row.get("n_false_positive", np.nan))
            if method != "baseline":
                false_positives_removed = float(row.get("fp_to_correct", np.nan)) - float(
                    row.get("correct_to_fp", np.nan)
                )
            else:
                false_positives_removed = 0.0
        pass_ms = LATENCY_MS_PER_PASS[kind]
        passes = 1 + EXTRA_PASSES.get(method, 0)
        latency_ms = passes * pass_ms
        rows.append(
            {
                "dataset": key,
                "unit": "patients" if kind == "3d" else "frames",
                "method": method,
                "boundary_failures_fixed_per_1000": fixed,
                "misses_added_per_1000": added,
                "lesion_free_patients": lesion_free,
                "false_positives_on_lesion_free": false_positives,
                "false_positives_removed": false_positives_removed,
                "forward_passes": passes,
                "latency_ms_per_frame": latency_ms if kind == "2d" else np.nan,
                "latency_s_per_volume": latency_ms / 1000.0 if kind == "3d" else np.nan,
                "latency_source": f"{pass_ms:.0f} ms per pass "
                + ("(3-D SegResNet 128^3)" if kind == "3d" else "(2-D, 352x352)"),
                "latency_is_estimated": True,
            }
        )
    return rows


def _external_rows() -> list[dict[str, Any]]:
    """Same terms for the external collections, from results/external_by_model."""
    output = RESULTS / "external_by_model"
    misses = _read(output / "misses.csv")
    summary = _read(output / "subgroup_summary.csv")
    rows: list[dict[str, Any]] = []
    if misses.empty:
        return rows
    for (dataset, family), block in misses.groupby(["dataset", "family"]):
        baseline = block.loc[block["method"] == "baseline"]
        if baseline.empty:
            continue
        total = float(baseline["n_cases"].iloc[0])
        for method in ("baseline", "tta4", "tta8"):
            row = block.loc[block["method"] == method]
            if row.empty:
                continue
            row = row.iloc[0]
            if method == "baseline":
                added = 0.0
            else:
                added = (
                    float(row["cases_became_majority_miss"]) - float(row["cases_left_majority_miss"])
                ) * 1000.0 / total
            # The external tables carry no HD95 failure-rate estimate (the internal aggregate
            # writes one), and the pixel scale differs per collection, so a per-1000 failure count
            # would not be comparable.  Left empty on purpose; the Dice/HD95 deltas are in
            # results/external_by_model.  None of these collections has empty references either,
            # so the false-positive columns stay empty.
            fixed = 0.0 if method == "baseline" else np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "unit": "frames",
                    "method": method,
                    "boundary_failures_fixed_per_1000": fixed,
                    "misses_added_per_1000": added,
                    "lesion_free_patients": np.nan,
                    "false_positives_on_lesion_free": np.nan,
                    "false_positives_removed": np.nan,
                    "forward_passes": 1 + EXTRA_PASSES.get(method, 0),
                    "latency_ms_per_frame": (1 + EXTRA_PASSES.get(method, 0))
                    * LATENCY_MS_PER_PASS["2d"],
                    "latency_s_per_volume": np.nan,
                    "latency_source": f"{LATENCY_MS_PER_PASS['2d']:.0f} ms per pass (2-D, 352x352)"
                    + ("" if family == "unet_resnet50" else ", SwinUNETR"),
                    "latency_is_estimated": True,
                    "note": "no HD95 failure-rate estimate in the external tables",
                }
            )
    return rows


def _number(value: float, *, signed: bool = True, decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"


def _fp_cell(row: pd.Series) -> str:
    """``20 of 33`` for BraTS, empty for the collections without lesion-free patients."""
    count = row["false_positives_on_lesion_free"]
    total = row["lesion_free_patients"]
    if not np.isfinite(count) or not np.isfinite(total):
        return ""
    return f"{int(count)} of {int(total)}"


def _latency_cell(row: pd.Series) -> str:
    """Absolute latency per image/volume, in the unit of that collection."""
    if np.isfinite(row["latency_s_per_volume"]):
        return f"{float(row['latency_s_per_volume']):.2f} s per volume"
    if np.isfinite(row["latency_ms_per_frame"]):
        return f"{float(row['latency_ms_per_frame']):.0f} ms per frame"
    return ""


def main() -> int:
    rows: list[dict[str, Any]] = []
    for key, directory, kind in (
        ("cvc_unet", RESULTS / "aggregate_cvc_clinicdb_unet_resnet50", "2d"),
        ("cvc_swin", RESULTS / "aggregate_cvc_clinicdb_swinunetr_2d", "2d"),
        ("brats", RESULTS / "aggregate_brats2021_segresnet_reduced", "3d"),
    ):
        if directory.is_dir():
            rows.extend(_tail_rows(directory, key, kind))
    rows.extend(_external_rows())
    table = pd.DataFrame(rows)
    if table.empty:
        print("no aggregate tables found", file=sys.stderr)
        return 2
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "deployment_terms.csv", index=False)
    lines = ["# Deployment terms per 1000 frames / patients", ""]
    lines.append(
        "Positive *fixed* and *removed* columns are benefits, positive *misses added* is a cost.  The "
        "false positives are **raw counts on the 33 lesion-free BraTS patients** (only BraTS has "
        "empty references), not a rate.  The latency column is the **absolute** cost of one "
        "image/volume (forward passes x measured per-pass latency), and it is **estimated**, not a "
        "measured end-to-end wall clock - see the ``passes`` and ``estimated`` columns and the note "
        "below.\n"
    )
    lines.extend(
        [
            "| dataset | unit | method | passes | boundary failures fixed | misses added | "
            "false positives on lesion-free | false positives removed | latency (absolute) | "
            "latency estimated |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in table.sort_values(["dataset", "method"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {row['unit']} | {row['method']} | "
            f"{int(row['forward_passes'])} | "
            f"{_number(row['boundary_failures_fixed_per_1000'])} | "
            f"{_number(row['misses_added_per_1000'])} | "
            f"{_fp_cell(row)} | "
            f"{_number(row['false_positives_removed'], decimals=0)} | "
            f"{_latency_cell(row)} | "
            f"{'yes' if bool(row['latency_is_estimated']) else 'no'} |"
        )
    lines.append("")
    lines.append(
        "Latency is the absolute per-image/per-volume cost (forward passes x per-pass latency) and "
        "is estimated from the measured per-pass constant ("
        + "; ".join(table["latency_source"].drop_duplicates().tolist())
        + "), not measured end to end. The 3-D value is reported in s per volume, the 2-D values in "
        "ms per frame."
    )
    (OUTPUT / "deployment_terms.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT / 'deployment_terms.csv'} ({len(table)} rows)")
    print(f"wrote {OUTPUT / 'deployment_terms.md'}")
    brats = table.loc[table["dataset"] == "brats"].sort_values("method")
    if not brats.empty:
        counts = ", ".join(
            f"{row['method']} {int(row['false_positives_on_lesion_free'])}"
            for _, row in brats.iterrows()
            if np.isfinite(row["false_positives_on_lesion_free"])
        )
        totals = {
            int(row["lesion_free_patients"])
            for _, row in brats.iterrows()
            if np.isfinite(row["lesion_free_patients"])
        }
        print(
            f"BraTS false positives on the {sorted(totals)[0] if totals else '?'} lesion-free "
            f"patients: {counts}"
        )
    for _, row in table.sort_values(["dataset", "method"]).iterrows():
        print(
            f"  {row['dataset']:12s} {row['method']:8s} {int(row['forward_passes'])} pass(es) "
            f"latency {_latency_cell(row):18s} "
            f"FP on lesion-free {_fp_cell(row) or '-':8s} removed "
            f"{_number(row['false_positives_removed'], decimals=0) or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


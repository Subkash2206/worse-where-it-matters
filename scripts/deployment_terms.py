"""Deployment terms: what TTA-4/TTA-8 buy and cost per 1000 frames (or patients).

For every dataset and model this script converts the aggregated tables into per-1000 numbers, so the
benefit and the price sit in one row:

* **large boundary failures fixed** - the drop in the HD95 failure rate (``tail_metrics.csv``,
  ``delta_hd95_failure_rate_estimate``) times 1000; negative means TTA creates failures;
* **misses added** - the net non-empty -> empty transitions per case (``detection`` tables
  internally, the majority-of-checkpoints counts externally) times 1000;
* **false positives removed** - only BraTS has empty references, so only BraTS fills this column
  (``false_positive_seed_averaged.csv``, fp -> correct minus correct -> fp, times 1000);
* **extra ms per frame** - the additional forward passes times the measured latency per pass.  The
  per-run JSON records that timing exists but not the values (they went to the console), so the two
  measured constants of this project are used and printed in the ``latency_source`` column: 45 ms per
  pass for the 2-D polyp models at 352x352, 1750 ms per pass for the 3-D SegResNet at 128^3.

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
    """Large-boundary failures fixed and misses added, from one aggregate directory."""
    rows: list[dict[str, Any]] = []
    tail = _read(directory / "tail_metrics.csv")
    detection = _read(directory / "detection_seed_averaged.csv")
    false_positive = _read(directory / "false_positive_seed_averaged.csv")
    for method in ("tta4", "tta8"):
        fixed = np.nan
        if not tail.empty and "delta_hd95_failure_rate_estimate" in tail:
            row = tail.loc[tail["method"] == method]
            if not row.empty:
                fixed = -float(row["delta_hd95_failure_rate_estimate"].iloc[0]) * 1000.0
        added = np.nan
        if not detection.empty and method in set(detection["method"]):
            row = detection.loc[detection["method"] == method].iloc[0]
            net = float(row.get("nonempty_to_empty", np.nan)) - float(row.get("empty_to_nonempty", np.nan))
            added = net * 1000.0 / max(float(row["n_cases"]), 1.0)
        fp_removed = np.nan
        if not false_positive.empty and method in set(false_positive["method"]):
            row = false_positive.loc[false_positive["method"] == method].iloc[0]
            removed = float(row.get("fp_to_correct", np.nan)) - float(row.get("correct_to_fp", np.nan))
            fp_removed = removed * 1000.0 / max(float(row["n_cases"]), 1.0)
        rows.append(
            {
                "dataset": key,
                "unit": "patients" if kind == "3d" else "frames",
                "method": method,
                "boundary_failures_fixed_per_1000": fixed,
                "misses_added_per_1000": added,
                "false_positives_removed_per_1000": fp_removed,
                "extra_ms_per_frame": EXTRA_PASSES[method] * LATENCY_MS_PER_PASS[kind],
                "latency_source": f"{LATENCY_MS_PER_PASS[kind]:.0f} ms per pass "
                + ("(3-D SegResNet 128^3)" if kind == "3d" else "(2-D, 352x352)"),
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
        for method in ("tta4", "tta8"):
            row = block.loc[block["method"] == method]
            if row.empty:
                continue
            row = row.iloc[0]
            added = (
                float(row["cases_became_majority_miss"]) - float(row["cases_left_majority_miss"])
            ) * 1000.0 / total
            # The external tables carry no HD95 failure-rate estimate (the internal aggregate
            # writes one), and the pixel scale differs per collection, so a per-1000 failure count
            # would not be comparable.  Left empty on purpose; the Dice/HD95 deltas are in
            # results/external_by_model.
            fixed = np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "unit": "frames",
                    "method": method,
                    "boundary_failures_fixed_per_1000": fixed,
                    "misses_added_per_1000": added,
                    "false_positives_removed_per_1000": np.nan,
                    "extra_ms_per_frame": EXTRA_PASSES[method] * LATENCY_MS_PER_PASS["2d"],
                    "latency_source": f"{LATENCY_MS_PER_PASS['2d']:.0f} ms per pass (2-D, 352x352)"
                    + ("" if family == "unet_resnet50" else ", SwinUNETR"),
                    "note": "no HD95 failure-rate estimate in the external tables",
                }
            )
    return rows


def _number(value: float) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:+.1f}"


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
        "Positive *fixed* and *removed* columns are benefits, positive *misses added* is a cost; the "
        "latency column is the extra compute of the additional forward passes, not the total.\n"
    )
    lines.extend(
        [
            "| dataset | unit | method | boundary failures fixed | misses added | "
            "false positives removed | extra ms per frame |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in table.sort_values(["dataset", "method"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {row['unit']} | {row['method']} | "
            f"{_number(row['boundary_failures_fixed_per_1000'])} | "
            f"{_number(row['misses_added_per_1000'])} | "
            f"{_number(row['false_positives_removed_per_1000'])} | "
            f"{row['extra_ms_per_frame']:.0f} |"
        )
    lines.append("")
    lines.append("Latency: " + "; ".join(table["latency_source"].drop_duplicates().tolist()) + ".")
    (OUTPUT / "deployment_terms.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT / 'deployment_terms.csv'} ({len(table)} rows)")
    print(f"wrote {OUTPUT / 'deployment_terms.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


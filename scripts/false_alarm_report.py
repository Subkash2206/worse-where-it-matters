"""False-alarm cost of TTA and of the detection-safe rules on polyp-free colonoscopy.

HyperKvasir's ``cecum`` class contains no polyp, so **every** prediction on those frames is
a false positive.  This script turns the per-case rows of such an evaluation into the table
the paper needs, one row per method:

* ``false_alarm_rate_mean`` - the share of frames on which the method reports a polyp,
  averaged over the checkpoints (each checkpoint scored all frames, so this is the mean of
  per-checkpoint rates, not a pooled case count);
* ``false_alarm_rate_majority`` - the share of frames on which a **majority** of the
  checkpoints reports a polyp (>= 8 of 15); it answers "would a lesion detector built on
  these checkpoints fire?", which the per-checkpoint mean hides;
* ``median_false_area_px`` - the size of the invented polyp in native pixels (median over
  every checkpoint x frame that reported one), plus the median of the per-checkpoint
  medians so a single trigger-happy checkpoint cannot dominate;
* the exact **McNemar** test against ``baseline`` and against plain ``tta8``, on the frame
  level (majority decision) as the primary unit and on the pooled checkpoint x frame pairs
  as a secondary one.

It reads the per-case CSVs that ``scripts/evaluate_external.py`` writes, so nothing is
recomputed: the numbers trace back to individual frames like every other table.

Usage
-----
    python scripts/false_alarm_report.py \\
        --run unet=results/false_alarm/unet \\
        --run segformer=results/false_alarm/segformer \\
        --output-dir results/false_alarm/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import side effect
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.eval import read_long_csv, write_markdown  # noqa: E402
from melba.stats.detection import mcnemar_exact  # noqa: E402
from melba.utils import git_revision  # noqa: E402

#: per-case metrics this report needs (written by the reference-free metric list)
METRICS: tuple[str, ...] = ("pred_empty", "pred_area_px", "max_prob")
#: controls of the McNemar tests
CONTROLS: tuple[str, ...] = ("baseline", "tta8")
#: display order of the methods of a detrules grid
METHOD_ORDER: tuple[str, ...] = (
    "baseline",
    "tta2",
    "tta4",
    "tta8",
    "tta4_anyview",
    "tta4_lowdet_20",
    "tta4_lowdet_30",
    "tta4_lowdet_40",
    "tta8_anyview",
    "tta8_lowdet_20",
    "tta8_lowdet_30",
    "tta8_lowdet_40",
)
DEFAULT_OUTPUT = _REPO_ROOT / "results" / "false_alarm" / "tables"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run",
        dest="runs",
        nargs="+",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="result root written by evaluate_external.py, e.g. unet=results/false_alarm/unet",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--majority",
        type=float,
        default=0.5,
        help="vote fraction that counts as a majority over the checkpoints (default 0.5)",
    )
    parser.add_argument("--dataset", default="hyperkvasir_cecum")
    return parser.parse_args(argv)


def _parse_runs(raw: Sequence[Sequence[str]]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for group in raw:
        for item in group:
            text = str(item)
            if "=" not in text:
                raise SystemExit(f"--run expects LABEL=DIR, got {text!r}")
            label, directory = text.split("=", 1)
            path = Path(directory)
            if not path.is_absolute():
                path = _REPO_ROOT / path
            runs.append((label.strip(), path))
    if not runs:
        raise SystemExit("--run is empty")
    return runs


def load_runs(directory: Path) -> pd.DataFrame:
    """Every per-case CSV below one result root, as one long-format frame."""
    paths = sorted(directory.rglob("*_per_case.csv"))
    if not paths:
        raise SystemExit(f"no per-case CSVs below {directory}")
    return pd.concat([read_long_csv(path) for path in paths], ignore_index=True)


def per_case_alarms(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(checkpoint, case, method)`` with its alarm bit and false area.

    ``alarm`` is ``1 - pred_empty``: an empty prediction is the correct answer on a
    polyp-free cohort, every other prediction is a false positive by construction.
    """
    missing = sorted({"run_id", "case_id", "method", "metric", "value"} - set(runs.columns))
    if missing:
        raise SystemExit(f"per-case frame is missing column(s) {missing}")
    block = runs.loc[runs["metric"].isin(METRICS)]
    if block.empty:
        raise SystemExit(
            "no pred_empty / pred_area_px / max_prob rows: re-run the evaluation with the "
            "reference-free metric list, not with Dice/HD95"
        )
    wide = block.pivot_table(
        index=["run_id", "case_id", "method"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    if "pred_empty" not in wide.columns:
        raise SystemExit("pred_empty is missing from the per-case rows")
    wide["alarm"] = 1.0 - wide["pred_empty"].astype(float)
    return wide


def method_order(methods: Sequence[str]) -> list[str]:
    """Configured display order, with unknown method names appended alphabetically."""
    known = [name for name in METHOD_ORDER if name in set(methods)]
    return known + sorted(set(methods) - set(known))


def _median_or_nan(values: pd.Series) -> float:
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    return float(numbers.median()) if len(numbers) else float("nan")


def summarise(cases: pd.DataFrame, *, majority: float = 0.5) -> pd.DataFrame:
    """Per method: per-checkpoint mean rate, majority rate, false area, peak probability."""
    rows: list[dict[str, Any]] = []
    for method, block in cases.groupby("method", sort=False):
        # one decision per frame: the fraction of checkpoints that report a polyp
        votes = block.groupby("case_id")["alarm"].mean()
        per_checkpoint = block.groupby("run_id")["alarm"].mean()
        alarmed = block.loc[block["alarm"] > 0.0]
        has_area = "pred_area_px" in block.columns
        per_checkpoint_area = (
            [_median_or_nan(group["pred_area_px"]) for _, group in alarmed.groupby("run_id")]
            if has_area and len(alarmed)
            else []
        )
        rows.append(
            {
                "method": str(method),
                "n_cases": int(block["case_id"].nunique()),
                "n_checkpoints": int(block["run_id"].nunique()),
                "false_alarm_rate_mean": float(per_checkpoint.mean()),
                "false_alarm_rate_min": float(per_checkpoint.min()),
                "false_alarm_rate_max": float(per_checkpoint.max()),
                "false_alarm_rate_majority": float((votes > float(majority)).mean()),
                "n_majority_nonempty": int((votes > float(majority)).sum()),
                "n_any_checkpoint_nonempty": int((votes > 0.0).sum()),
                "n_false_positives": int(len(alarmed)),
                "median_false_area_px": _median_or_nan(alarmed["pred_area_px"])
                if has_area
                else float("nan"),
                "median_false_area_px_per_checkpoint": float(np.nanmedian(per_checkpoint_area))
                if per_checkpoint_area
                else float("nan"),
                "mean_max_prob": float(block["max_prob"].mean())
                if "max_prob" in block.columns
                else float("nan"),
            }
        )
    order = method_order([row["method"] for row in rows])
    table = pd.DataFrame(rows)
    table["__order"] = table["method"].map({name: index for index, name in enumerate(order)})
    return table.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def mcnemar_table(
    cases: pd.DataFrame, *, controls: Sequence[str] = CONTROLS, majority: float = 0.5
) -> pd.DataFrame:
    """Exact McNemar of every method against ``baseline`` and plain ``tta8``.

    Two units are reported because they answer different questions: ``case`` uses the
    majority decision of the checkpoints (one decision per frame, n = the frames) and
    ``checkpoint_case`` pools every (checkpoint, frame) pair.  ``new_false_alarms`` counts
    the frames the method invents and the control does not, ``avoided_false_alarms`` the
    reverse.
    """
    majority_flags = (
        cases.groupby(["method", "case_id"])["alarm"].mean() > float(majority)
    ).unstack("method")
    if isinstance(majority_flags, pd.Series):  # a single method: keep the frame protocol
        majority_flags = majority_flags.to_frame()
    majority_flags = majority_flags.astype(bool)
    checkpoint_flags = (
        cases.pivot_table(
            index=["run_id", "case_id"], columns="method", values="alarm", aggfunc="mean"
        )
        > 0.5
    )
    methods = method_order(list(cases["method"].unique()))
    rows: list[dict[str, Any]] = []
    for unit, flags in (("case", majority_flags), ("checkpoint_case", checkpoint_flags)):
        for control in controls:
            if control not in flags.columns:
                continue
            reference = flags[control]
            for method in methods:
                if method == control or method not in flags.columns:
                    continue
                treatment = flags[method]
                new = int((treatment & ~reference).sum())
                avoided = int((~treatment & reference).sum())
                rows.append(
                    {
                        "unit": unit,
                        "method": method,
                        "control": control,
                        "n_units": int(len(reference)),
                        "new_false_alarms": new,
                        "avoided_false_alarms": avoided,
                        "mcnemar_p": mcnemar_exact(new, avoided),
                    }
                )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    order = {name: index for index, name in enumerate(methods)}
    table["__order"] = table["method"].map(order)
    return (
        table.sort_values(["unit", "control", "__order"])
        .drop(columns="__order")
        .reset_index(drop=True)
    )


def _write(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = _parse_runs(args.runs)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    majority = float(args.majority)

    combined: list[pd.DataFrame] = []
    sections: list[tuple[str, Any, str]] = []
    written: list[Path] = []
    for label, directory in runs:
        runs_frame = load_runs(directory)
        cases = per_case_alarms(runs_frame)
        summary = summarise(cases, majority=majority)
        tests = mcnemar_table(cases, majority=majority)
        _write(summary, output / label / "summary.csv")
        _write(tests, output / label / "mcnemar.csv")
        written += [output / label / "summary.csv", output / label / "mcnemar.csv"]
        marked = summary.copy()
        marked.insert(0, "model", label)
        combined.append(marked)
        sections.append(
            (
                f"{label}: false alarms on {args.dataset}",
                summary,
                "`false_alarm_rate_mean` is the mean over the checkpoints of the share of "
                "frames with a non-empty prediction; `..._majority` counts a frame when a "
                "majority of the checkpoints reports a polyp; `median_false_area_px` is the "
                "median area of those (false) predictions in native pixels.",
            )
        )
        sections.append((f"{label}: exact McNemar vs baseline and TTA8", tests, ""))
        print(
            f"\n{label}: {cases['run_id'].nunique()} checkpoints, "
            f"{int(summary['n_cases'].max())} frames"
        )
        print(summary.to_string(index=False))
        print(tests.to_string(index=False))

    if combined:
        pooled = pd.concat(combined, ignore_index=True)
        _write(pooled, output / "false_alarm_all.csv")
        written.append(output / "false_alarm_all.csv")

    write_markdown(
        str(output / "false_alarm.md"),
        title=f"MELBA - false-alarm cost on polyp-free colonoscopy ({args.dataset})",
        preamble=(
            "Every frame of this cohort is polyp-free, so any non-empty prediction is a false "
            "positive: the tables below are the price of TTA and of the detection-safe rules, "
            "not a segmentation score. Dice and HD95 are undefined here and are not reported."
        ),
        sections=sections,
    )
    written.append(output / "false_alarm.md")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "runs": {label: str(directory) for label, directory in runs},
                "majority": majority,
                "metrics": list(METRICS),
                "git_revision": git_revision(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(output / "manifest.json")
    print("\nwritten:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




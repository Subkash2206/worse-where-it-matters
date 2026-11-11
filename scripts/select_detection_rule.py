"""Choose the ``lowdet`` threshold on INTERNAL data only, and record the decision.

The rules of Part 1 lower the detection threshold only when the TTA mean map peaks above it
while the 0.5 mask is empty.  Which threshold that should be is a hyper-parameter, so it is
selected here - on the internal CVC-ClinicDB evaluation of the 15 UNet checkpoints - and the
chosen value is written next to the numbers that justify it.  Nothing external is read: the
script refuses a path that does not look like the internal detrules experiment.

Selection rule (pre-registered, applied mechanically):

1. candidates are ``tta8_lowdet_20``, ``tta8_lowdet_30``, ``tta8_lowdet_40``;
2. drop every candidate whose mean Dice is more than 0.002 below plain ``tta8``;
3. of the rest, take the one with the fewest majority-missed cases (>= 8 of the 15
   checkpoints predicted nothing), then the fewest missed checkpoints over all cases.

Every candidate is reported, including the ones the rule rejects, so the choice can be
re-derived from the table.

Usage
-----
    python scripts/select_detection_rule.py
    python scripts/select_detection_rule.py --per-case-dir results/<experiment>/evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import side effect
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.eval import read_long_csv  # noqa: E402
from melba.utils import git_revision  # noqa: E402

DEFAULT_PER_CASE = _REPO_ROOT / "results" / "cvc_clinicdb_unet_resnet50_detrules" / "evaluation"
DEFAULT_OUTPUT = _REPO_ROOT / "results" / "detection_rules"
CANDIDATES = ("tta8_lowdet_20", "tta8_lowdet_30", "tta8_lowdet_40")
REFERENCE = "tta8"
MAX_DICE_DROP = 0.002
MAJORITY = 8  # of the 15 checkpoints


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--per-case-dir", default=str(DEFAULT_PER_CASE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--max-dice-drop", type=float, default=MAX_DICE_DROP)
    return parser.parse_args(argv)


def load_runs(directory: Path) -> pd.DataFrame:
    """Every per-case CSV below ``directory``, with the run's own fold/seed kept."""
    paths = sorted(directory.rglob("*_per_case.csv"))
    if not paths:
        raise SystemExit(f"no per-case CSVs below {directory}")
    frames = [read_long_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def method_summary(runs: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Dice, per-checkpoint misses and majority-missed cases of every method."""
    dice = runs.loc[runs["metric"] == "dice"]
    rows = []
    for method in methods:
        block = dice.loc[dice["method"] == method]
        if block.empty:
            continue
        per_run = block.groupby(["fold", "seed", "case_id"])["value"].mean()
        misses = block.groupby(["fold", "seed", "case_id"])["missed_detection"].mean()
        rows.append(
            {
                "method": method,
                "n_checkpoints": int(block["run_id"].nunique()),
                "n_cases": int(per_run.size),
                "dice_mean": float(per_run.mean()),
                "dice_median": float(per_run.median()),
                "missed_checkpoints": int(block["missed_detection"].sum()),
                "missed_checkpoint_rate_pct": float(block["missed_detection"].mean() * 100.0),
                "cases_missed_any": int((misses > 0).sum()),
                "cases_majority_missed": int((misses >= MAJORITY / 15.0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)



def choose(table: pd.DataFrame, reference: str, candidates: list[str], max_drop: float) -> dict:
    """Apply the pre-registered selection rule and return the decision with its reasons."""
    reference_row = table.loc[table["method"] == reference]
    if reference_row.empty:
        raise SystemExit(f"reference method {reference!r} is not in the internal results")
    reference_dice = float(reference_row["dice_mean"].iloc[0])
    reference_misses = int(reference_row["cases_majority_missed"].iloc[0])
    eligible: list[dict] = []
    rejected: list[dict] = []
    for method in candidates:
        row = table.loc[table["method"] == method]
        if row.empty:
            continue
        dice = float(row["dice_mean"].iloc[0])
        drop = reference_dice - dice
        entry = {
            "method": method,
            "dice_mean": dice,
            "dice_drop_vs_reference": drop,
            "cases_majority_missed": int(row["cases_majority_missed"].iloc[0]),
            "missed_checkpoints": int(row["missed_checkpoints"].iloc[0]),
        }
        (rejected if drop > max_drop else eligible).append(entry)
    if not eligible:
        raise SystemExit(
            f"every lowdet candidate loses more than {max_drop} Dice against {reference}; "
            "the rule is not admissible on this data"
        )
    eligible.sort(
        key=lambda item: (
            item["cases_majority_missed"],
            item["missed_checkpoints"],
            item["dice_drop_vs_reference"],
        )
    )
    chosen = eligible[0]
    threshold = int(chosen["method"].rsplit("_", 1)[-1]) / 100.0
    return {
        "reference": reference,
        "reference_dice_mean": reference_dice,
        "reference_cases_majority_missed": reference_misses,
        "max_dice_drop": max_drop,
        "chosen_method": chosen["method"],
        "chosen_threshold": threshold,
        "chosen": chosen,
        "eligible": eligible,
        "rejected": rejected,
        "rule": (
            "minimise majority-missed cases among the lowdet candidates whose mean Dice is at "
            f"most {max_drop} below plain {reference}; ties break on missed checkpoints"
        ),


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    per_case_dir = Path(args.per_case_dir)
    if "detrules" not in per_case_dir.as_posix():
        raise SystemExit(
            f"refusing {per_case_dir}: the threshold is selected on the internal CVC-ClinicDB "
            "detrules experiment only, never on external or BraTS data"
        )
    runs = load_runs(per_case_dir)
    datasets = sorted(runs["dataset"].unique())
    if datasets != ["cvc_clinicdb"]:
        raise SystemExit(f"expected only cvc_clinicdb, found {datasets}")

    table = method_summary(runs, [args.reference, *args.candidates])
    decision = choose(table, args.reference, list(args.candidates), float(args.max_dice_drop))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "internal_rule_selection.csv", index=False)
    payload = {
        **decision,
        "per_case_dir": str(per_case_dir),
        "n_per_case_files": int(len(list(per_case_dir.rglob("*_per_case.csv")))),
        "git_revision": git_revision(),
        "table": table.to_dict(orient="records"),
    }
    (output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("internal CVC-ClinicDB (15 checkpoints) - lowdet threshold selection")
    print(table.to_string(index=False))
    print(
        f"\nchosen: {decision['chosen_method']} (t = {decision['chosen_threshold']:.2f}); "
        f"Dice {decision['chosen']['dice_mean']:.4f} vs {decision['reference']} "
        f"{decision['reference_dice_mean']:.4f} "
        f"(drop {decision['chosen']['dice_drop_vs_reference']:+.4f}), majority-missed "
        f"{decision['chosen']['cases_majority_missed']} vs "
        f"{decision['reference_cases_majority_missed']}"
    )
    for entry in decision["rejected"]:
        print(
            f"  rejected {entry['method']}: Dice drop "
            f"{entry['dice_drop_vs_reference']:+.4f} > {decision['max_dice_drop']}"
        )
    print(f"written: {output / 'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

    }

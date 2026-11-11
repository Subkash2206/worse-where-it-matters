"""Report the detection-safe TTA rules: internal, external and BraTS.

Three scopes, three units, and one question: do the rules recover the polyps that plain TTA
silences without giving up the Dice/HD95 gain?

* **internal** (CVC-ClinicDB, 15 checkpoints): one value per video (fold- and seed-averaged),
  so the paired tests run over 29 videos;
* **external** (Kvasir-SEG, CVC-ColonDB, ETIS-Larib, 15 checkpoints each): one value per
  external case, averaged over the checkpoints - the same unit the external tables use -
  plus the majority-missed counts (>= 8 of 15 empty) with exact McNemar tests;
* **BraTS** (5 folds): one value per patient.  The rules can only *add* detections, so the
  interesting numbers are the false-positive rate on patients without an enhancing tumour
  and the missed-ET rate on the others.

Every comparison is against the plain ``baseline`` **and** against plain ``tta4``/``tta8``,
with a paired Wilcoxon test, a 95 % bootstrap interval of the mean difference and Holm
adjustment across the compared rules.  Tables go to ``results/detection_rules``.

Usage
-----
    python scripts/detection_rules_report.py
    python scripts/detection_rules_report.py --external-root results/detrules --output-dir results/detection_rules
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import side effect
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.eval import read_long_csv  # noqa: E402
from melba.stats.detection import mcnemar_exact  # noqa: E402
from melba.stats.paired import analyze_long  # noqa: E402
from melba.utils import git_revision  # noqa: E402

METRICS = ("dice", "boundary_f1", "hd95")
INTERNAL = _REPO_ROOT / "results" / "cvc_clinicdb_unet_resnet50_detrules" / "evaluation"
EXTERNAL_ROOT = _REPO_ROOT / "results" / "detrules"
BRATS = _REPO_ROOT / "results" / "brats2021_segresnet_reduced_detrules" / "evaluation"
OUTPUT = _REPO_ROOT / "results" / "detection_rules"
RULES = (
    "tta4_anyview",
    "tta4_lowdet_20",
    "tta4_lowdet_30",
    "tta4_lowdet_40",
    "tta8_anyview",
    "tta8_lowdet_20",
    "tta8_lowdet_30",
    "tta8_lowdet_40",
)
CONTROLS = ("baseline", "tta4", "tta8")
MAJORITY = 8.0 / 15.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--internal", default=str(INTERNAL))
    parser.add_argument("--external-root", default=str(EXTERNAL_ROOT))
    parser.add_argument("--brats", default=str(BRATS))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args(argv)


def load_dir(directory: Path) -> pd.DataFrame:
    """All per-case CSVs below one evaluation directory (long format)."""
    paths = sorted(directory.rglob("*_per_case.csv"))
    if not paths:
        raise SystemExit(f"no per-case CSVs below {directory}")
    return pd.concat([read_long_csv(path) for path in paths], ignore_index=True)


def unit_averaged(runs: pd.DataFrame, unit: str) -> pd.DataFrame:
    """One row per (unit id, method, metric): the mean over that unit's checkpoints.

    ``unit='group'`` pools the folds/seeds of the internal grid (each video is scored once
    per seed and its fold), so a paired test over ``group_id`` never double-counts a video;
    ``unit='case'`` averages the external checkpoints, and for BraTS the patient is the
    group, which is already one case per fold.
    """
    key = "group_id" if unit == "group" else "case_id"
    subset = runs.loc[runs["metric"].isin(METRICS)]
    averaged = (
        subset.groupby([key, "method", "metric"], as_index=False)["value"]
        .mean()
        .rename(columns={key: "case_id"})
    )
    # the majority empty-mask state, which the paired analysis uses to restrict the
    # distance metrics to cases that both methods predicted something on
    flags = (
        subset.groupby([key, "method"])["empty_flag"]
        .agg(lambda values: values.value_counts().idxmax())
        .reset_index()
        .rename(columns={key: "case_id", "empty_flag": "empty_flag"})
    )
    averaged = averaged.merge(flags, on=["case_id", "method"], how="left")
    averaged["group_id"] = averaged["case_id"]
    averaged["split"] = "test"
    averaged["fold"] = 0
    averaged["seed"] = 0
    return averaged


def paired_block(
    frame: pd.DataFrame,
    control: str,
    treatments: list[str],
    *,
    unit: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Paired tests of every treatment against one control (Holm inside the block)."""
    present = [name for name in treatments if name in set(frame["method"])]
    if control not in set(frame["method"]) or not present:
        return pd.DataFrame()
    table = analyze_long(
        frame,
        treatment=present[0],
        control=control,
        metrics=list(METRICS),
        unit=unit,
        n_bootstrap=n_bootstrap,
        seed=seed,
        correction="holm",
    )
    table["treatment"] = present[0]
    frames = [table]
    for treatment in present[1:]:
        part = analyze_long(
            frame,
            treatment=treatment,
            control=control,
            metrics=list(METRICS),
            unit=unit,
            n_bootstrap=n_bootstrap,
            seed=seed,
            correction="holm",
        )
        part["treatment"] = treatment
        frames.append(part)
    combined = pd.concat(frames, ignore_index=True)
    combined["control"] = control
    return combined


def miss_table(
    runs: pd.DataFrame,
    methods: list[str],
    *,
    reference: str,
    group: str = "case_id",
) -> pd.DataFrame:
    """Miss counts per method plus the exact McNemar test against one reference method.

    Two definitions are reported because they answer different questions: the share of
    *checkpoints* that predicted nothing (a per-run failure rate) and the share of *units*
    that a majority of checkpoints missed (the definition the paper's figures use).
    """
    dice = runs.loc[runs["metric"] == "dice"]
    per_unit = dice.groupby([group, "method"])["missed_detection"].agg(["mean", "count"])
    missed = (per_unit["mean"] > 0.5).unstack("method")
    rows = []
    reference_missed = missed.get(reference)
    for method in methods:
        block = dice.loc[dice["method"] == method]
        if block.empty:
            continue
        unit_missed = missed[method].fillna(False).astype(bool)
        entry = {
            "method": method,
            "n_units": int(len(unit_missed)),
            "checkpoints": int(block["run_id"].nunique()),
            "missed_checkpoints": int(block["missed_detection"].sum()),
            "missed_checkpoint_rate_pct": float(block["missed_detection"].mean() * 100.0),
            "units_missed_any": int((per_unit.loc[(slice(None), method), "mean"] > 0).sum()),
            "units_majority_missed": int(unit_missed.sum()),
            "majority_miss_rate_pct": float(unit_missed.mean() * 100.0),
        }
        if reference_missed is not None and method != reference:
            other = reference_missed.fillna(False).astype(bool).reindex(unit_missed.index).fillna(False)
            b = int((unit_missed & ~other).sum())  # missed only by this method
            c = int((other & ~unit_missed).sum())  # missed only by the reference
            entry["new_vs_reference"] = b
            entry["left_vs_reference"] = c
            entry["mcnemar_p_vs_reference"] = mcnemar_exact(b, c)
        rows.append(entry)
    table = pd.DataFrame(rows)
    table["reference"] = reference
    return table.sort_values("method").reset_index(drop=True)


def false_positive_table(
    runs: pd.DataFrame, methods: list[str], *, reference: str
) -> pd.DataFrame:
    """False positives on lesion-free patients, and missed ET on the others (BraTS).

    A lesion-free patient is one whose reference has no enhancing tumour, identified from the
    reference method's ``gt_empty`` flag (the reference mask never changes between methods).
    A prediction on such a patient is a false positive; silence on an ET-positive patient is
    a missed ET.  Both are case-majority decisions, tested with the exact McNemar test
    against the reference method.
    """
    dice = runs.loc[runs["metric"] == "dice"]
    volume = runs.loc[runs["metric"] == "pred_volume_mm3"]
    flags = (
        dice.groupby(["case_id", "method"])["empty_flag"]
        .agg(lambda values: values.value_counts().idxmax())
        .unstack("method")
    )
    if reference not in flags.columns:
        return pd.DataFrame()
    lesion_free = set(flags.index[flags[reference] == "gt_empty"])
    et_cases = set(flags.index) - lesion_free
    reference_fp = {case for case in lesion_free if flags.loc[case, reference] != "pred_empty"}
    reference_missed = {case for case in et_cases if flags.loc[case, reference] == "pred_empty"}
    volumes = volume.groupby(["method", "case_id"])["value"].mean()
    rows = []
    for method in methods:
        if method not in flags.columns:
            continue
        column = flags[method]
        fp = {case for case in lesion_free if column.get(case) != "pred_empty"}
        missed = {case for case in et_cases if column.get(case) == "pred_empty"}
        b_fp, c_fp = len(fp - reference_fp), len(reference_fp - fp)
        b_et, c_et = len(missed - reference_missed), len(reference_missed - missed)
        fp_volumes = volumes.loc[method].reindex(sorted(fp)).dropna()
        rows.append(
            {
                "method": method,
                "lesion_free_patients": len(lesion_free),
                "false_positive_patients": len(fp),
                "false_positive_rate_pct": 100.0 * len(fp) / max(1, len(lesion_free)),
                "median_fp_volume_mm3": float(fp_volumes.median()) if len(fp_volumes) else 0.0,
                "max_fp_volume_mm3": float(fp_volumes.max()) if len(fp_volumes) else 0.0,
                "fp_new_vs_reference": b_fp,
                "fp_left_vs_reference": c_fp,
                "fp_mcnemar_p": mcnemar_exact(b_fp, c_fp),
                "et_patients": len(et_cases),
                "missed_et_patients": len(missed),
                "missed_et_rate_pct": 100.0 * len(missed) / max(1, len(et_cases)),
                "missed_et_new_vs_reference": b_et,
                "missed_et_left_vs_reference": c_et,
                "missed_et_mcnemar_p": mcnemar_exact(b_et, c_et),
                "reference": reference,
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)



def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _print_pairs(table: pd.DataFrame, title: str) -> None:
    """Compact console view of a paired block (Dice/BF1/HD95 only)."""
    if table.empty:
        print(f"\n{title}: no data")
        return
    view = table.loc[
        table["metric"].isin(METRICS),
        [
            "control",
            "treatment",
            "metric",
            "n_pairs",
            "mean_control",
            "mean_treatment",
            "delta_mean",
            "ci_low",
            "ci_high",
            "p_value",
            "p_value_adjusted",
        ],
    ]
    print(f"\n{title}")
    print(view.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _means(frame: pd.DataFrame) -> pd.DataFrame:
    table = (
        frame.groupby(["method", "metric"])["value"]
        .agg(n_units="count", mean="mean", median="median", std="std")
        .reset_index()
    )
    return table.loc[table["metric"].isin(METRICS)].sort_values(["method", "metric"])



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ---------------------------------------------------------------- internal
    internal_runs = load_dir(Path(args.internal))
    methods = [
        name
        for name in ("baseline", "tta2", "tta4", "tta8", *RULES)
        if name in set(internal_runs["method"])
    ]
    internal = unit_averaged(internal_runs, "group")
    internal_pairs = pd.concat(
        [
            paired_block(
                internal,
                control,
                [name for name in methods if name != control],
                unit="group",
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            for control in CONTROLS
        ],
        ignore_index=True,
    )
    internal_miss = miss_table(internal_runs, methods, reference="tta8")
    for name, table in (
        ("internal_paired", internal_pairs),
        ("internal_means", _means(internal)),
        ("internal_misses", internal_miss),
    ):
        path = output / f"{name}.csv"
        _write(table, path)
        written.append(path)
    print(
        f"internal: {internal_runs['run_id'].nunique()} checkpoints, "
        f"{internal_miss['n_units'].iloc[0]} videos"
    )
    _print_pairs(internal_pairs, "internal CVC-ClinicDB: paired Dice/BF1/HD95 (unit = video)")
    print("\ninternal: misses (majority = >= 8 of 15 checkpoints)")
    print(internal_miss.to_string(index=False))

    # ---------------------------------------------------------------- external
    external_root = Path(args.external_root)
    if external_root.is_dir():
        for dataset_dir in sorted(external_root.glob("external_*")):
            dataset = dataset_dir.name.removeprefix("external_")
            runs = load_dir(dataset_dir)
            present = [name for name in methods if name in set(runs["method"])]
            frame = unit_averaged(runs, "case")
            pairs = pd.concat(
                [
                    paired_block(
                        frame,
                        control,
                        [name for name in present if name != control],
                        unit="case",
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed,
                    )
                    for control in CONTROLS
                ],
                ignore_index=True,
            )
            tables = {
                "paired": pairs,
                "misses_vs_tta8": miss_table(runs, present, reference="tta8"),
                "misses_vs_tta4": miss_table(runs, present, reference="tta4"),
                "means": _means(frame),
            }
            for name, table in tables.items():
                path = output / "external" / dataset / f"{name}.csv"
                _write(table, path)
                written.append(path)
            print(
                f"\nexternal {dataset}: {runs['run_id'].nunique()} checkpoints, "
                f"{tables['means']['n_units'].max()} cases"
            )
            _print_pairs(pairs, f"external {dataset}: paired Dice/BF1/HD95 (unit = case)")
            print("\nmisses vs tta8 (majority = >= 8 of 15)")
            print(tables["misses_vs_tta8"].to_string(index=False))

    # ---------------------------------------------------------------- BraTS
    brats_dir = Path(args.brats)
    if brats_dir.is_dir():
        runs = load_dir(brats_dir)
        present = [name for name in methods if name in set(runs["method"])]
        frame = unit_averaged(runs, "group")
        pairs = pd.concat(
            [
                paired_block(
                    frame,
                    control,
                    [name for name in present if name != control],
                    unit="group",
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                )
                for control in CONTROLS
            ],
            ignore_index=True,
        )
        tables = {
            "paired": pairs,
            "false_positive": false_positive_table(runs, present, reference="tta8"),
            "misses": miss_table(runs, present, reference="tta8"),
        }
        for name, table in tables.items():
            path = output / "brats" / f"{name}.csv"
            _write(table, path)
            written.append(path)
        print(
            f"\nbraTS: {runs['run_id'].nunique()} runs, "
            f"{tables['misses']['n_units'].iloc[0]} patients"
        )
        _print_pairs(pairs, "braTS: paired Dice/BF1/HD95 (unit = patient)")
        print("\nbraTS: false positives on lesion-free patients and missed ET (vs tta8)")
        print(tables["false_positive"].to_string(index=False))
        print("\nbraTS: misses vs tta8")
        print(tables["misses"].to_string(index=False))

    manifest = pd.DataFrame(
        {
            "file": [str(path.relative_to(output)) for path in written],
            "git_revision": git_revision(),
        }
    )
    _write(manifest, output / "manifest.csv")
    print(f"\nwritten {len(written)} table(s) to {output} (git {git_revision()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


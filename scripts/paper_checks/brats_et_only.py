"""BraTS tail and uncertainty numbers on the 1218 patients WITH enhancing tumour.

Why: aggregate_brats2021_segresnet_reduced/tail_metrics.csv and uncertainty_audit_auroc.csv
also include the 33 lesion-free patients. In tail_table() a lesion-free patient with a false
positive has undefined HD95 and counts as a large failure; in the uncertainty audit a
lesion-free patient (Dice undefined) is labelled a clean success. The paper reports those
patients only as false alarms, so these numbers are recomputed on ET patients only.

Uses the repo's own tail_table(), _auroc(), auroc_ci() and auroc_difference() unchanged,
with the same settings as scripts/aggregate.py (10,000 bootstrap draws for the tail table,
1,000 for the audit, seed 0). As a control it first reruns everything on all 1251 patients
and asserts that the published files are reproduced.

Run from the repository root:  python scripts/paper_checks/brats_et_only.py
Writes: results/paper_checks/brats_tail_et_only.csv, brats_uncertainty_et_only.csv,
        brats_uncertainty_tests_et_only.csv
Needs only numpy, pandas, scipy (no torch, no GPU). Reads results; changes nothing else.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results" / "brats2021_segresnet_reduced" / "evaluation"
AGG = ROOT / "results" / "aggregate_brats2021_segresnet_reduced"
OUT = ROOT / "results" / "paper_checks"

# --- the repo's own functions, loaded without importing torch --------------------------
spec = importlib.util.spec_from_file_location("melba_report", ROOT / "src/melba/eval/report.py")
report = importlib.util.module_from_spec(spec)
sys.modules["melba_report"] = report
spec.loader.exec_module(report)
_src = (ROOT / "src/melba/eval/uncertainty_audit.py").read_text(encoding="utf-8")
_ns: dict[str, Any] = {"np": np, "pd": pd, "scipy_stats": scipy_stats, "Any": Any, "Iterable": Iterable}
for _node in ast.parse(_src).body:
    if isinstance(_node, ast.FunctionDef) and _node.name in ("_auroc", "auroc_ci", "auroc_difference"):
        exec(compile(ast.Module([_node], []), "uncertainty_audit.py", "exec"), _ns)
auroc_ci, auroc_difference = _ns["auroc_ci"], _ns["auroc_difference"]

ET_FLAGS = ("both_nonempty", "pred_empty")  # reference non-empty
THRESHOLDS = report.TailThresholds(hd95_px=10.0, dice=0.5)  # stats.tail_hd95 / tail_dice


def load() -> pd.DataFrame:
    files = sorted(EVAL.glob("brats2021_test_fold*_seed0_per_case.csv"))
    assert len(files) == 5, f"expected 5 per-case files, found {len(files)}"
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame["case_id"] = frame["case_id"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    assert frame["case_id"].nunique() == 1251, frame["case_id"].nunique()
    assert not frame.duplicated(["case_id", "method", "metric"]).any(), "duplicate rows"
    return frame


def tail(frame: pd.DataFrame) -> pd.DataFrame:
    return report.tail_table(frame, control="baseline", thresholds=THRESHOLDS,
                             n_bootstrap=10000, ci=0.95, seed=0)


def case_table(frame: pd.DataFrame) -> pd.DataFrame:
    def wide(method: str, metrics: list[str], prefix: str) -> pd.DataFrame:
        sub = frame[(frame.method == method) & frame.metric.isin(metrics)]
        return sub.pivot(index="case_id", columns="metric", values="value").add_prefix(prefix)
    t = wide("tta4", ["dice", "entropy_mean_roi", "variance_mean_roi", "view_disagreement", "entropy_mean"], "t4_")
    t = t.join(wide("baseline", ["entropy_mean_roi"], "b_"))
    flags = frame[(frame.method == "tta4") & (frame.metric == "dice")].set_index("case_id")
    t = t.join(flags[["empty_flag", "missed_detection"]])
    t["group"] = t.index
    t["has_et"] = t["empty_flag"].isin(ET_FLAGS)
    return t


PREDICTORS = {
    "tta4_entropy_roi": "t4_entropy_mean_roi",
    "tta4_variance_roi": "t4_variance_mean_roi",
    "tta4_view_disagreement": "t4_view_disagreement",
    "tta4_entropy_image": "t4_entropy_mean",
}


def audit(t: pd.DataFrame, et_only: bool) -> pd.DataFrame:
    missed = t["missed_detection"].astype(bool)
    dice = t["t4_dice"].astype(float)
    definitions = {  # identical to run_uncertainty_audit()
        "all_failures": (missed | (dice < 0.5), None),
        "non_missed_failures": ((~missed) & (dice < 0.5), ~missed),
        "missed_vs_clean": (missed, missed | ((~missed) & (dice >= 0.5))),
    }
    rows = []
    for name, column in PREDICTORS.items():
        for definition, (labels, subset) in definitions.items():
            sel = t if subset is None else t[subset]
            if et_only:
                sel = sel[sel["has_et"]]
            r = auroc_ci(sel[column], labels.loc[sel.index], sel["group"], n_bootstrap=1000, ci=0.95, seed=0)
            rows.append({"predictor": name, "failure_definition": definition, **r})
    return pd.DataFrame(rows)


def tests(t: pd.DataFrame) -> pd.DataFrame:
    missed = t["missed_detection"].astype(bool)
    dice = t["t4_dice"].astype(float)
    sel = t[(~missed) & t["has_et"]]
    labels = ((~missed) & (dice < 0.5)).loc[sel.index]
    rows = []
    for a, b in (("t4_entropy_mean_roi", "b_entropy_mean_roi"), ("t4_entropy_mean_roi", "t4_variance_mean_roi"),
                 ("t4_entropy_mean_roi", "t4_view_disagreement")):
        r = auroc_difference(sel[a], sel[b], labels, sel["group"], n_bootstrap=1000, ci=0.95, seed=0)
        rows.append({"predictor": a, "compared_with": b, "failure_definition": "non_missed_failures", **r})
    return pd.DataFrame(rows)


def main() -> None:
    frame = load()
    t = case_table(frame)
    n_et, n_free = int(t["has_et"].sum()), int((~t["has_et"]).sum())
    assert (n_et, n_free) == (1218, 33), (n_et, n_free)

    # control 1: the published tail table is reproduced on all 1251 patients
    published = pd.read_csv(AGG / "tail_metrics.csv").set_index("method")
    rerun = tail(frame).set_index("method")
    numeric = [c for c in rerun.columns if c in published.columns and pd.api.types.is_numeric_dtype(rerun[c])]
    diff = (rerun[numeric] - published.loc[rerun.index, numeric]).abs().max().max()
    assert diff < 1e-9, f"tail table not reproduced (max diff {diff})"

    # control 2: the published audit AUROCs are reproduced on all patients
    pub = pd.read_csv(AGG / "uncertainty_audit_auroc.csv")
    full = audit(t, et_only=False)
    merged = full.merge(pub[["predictor", "failure_definition", "auroc"]].drop_duplicates(["predictor", "failure_definition"]),
                        on=["predictor", "failure_definition"], suffixes=("", "_published"))
    gap = (merged["auroc"] - merged["auroc_published"]).abs().max()
    assert gap < 1e-9, f"audit AUROC not reproduced (max diff {gap})"

    OUT.mkdir(parents=True, exist_ok=True)
    et_frame = frame[frame["case_id"].isin(t.index[t["has_et"]])]
    tail_et = tail(et_frame)
    assert (tail_et["n_cases"] == 1218).all()
    tail_et.to_csv(OUT / "brats_tail_et_only.csv", index=False)
    audit(t, et_only=True).to_csv(OUT / "brats_uncertainty_et_only.csv", index=False)
    tests(t).to_csv(OUT / "brats_uncertainty_tests_et_only.csv", index=False)
    print(f"controls passed (tail max diff {diff:.1e}, audit max diff {gap:.1e}); wrote {OUT}")
    print(tail_et[["method", "n_cases", "hd95_failure_rate", "delta_hd95_failure_rate_estimate",
                   "delta_hd95_failure_rate_ci_low", "delta_hd95_failure_rate_ci_high",
                   "delta_hd95_p90_estimate", "delta_hd95_p90_ci_low", "delta_hd95_p90_ci_high"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

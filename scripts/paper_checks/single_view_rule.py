"""Low-threshold rule on the single view versus on TTA-8 (UNet, external data).

Control for the detection-rule results: the rule ``lowdet_T`` (if the 0.5 mask is empty but
the probability map peaks at >= T, threshold the map at T) is applied to the single identity
view (``baseline_lowdet_T``, configs/cvc_clinicdb/unet_resnet50_baserules.yaml) and compared
with the same rule on the TTA-8 mean map (``tta8_lowdet_T``) on the same 15 checkpoints.

Per method: images left empty and images missed (no overlap, Dice = 0) by at least 8 of 15
checkpoints on CVC-ColonDB and ETIS-Larib, mean Dice, and the false-alarm rate on the 1009
polyp-free HyperKvasir cecum frames (mean over checkpoints of the share of non-empty masks).
Paired exact McNemar tests compare single view + rule with TTA-8 + rule at the same T.

Run from the repository root:  python scripts/paper_checks/single_view_rule.py
Writes: results/paper_checks/single_view_rule.csv, single_view_rule_tests.csv
Needs only pandas. Reads results; changes nothing else.
"""
from __future__ import annotations

import glob
from math import comb
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "paper_checks"
K = 8
POLYP = {
    "single": "baserules/external_{ds}/unet_resnet50_baserules",
    "tta": "detrules/external_{ds}/unet_resnet50_detrules",
}
CECUM = {
    "single": "false_alarm_baserules/unet/external_hyperkvasir_cecum/unet_resnet50_baserules",
    "tta": "false_alarm/unet/external_hyperkvasir_cecum/unet_resnet50_detrules",
}
METHODS = {
    "single": ["baseline", "baseline_lowdet_40", "baseline_lowdet_30", "baseline_lowdet_20"],
    "tta": ["baseline", "tta8", "tta8_lowdet_40", "tta8_lowdet_30", "tta8_lowdet_20", "tta8_anyview"],
}


def mcnemar_exact(new: int, recovered: int) -> float:
    n = new + recovered
    if n == 0:
        return 1.0
    k = min(new, recovered)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0**n)


def load(directory: str, metric: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RES / directory / "evaluation" / "*_per_case.csv")))
    if len(files) != 15:
        raise FileNotFoundError(f"expected 15 per-case files under results/{directory}, found {len(files)}")
    cols = ["seed", "fold", "method", "case_id", "empty_flag", "metric", "value"]
    d = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
    d = d[d.metric == metric].copy()
    d["checkpoint"] = d.fold.astype(str) + "_" + d.seed.astype(str)
    return d


def main() -> None:
    rows, per_case = [], {}
    for source in ("single", "tta"):
        cecum = load(CECUM[source], "pred_empty")
        fa = 100.0 * (1.0 - cecum.groupby(["method", "checkpoint"]).value.mean().groupby("method").mean())
        polyp = {ds: load(POLYP[source].format(ds=ds), "dice") for ds in ("cvc_colondb", "etis_larib")}
        for m in METHODS[source]:
            rec = dict(source=source, method=m, false_alarm_pct=round(float(fa[m]), 2))
            for ds, d in polyp.items():
                x = d[d.method == m]
                assert x.case_id.nunique() in (380, 196) and x.checkpoint.nunique() == 15
                g = pd.DataFrame({
                    "empty": (x.empty_flag == "pred_empty").groupby(x.case_id).sum(),
                    "no_overlap": (x.value == 0).groupby(x.case_id).sum(),
                })
                per_case[(source, m, ds)] = g
                rec[f"{ds}_majority_empty"] = int((g["empty"] >= K).sum())
                rec[f"{ds}_majority_no_overlap"] = int((g.no_overlap >= K).sum())
                rec[f"{ds}_pairs_empty"] = int((x.empty_flag == "pred_empty").sum())
                rec[f"{ds}_pairs_no_overlap"] = int((x.value == 0).sum())
                rec[f"{ds}_mean_dice"] = round(float(x.value.mean()), 4)
            rows.append(rec)
    table = pd.DataFrame(rows)

    # control: the single-view baseline of the new run must equal the old run
    a = table[(table.source == "single") & (table.method == "baseline")].drop(columns=["source"]).iloc[0]
    b = table[(table.source == "tta") & (table.method == "baseline")].drop(columns=["source"]).iloc[0]
    for col in a.index:
        if col.endswith(("majority_empty", "majority_no_overlap", "pairs_empty", "pairs_no_overlap")):
            assert a[col] == b[col], (col, a[col], b[col])
    assert abs(a.false_alarm_pct - b.false_alarm_pct) < 0.01

    tests = []
    for t in ("40", "30", "20"):
        for ds in ("cvc_colondb", "etis_larib"):
            s = per_case[("single", f"baseline_lowdet_{t}", ds)]
            u = per_case[("tta", f"tta8_lowdet_{t}", ds)].reindex(s.index)
            rec = dict(threshold=f"0.{t}", dataset=ds)
            for name in ("empty", "no_overlap"):
                only_tta = int(((u[name] >= K) & (s[name] < K)).sum())
                only_single = int(((s[name] >= K) & (u[name] < K)).sum())
                rec[f"majority_{name}_only_tta8_rule"] = only_tta
                rec[f"majority_{name}_only_single_rule"] = only_single
                rec[f"majority_{name}_mcnemar_p"] = mcnemar_exact(only_tta, only_single)
            tests.append(rec)
    tests = pd.DataFrame(tests)

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "single_view_rule.csv", index=False)
    tests.to_csv(OUT / "single_view_rule_tests.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.to_string(index=False))
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()

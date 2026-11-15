"""Missed lesions with an overlap criterion (Dice = 0), next to the empty-mask counts.

Why: the aggregate tables count a lesion as missed only when the predicted mask is empty.
A non-empty mask that lies entirely off the lesion then counts as a detection. This script
recounts every case with two definitions, from the per-case files the evaluation already
wrote:

  empty       the predicted mask has no foreground pixel (the existing definition);
  no overlap  the prediction shares no pixel with the reference (Dice = 0). This includes
              empty masks and non-empty masks that lie entirely off the lesion.

A case is counted "by majority" when at least k checkpoints agree: 8 of 15 (external),
2 of 3 seeds (internal CVC-ClinicDB), 1 of 1 (BraTS, one seed per fold). Cases with an
empty reference (33 BraTS patients without enhancing tumour) are excluded.

Run from the repository root:  python scripts/paper_checks/overlap_misses.py
Writes: results/paper_checks/overlap_misses.csv       (counts per model, dataset, method)
        results/paper_checks/overlap_transitions.csv  (paired changes and exact McNemar p)
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
METHODS = ["baseline", "tta4", "tta8", "tta8_anyview", "tta8_lowdet_20", "tta8_lowdet_40"]
# (scope, dataset, model, evaluation directory, majority k)
RUNS = [
    ("internal", "cvc_clinicdb", "unet", "cvc_clinicdb_unet_resnet50_detrules", 2),
    ("internal", "cvc_clinicdb", "segformer", "cvc_clinicdb_segformer_mit_b2_detrules", 2),
    ("internal", "cvc_clinicdb", "swinunetr", "cvc_clinicdb_swinunetr_2d", 2),
    ("internal", "brats2021", "segresnet", "brats2021_segresnet_reduced_detrules", 1),
]
for ds in ("kvasir_seg", "cvc_colondb", "etis_larib"):
    RUNS.append(("external", ds, "unet", f"detrules/external_{ds}/unet_resnet50_detrules", 8))
    seg = f"detrules_segformer/external_{ds}/segformer_mit_b2_detrules"
    if not (RES / seg).exists():  # Kvasir-SEG was not rerun with the rules for SegFormer
        seg = f"external_{ds}/segformer_mit_b2"
    RUNS.append(("external", ds, "segformer", seg, 8))
    RUNS.append(("external", ds, "swinunetr", f"external_{ds}/swinunetr_2d", 8))


def mcnemar_exact(new: int, recovered: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs."""
    n = new + recovered
    if n == 0:
        return 1.0
    k = min(new, recovered)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0**n)


def load(directory: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RES / directory / "evaluation" / "*_per_case.csv")))
    if not files:
        raise FileNotFoundError(f"no per-case files under results/{directory}/evaluation")
    cols = ["seed", "fold", "method", "case_id", "empty_flag", "metric", "value"]
    d = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
    d = d[d.metric == "dice"].copy()
    d = d[~d.empty_flag.isin(["gt_empty", "both_empty"])]  # lesion present in the reference
    d["checkpoint"] = d.fold.astype(str) + "_" + d.seed.astype(str)
    d["empty"] = (d.empty_flag == "pred_empty").astype(int)
    d["no_overlap"] = (d.value == 0).astype(int)
    # an empty prediction must have Dice 0, and Dice is exact (no smoothing term)
    assert (d.loc[d["empty"] == 1, "value"] == 0).all()
    assert not ((d.value > 0) & (d.value < 1e-6)).any()
    return d


def main() -> None:
    rows, trans = [], []
    for scope, dataset, model, directory, k in RUNS:
        d = load(directory)
        per_case, per_pair = {}, {}
        for m in METHODS:
            x = d[d.method == m]
            if x.empty:
                continue
            g = x.groupby("case_id").agg(empty=("empty", "sum"), no_overlap=("no_overlap", "sum"))
            per_case[m] = g
            per_pair[m] = x.set_index(["checkpoint", "case_id"])
            rows.append(dict(
                scope=scope, dataset=dataset, model=model, method=m,
                n_cases=x.case_id.nunique(), n_checkpoints=x.checkpoint.nunique(), majority_k=k,
                pairs=len(x), pairs_empty=int(x["empty"].sum()), pairs_no_overlap=int(x.no_overlap.sum()),
                pairs_off_target=int(x.no_overlap.sum() - x["empty"].sum()),
                majority_empty=int((g["empty"] >= k).sum()), majority_no_overlap=int((g.no_overlap >= k).sum()),
                mean_dice=round(float(x.value.mean()), 4),
            ))
        for ref, others in (("baseline", METHODS[1:]), ("tta8", METHODS[3:])):
            if ref not in per_case:
                continue
            b, pb = per_case[ref], per_pair[ref]
            for m in others:
                if m not in per_case:
                    continue
                t, pt = per_case[m].reindex(b.index), per_pair[m].reindex(pb.index)
                rec = dict(scope=scope, dataset=dataset, model=model, reference=ref, method=m)
                for name in ("empty", "no_overlap"):
                    new = int(((t[name] >= k) & (b[name] < k)).sum())
                    gone = int(((b[name] >= k) & (t[name] < k)).sum())
                    rec[f"majority_{name}_new"] = new
                    rec[f"majority_{name}_recovered"] = gone
                    rec[f"majority_{name}_mcnemar_p"] = mcnemar_exact(new, gone)
                new_empty = (t["empty"] >= k) & (b["empty"] < k)
                rec["new_majority_empty_already_no_overlap"] = int((new_empty & (b.no_overlap >= k)).sum())
                went_empty = (pb["empty"] == 0) & (pt["empty"] == 1)
                rec["pairs_nonempty_to_empty"] = int(went_empty.sum())
                rec["pairs_nonempty_to_empty_had_overlap"] = int((went_empty & (pb.value > 0)).sum())
                rec["pairs_overlap_lost"] = int(((pb.value > 0) & (pt.value == 0)).sum())
                rec["pairs_overlap_gained"] = int(((pb.value == 0) & (pt.value > 0)).sum())
                trans.append(rec)
    counts, changes = pd.DataFrame(rows), pd.DataFrame(trans)

    # control: the empty-mask counts must reproduce the numbers already in the paper
    def maj(dataset, model, method, col="majority_empty"):
        r = counts[(counts.dataset == dataset) & (counts.model == model) & (counts.method == method)]
        return int(r[col].iloc[0])
    expected = {("etis_larib", "unet", "baseline"): 5, ("etis_larib", "unet", "tta8"): 18,
                ("cvc_colondb", "unet", "baseline"): 55, ("cvc_colondb", "unet", "tta8"): 68,
                ("cvc_colondb", "unet", "tta8_anyview"): 13, ("etis_larib", "unet", "tta8_anyview"): 0,
                ("cvc_colondb", "segformer", "tta8"): 1, ("cvc_clinicdb", "unet", "baseline"): 15,
                ("cvc_clinicdb", "unet", "tta4"): 19, ("brats2021", "segresnet", "baseline"): 6,
                ("brats2021", "segresnet", "tta4"): 9, ("brats2021", "segresnet", "tta8"): 7}
    for key, value in expected.items():
        assert maj(*key) == value, (key, maj(*key), value)

    OUT.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT / "overlap_misses.csv", index=False)
    changes.to_csv(OUT / "overlap_transitions.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print(counts.drop(columns=["scope", "n_checkpoints", "majority_k"]).to_string(index=False))
    print(changes.drop(columns=["scope"]).to_string(index=False))


if __name__ == "__main__":
    main()

"""Three robustness checks on the external missed-lesion counts (no GPU, no new inference).

1. Stricter criterion. The paper counts a lesion as found with any overlap (Dice > 0). Here a
   lesion is also counted as missed when the intersection over union with the reference is
   below 0.10 or below 0.25 (per-case metric ``jaccard``). Majority = at least 8 of 15
   checkpoints, as in the paper.
2. Spread over checkpoints. For every one of the 15 checkpoints: the change from the single
   view to TTA-8 in the number of empty predictions and of missed images.
3. Clustering. External images carry no sequence labels. As a rough proxy, the images that
   become empty (UNet) or missed (SwinUNETR) by majority under TTA-8 are grouped into runs of
   consecutive image numbers (CVC-ColonDB and ETIS-Larib; Kvasir-SEG file names are not numbers), and the number of runs is compared with random placement
   (10,000 permutations, seed 0). Few runs = the images cluster.

Run from the repository root:  python scripts/paper_checks/robustness_checks.py
4. IoU sweep. Missed images for the single view and TTA-8 at IoU thresholds from 0 (no overlap)
   to 0.50, with newly missed and recovered images, and the mean miss rate per checkpoint.

Writes: results/paper_checks/strict_overlap.csv, strict_overlap_transitions.csv,
        per_checkpoint_spread.csv, miss_clustering.csv, iou_sweep.csv
Needs only numpy and pandas. Reads results; changes nothing else.
"""
from __future__ import annotations

import glob
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "paper_checks"
K = 8
DATASETS = ("kvasir_seg", "cvc_colondb", "etis_larib")
CRITERIA = {"no_overlap": 0.0, "iou_below_010": 0.10, "iou_below_025": 0.25}
RULES = ["tta8_anyview", "tta8_lowdet_20", "tta8_lowdet_40"]


def directory(model: str, ds: str) -> str:
    if model == "unet":
        return f"detrules/external_{ds}/unet_resnet50_detrules"
    if model == "segformer":
        d = f"detrules_segformer/external_{ds}/segformer_mit_b2_detrules"
        return d if (RES / d).exists() else f"external_{ds}/segformer_mit_b2"
    if model == "swinunetr":
        return f"external_{ds}/swinunetr_2d"
    if model == "unet_single":
        return f"baserules/external_{ds}/unet_resnet50_baserules"
    raise ValueError(model)


def mcnemar_exact(new: int, recovered: int) -> float:
    n = new + recovered
    if n == 0:
        return 1.0
    k = min(new, recovered)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0**n)


def load(d: str, n_files: int = 15) -> pd.DataFrame:
    files = sorted(glob.glob(str(RES / d / "evaluation" / "*_per_case.csv")))
    if len(files) != n_files:
        raise FileNotFoundError(f"expected {n_files} per-case files under results/{d}, found {len(files)}")
    cols = ["seed", "fold", "method", "case_id", "empty_flag", "metric", "value"]
    x = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
    x = x[x.metric.isin(["dice", "jaccard"])]
    x = x[~x.empty_flag.isin(["gt_empty", "both_empty"])]  # lesion present in the reference
    x["checkpoint"] = x.fold.astype(str) + "_" + x.seed.astype(str)
    wide = x.pivot_table(index=["method", "checkpoint", "case_id"], columns="metric", values="value").reset_index()
    flags = x[x.metric == "dice"][["method", "checkpoint", "case_id", "empty_flag"]]
    wide = wide.merge(flags, on=["method", "checkpoint", "case_id"])
    wide["empty"] = (wide.empty_flag == "pred_empty").astype(int)
    wide["no_overlap"] = (wide.dice == 0).astype(int)
    wide["iou_below_010"] = (wide.jaccard < 0.10).astype(int)
    wide["iou_below_025"] = (wide.jaccard < 0.25).astype(int)
    assert (wide.loc[wide.no_overlap == 1, "jaccard"] == 0).all()
    return wide


def majority(frame: pd.DataFrame, method: str, column: str, k: int = K) -> pd.Series:
    x = frame[frame.method == method]
    return x.groupby("case_id")[column].sum() >= k


def sign_test(up: int, down: int) -> float:
    """Two-sided sign test over checkpoints (ties dropped)."""
    return mcnemar_exact(up, down)


SWEEP = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def runs(ids: np.ndarray) -> int:
    ids = np.sort(ids)
    return 0 if len(ids) == 0 else int(1 + (np.diff(ids) > 1).sum())


def main() -> None:
    counts, trans, spread, cluster, sweep = [], [], [], [], []
    rng = np.random.default_rng(0)
    for ds in DATASETS:
        frames = {m: load(directory(m, ds)) for m in ("unet", "segformer", "swinunetr")}
        if ds != "kvasir_seg":
            frames["unet_single"] = load(directory("unet_single", ds))
        for model, frame in frames.items():
            methods = [m for m in ["baseline", "tta8"] + RULES + ["baseline_lowdet_20"] if (frame.method == m).any()]
            for m in methods:
                rec = dict(dataset=ds, model=model, method=m, n_images=frame.case_id.nunique())
                rec["majority_empty"] = int(majority(frame, m, "empty").sum())
                for c in CRITERIA:
                    rec[f"majority_{c}"] = int(majority(frame, m, c).sum())
                counts.append(rec)
            pairs = [("baseline", m) for m in methods if m != "baseline"] + [("tta8", m) for m in methods if m in RULES]
            for ref, m in pairs:
                rec = dict(dataset=ds, model=model, reference=ref, method=m)
                for c in CRITERIA:
                    b, t = majority(frame, ref, c), majority(frame, m, c)
                    new, gone = int((t & ~b).sum()), int((b & ~t).sum())
                    rec[f"{c}_new"], rec[f"{c}_recovered"], rec[f"{c}_p"] = new, gone, mcnemar_exact(new, gone)
                trans.append(rec)
            if model == "unet_single":
                continue
            # 2. spread over checkpoints, single view -> TTA-8
            b = frame[frame.method == "baseline"].groupby("checkpoint")[["empty", "no_overlap", "iou_below_010"]].sum()
            t = frame[frame.method == "tta8"].groupby("checkpoint")[["empty", "no_overlap", "iou_below_010"]].sum()
            diff = t - b
            for c in diff.columns:
                spread.append(dict(dataset=ds, model=model, count=c, n_checkpoints=len(diff),
                                   baseline_min=int(b[c].min()), baseline_median=float(b[c].median()), baseline_max=int(b[c].max()),
                                   change_min=int(diff[c].min()), change_median=float(diff[c].median()), change_max=int(diff[c].max()),
                                   checkpoints_up=int((diff[c] > 0).sum()), checkpoints_down=int((diff[c] < 0).sum()),
                                   checkpoints_same=int((diff[c] == 0).sum()),
                                   sign_test_p=sign_test(int((diff[c] > 0).sum()), int((diff[c] < 0).sum()))))
            # 4. IoU sweep
            for thr in SWEEP:
                flag = ((frame.jaccard < thr) if thr > 0 else (frame.dice == 0)).astype(int)
                tmp = frame.assign(flag=flag)
                bm, tm = majority(tmp, "baseline", "flag"), majority(tmp, "tta8", "flag")
                new, gone = int((tm & ~bm).sum()), int((bm & ~tm).sum())
                rb = tmp[tmp.method == "baseline"].groupby("checkpoint").flag.mean()
                rt = tmp[tmp.method == "tta8"].groupby("checkpoint").flag.mean()
                up, down = int((rt > rb).sum()), int((rt < rb).sum())
                sweep.append(dict(dataset=ds, model=model, iou_threshold=thr, baseline=int(bm.sum()), tta8=int(tm.sum()),
                                  new=new, recovered=gone, mcnemar_p=mcnemar_exact(new, gone),
                                  rate_baseline_pct=round(100 * rb.mean(), 2), rate_tta8_pct=round(100 * rt.mean(), 2),
                                  checkpoints_up=up, checkpoints_down=down, sign_test_p=sign_test(up, down)))
            # 3. clustering of newly empty / newly without overlap images, and of images recovered by IoU
            for c in ("empty", "no_overlap", "iou_below_010_recovered"):
                column = c.replace("_recovered", "")
                bm, tm = majority(frame, "baseline", column), majority(frame, "tta8", column)
                changed = (bm & ~tm) if c.endswith("_recovered") else (tm & ~bm)
                new_ids = np.array(list(tm.index[changed]))
                if len(new_ids) < 5:
                    continue
                try:  # image numbers only exist where file names are numeric (not Kvasir-SEG)
                    new_ids = new_ids.astype(int)
                    all_ids = np.array(sorted(int(i) for i in tm.index))
                except ValueError:
                    continue
                observed = runs(new_ids)
                sims = np.array([runs(rng.choice(all_ids, size=len(new_ids), replace=False)) for _ in range(10000)])
                cluster.append(dict(dataset=ds, model=model, count=c, newly=len(new_ids), runs_observed=observed,
                                    runs_expected_random=round(float(sims.mean()), 2),
                                    p_fewer_or_equal_runs=float((sims <= observed).mean()),
                                    image_numbers=" ".join(str(i) for i in new_ids)))
    # internal data: majority = 2 of 3 seeds (CVC-ClinicDB), 1 of 1 (BraTS)
    internal = [("cvc_clinicdb", "unet", "cvc_clinicdb_unet_resnet50_detrules", 15, 2),
                ("cvc_clinicdb", "segformer", "cvc_clinicdb_segformer_mit_b2_detrules", 15, 2),
                ("cvc_clinicdb", "swinunetr", "cvc_clinicdb_swinunetr_2d", 15, 2),
                ("brats2021", "segresnet", "brats2021_segresnet_reduced_detrules", 5, 1)]
    for ds, model, d, n_files, k in internal:
        frame = load(d, n_files)
        methods = [m for m in ["baseline", "tta4", "tta8"] + RULES if (frame.method == m).any()]
        for m in methods:
            rec = dict(dataset=ds, model=model, method=m, n_images=frame.case_id.nunique())
            rec["majority_empty"] = int(majority(frame, m, "empty", k).sum())
            for c in CRITERIA:
                rec[f"majority_{c}"] = int(majority(frame, m, c, k).sum())
            counts.append(rec)
        for ref, m in [("baseline", x) for x in methods if x != "baseline"] + [("tta8", x) for x in methods if x in RULES]:
            rec = dict(dataset=ds, model=model, reference=ref, method=m)
            for c in CRITERIA:
                b, t = majority(frame, ref, c, k), majority(frame, m, c, k)
                new, gone = int((t & ~b).sum()), int((b & ~t).sum())
                rec[f"{c}_new"], rec[f"{c}_recovered"], rec[f"{c}_p"] = new, gone, mcnemar_exact(new, gone)
            trans.append(rec)
    # single view + rule against TTA-8 + rule (UNet, same threshold 0.20)
    for ds in ("cvc_colondb", "etis_larib"):
        single, tta = load(directory("unet_single", ds)), load(directory("unet", ds))
        rec = dict(dataset=ds, model="unet", reference="baseline_lowdet_20 (single view)", method="tta8_lowdet_20")
        for c in CRITERIA:
            b, t = majority(single, "baseline_lowdet_20", c), majority(tta, "tta8_lowdet_20", c)
            t = t.reindex(b.index)
            new, gone = int((t & ~b).sum()), int((b & ~t).sum())
            rec[f"{c}_new"], rec[f"{c}_recovered"], rec[f"{c}_p"] = new, gone, mcnemar_exact(new, gone)
        trans.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {"strict_overlap": pd.DataFrame(counts), "strict_overlap_transitions": pd.DataFrame(trans),
              "per_checkpoint_spread": pd.DataFrame(spread), "miss_clustering": pd.DataFrame(cluster),
              "iou_sweep": pd.DataFrame(sweep)}
    # control: the any-overlap majority counts must equal those in overlap_misses.csv
    old = pd.read_csv(OUT / "overlap_misses.csv")
    for _, r in tables["strict_overlap"].iterrows():
        if r.model == "unet_single":
            continue
        o = old[(old.dataset == r.dataset) & (old.model == r.model) & (old.method == r.method)]
        if len(o):
            assert int(o.majority_no_overlap.iloc[0]) == r.majority_no_overlap, (r.dataset, r.model, r.method)
            assert int(o.majority_empty.iloc[0]) == r.majority_empty, (r.dataset, r.model, r.method)
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200, "display.max_colwidth", 80)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n== {name}")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()

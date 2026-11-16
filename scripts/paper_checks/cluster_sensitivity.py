"""Cluster-aware sensitivity for the external counts, and per-checkpoint distributions.

External images carry no sequence labels, so the image-level tests in the paper treat images
as independent and their p-values are optimistic. Here the headline external comparisons are
repeated with a block bootstrap: images are cut into contiguous blocks of consecutive image
numbers whose length matches the mean sequence length reported for each collection
(CVC-ColonDB 380 images / 15 sequences = 25; ETIS-Larib 196 / 34 = 6; Kvasir-SEG has no
sequence structure and is not blocked), blocks are resampled with replacement 10,000 times,
and the 95 % interval of the change in the count is reported. The blocks are a proxy for
sequences, not the real ones.

It also writes, per checkpoint, the three counts for the single view and TTA-8, so the
majority-of-15 summary in the paper can be read against the checkpoint-to-checkpoint spread.

Run from the repository root:  python scripts/paper_checks/cluster_sensitivity.py
Writes: results/paper_checks/cluster_sensitivity.csv, per_checkpoint_counts.csv
Needs only numpy and pandas. Reads results; changes nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robustness_checks as rc  # noqa: E402

OUT = rc.OUT
BLOCK = {"cvc_colondb": 25, "etis_larib": 6, "kvasir_seg": None}
COUNTS = ("empty", "no_overlap", "iou_below_010")
DRAWS = 10000


def main() -> None:
    rows, per_ck = [], []
    for ds in rc.DATASETS:
        for model in ("unet", "segformer", "swinunetr"):
            frame = rc.load(rc.directory(model, ds))
            size = BLOCK[ds]
            if size:  # image numbers are consecutive, so contiguous blocks stand in for sequences
                ids = np.array(sorted(int(i) for i in frame.case_id.unique()))
                blocks = [ids[i:i + size] for i in range(0, len(ids), size)]
            else:  # Kvasir-SEG: no sequence structure and no numbering, so one image per block
                ids = np.array(sorted(frame.case_id.unique()))
                blocks = [np.array([i]) for i in ids]
            rng = np.random.default_rng(0)
            for c in COUNTS:
                b = rc.majority(frame, "baseline", c).reindex(ids).to_numpy()
                t = rc.majority(frame, "tta8", c).reindex(ids).to_numpy()
                pos = {v: k for k, v in enumerate(ids)}
                per_block = np.array([sum(int(t[pos[i]]) - int(b[pos[i]]) for i in blk) for blk in blocks], dtype=float)
                n = len(blocks)
                draws = per_block[rng.integers(0, n, size=(DRAWS, n))].sum(axis=1) * (len(ids) / (n * np.mean([len(x) for x in blocks])))
                lo, hi = np.percentile(draws, [2.5, 97.5])
                rows.append(dict(dataset=ds, model=model, count=c, n_images=len(ids),
                                 block_size=size or 1, n_blocks=n,
                                 baseline=int(b.sum()), tta8=int(t.sum()), change=int(t.sum() - b.sum()),
                                 block_bootstrap_lo=round(float(lo), 1), block_bootstrap_hi=round(float(hi), 1),
                                 excludes_zero=bool(lo > 0 or hi < 0)))
            if model == "unet":
                for method, label in (("baseline", "single view"), ("tta8", "TTA-8")):
                    x = frame[frame.method == method]
                    g = x.groupby("checkpoint")[list(COUNTS)].sum()
                    for ck, r in g.iterrows():
                        per_ck.append(dict(dataset=ds, model=model, method=label, checkpoint=ck,
                                           **{c: int(r[c]) for c in COUNTS}))
    cs, pc = pd.DataFrame(rows), pd.DataFrame(per_ck)
    # control: the majority counts must match the table the paper is built from
    old = pd.read_csv(OUT / "strict_overlap.csv")
    for _, r in cs.iterrows():
        o = old[(old.dataset == r.dataset) & (old.model == r.model) & (old.method == "baseline")].iloc[0]
        assert int(o[f"majority_{r['count']}"]) == r.baseline, (r.dataset, r.model, r["count"])
    cs.to_csv(OUT / "cluster_sensitivity.csv", index=False)
    pc.to_csv(OUT / "per_checkpoint_counts.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_rows", 100)
    print(cs.to_string(index=False))
    print(pc.groupby(["dataset", "method"])[list(COUNTS)].describe().loc[:, (slice(None), ["min", "50%", "max"])].to_string())


if __name__ == "__main__":
    main()

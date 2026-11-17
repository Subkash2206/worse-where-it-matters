"""Per-checkpoint counts for every model, collection and criterion (no GPU, no new inference).

robustness_checks.py computes these per-checkpoint counts but writes only their summary
(per_checkpoint_spread.csv) plus the full table for UNet. The body figure needs the full
table for all three architectures, so this script re-reads the same per-case files and
writes it. Reuses load() and directory() from robustness_checks, so the numbers are the
same by construction.

Run from the repository root:  python scripts/paper_checks/per_checkpoint_table.py
Writes: results/paper_checks/per_checkpoint_all.csv
Needs only numpy and pandas. Reads results; changes nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_checks import DATASETS, OUT, directory, load  # noqa: E402

COUNTS = ["empty", "no_overlap", "iou_below_010"]
MODELS = ("unet", "segformer", "swinunetr")


def main() -> None:
    rows = []
    for ds in DATASETS:
        for model in MODELS:
            frame = load(directory(model, ds))
            per = {}
            for method in ("baseline", "tta8"):
                per[method] = frame[frame.method == method].groupby("checkpoint")[COUNTS].sum()
            diff = per["tta8"] - per["baseline"]
            for ckpt in diff.index:
                for c in COUNTS:
                    rows.append(dict(dataset=ds, model=model, checkpoint=ckpt, count=c,
                                     n_images=frame.case_id.nunique(),
                                     baseline=int(per["baseline"].loc[ckpt, c]),
                                     tta8=int(per["tta8"].loc[ckpt, c]),
                                     change=int(diff.loc[ckpt, c])))
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "per_checkpoint_all.csv", index=False)
    print(f"wrote {OUT / 'per_checkpoint_all.csv'}  rows={len(table)}")
    print(table.groupby(["dataset", "model", "count"]).change.median().to_string())


if __name__ == "__main__":
    main()

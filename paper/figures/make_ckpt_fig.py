"""Body figure 2: the paired per-checkpoint change, which is the paper's primary estimand.

One panel per architecture. Within a panel, the three miss criteria side by side, and
within each criterion the three external collections. Each small dot is one of the 15
checkpoints: the change in the number of images meeting that criterion when TTA-8 replaces
the single view, as a percentage of the collection. The bar is the median of those 15
paired changes. Above zero TTA looks harmful under that criterion, below zero helpful.
"""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
plt.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.0, "axes.labelsize": 8.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.linewidth": 0.5, "grid.alpha": 0.35, "grid.color": "#b0b0b0",
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.frameon": False,
    "lines.linewidth": 1.4, "pdf.fonttype": 42,
})

SRC = "/mnt/user-data/uploads/melba-tta/results/paper_checks"
N = {"kvasir_seg": 1000, "cvc_colondb": 380, "etis_larib": 196}
DSETS = ("kvasir_seg", "cvc_colondb", "etis_larib")
DSLABEL = {"kvasir_seg": "Kvasir-SEG", "cvc_colondb": "CVC-ColonDB", "etis_larib": "ETIS-Larib"}
COLOUR = {"kvasir_seg": "#1f6fb4", "cvc_colondb": "#d1821c", "etis_larib": "#1b9e77"}
MODELS = ("unet", "segformer", "swinunetr")
MLABEL = {"unet": "(a) UNet", "segformer": "(b) SegFormer", "swinunetr": "(c) SwinUNETR"}
COUNTS = ("empty", "no_overlap", "iou_below_010")
CLABEL = {"empty": "empty", "no_overlap": "no overlap", "iou_below_010": "missed"}

per = pd.read_csv(f"{SRC}/per_checkpoint_all.csv")
spread = pd.read_csv(f"{SRC}/per_checkpoint_spread.csv")

# Holm across all 27 cells, matching the Methods description.
p = spread.sign_test_p.to_numpy()
order = np.argsort(p)
m = len(p)
stepped = np.maximum.accumulate([min(1.0, (m - i) * p[order][i]) for i in range(m)])
holm = np.empty(m)
holm[order] = stepped
spread["holm_p"] = holm
key = {(r.dataset, r.model, r.count): (r.sign_test_p, r.holm_p) for r in spread.itertuples()}

rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.45), sharey=True)
OFF = {"kvasir_seg": -0.26, "cvc_colondb": 0.0, "etis_larib": 0.26}

for ax, model in zip(axes, MODELS):
    ax.axhline(0.0, color="#404040", linewidth=0.9, zorder=1)
    for xi, c in enumerate(COUNTS):
        if xi:  # faint separator between criteria
            ax.axvline(xi - 0.5, color="#d8d8d8", linewidth=0.6, zorder=0)
        for ds in DSETS:
            g = per[(per.dataset == ds) & (per.model == model) & (per["count"] == c)]
            pct = 100.0 * g.change.to_numpy() / N[ds]
            x = xi + OFF[ds]
            jit = rng.uniform(-0.065, 0.065, size=len(pct))
            ax.plot(x + jit, pct, linestyle="none", marker="o", markersize=1.9,
                    color=COLOUR[ds], alpha=0.5, markeredgewidth=0, zorder=2)
            med = float(np.median(pct))
            ax.plot([x - 0.105, x + 0.105], [med, med], color=COLOUR[ds],
                    linewidth=1.8, solid_capstyle="butt", zorder=4)
            nom, hp = key[(ds, model, c)]
            if nom < 0.01:
                ax.plot([x], [med], marker="o", markersize=3.6, color=COLOUR[ds],
                        markerfacecolor=COLOUR[ds] if hp < 0.01 else "white",
                        markeredgewidth=1.0, zorder=5)
    ax.set_title(MLABEL[model], loc="left")
    ax.set_xticks(range(len(COUNTS)))
    ax.set_xticklabels([CLABEL[c] for c in COUNTS], rotation=20.0, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.55, len(COUNTS) - 0.45)

fig.subplots_adjust(wspace=0.12)
axes[0].set_ylabel("per-checkpoint change in\ncount with TTA-8 [%]")
handles = [Line2D([], [], color=COLOUR[d], marker="o", markersize=3.2, linewidth=1.8,
                  label=DSLABEL[d]) for d in DSETS]
axes[0].legend(handles=handles, loc="lower left", handletextpad=0.5, borderaxespad=0.3,
               labelspacing=0.25)

out = "/tmp/claude-0/-home-claude/60cb95db-11db-5a90-870f-67b7dca4df20/scratchpad/fig_ckpt.pdf"
fig.savefig(out)
print("wrote", out)

rows = spread[["dataset", "model", "count", "change_median", "checkpoints_up",
               "checkpoints_down", "checkpoints_same", "sign_test_p", "holm_p"]]
print(rows[rows.sign_test_p < 0.01].to_string(index=False))
ymin = min(100.0 * per[per.dataset == d].change.min() / N[d] for d in DSETS)
ymax = max(100.0 * per[per.dataset == d].change.max() / N[d] for d in DSETS)
print(f"y range across dots: {ymin:.1f}% to {ymax:.1f}%")

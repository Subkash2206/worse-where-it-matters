"""Body figure: the verdict on TTA as a function of the localisation bar.

x: IoU threshold required for an image to count as localised.
y: change in missed images with TTA-8, as a percentage of the collection.
Above zero means TTA looks harmful under that bar; below zero means it looks helpful.
Filled markers mark exact McNemar p < 0.01.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

SRC = "/mnt/user-data/uploads/melba-tta/results/paper_checks/iou_sweep.csv"
N = {"kvasir_seg": 1000, "cvc_colondb": 380, "etis_larib": 196}
TITLE = {"kvasir_seg": "(a) Kvasir-SEG", "cvc_colondb": "(b) CVC-ColonDB", "etis_larib": "(c) ETIS-Larib"}
COLOUR = {"unet": "#1f6fb4", "segformer": "#d1821c", "swinunetr": "#1b9e77"}
LABEL = {"unet": "UNet", "segformer": "SegFormer", "swinunetr": "SwinUNETR"}
MARK = {"unet": "o", "segformer": "s", "swinunetr": "^"}

t = pd.read_csv(SRC)
fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.15), sharey=True)

for ax, ds in zip(axes, ("kvasir_seg", "cvc_colondb", "etis_larib")):
    ax.axhline(0.0, color="#404040", linewidth=0.9, zorder=1)
    ax.axvspan(0.05, 0.10, color="#cccccc", alpha=0.45, zorder=0, linewidth=0)
    for model in ("unet", "segformer", "swinunetr"):
        g = t[(t.dataset == ds) & (t.model == model)].sort_values("iou_threshold")
        pct = 100.0 * (g.tta8 - g.baseline) / N[ds]
        ax.plot(g.iou_threshold, pct, color=COLOUR[model], marker=MARK[model],
                markersize=3.4, markerfacecolor="white", markeredgewidth=1.0,
                label=LABEL[model], zorder=3)
        sig = g.mcnemar_p < 0.01
        ax.plot(g.iou_threshold[sig], pct[sig], linestyle="none", marker=MARK[model],
                markersize=3.4, color=COLOUR[model], markerfacecolor=COLOUR[model],
                markeredgewidth=1.0, zorder=4)
    ax.set_title(TITLE[ds], loc="left")
    ax.set_xlabel("IoU required")
    ax.set_xlim(-0.02, 0.52)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

axes[0].set_ylabel("change in missed\nimages with TTA-8 [%]")
axes[0].legend(loc="lower left", ncol=1, handletextpad=0.5, borderaxespad=0.3)
# The shaded band marks where the verdict turns; the caption says so.
out = "/tmp/claude-0/-home-claude/60cb95db-11db-5a90-870f-67b7dca4df20/scratchpad/fig_sweep.pdf"
fig.savefig(out)
print("wrote", out)

for ds in N:
    g = t[(t.dataset == ds) & (t.model == "swinunetr")].sort_values("iou_threshold")
    print(ds, "swin change:", [(r.iou_threshold, int(r.tta8 - r.baseline), r.mcnemar_p < 0.01) for r in g.itertuples()][:4])

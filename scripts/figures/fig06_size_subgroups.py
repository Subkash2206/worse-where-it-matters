"""Figure 6 - size subgroups: TTA-8 Dice gain and miss rate per size tertile and dataset.

Two panels share the x axis (small / medium / large tertile of the reference polyp area, cut per
dataset), and every collection is marked as external:

* top: the paired TTA-8 Dice gain with its 95 % bootstrap CI, so a bar whose interval crosses zero is
  visible as such;
* bottom: a paired dot plot of the share of cases that a **majority of the 15 checkpoints** miss,
  baseline against TTA-8, for the same tertile and dataset.  The two dots of a pair are joined by a
  short vertical segment and nothing is drawn across tertiles, so a reader never confuses a
  between-dataset trend with a within-pair change - which is the price paid for the gain.

Both panels come straight from ``results/external_by_model`` and update themselves when the SwinUNETR
runs land.  Colours are the shared Okabe-Ito palette of ``style.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import style  # noqa: E402

DELTA = style.RESULTS / "external_by_model" / "subgroup_delta.csv"
SUMMARY = style.RESULTS / "external_by_model" / "subgroup_summary.csv"
BINS = ("small", "medium", "large")



def main() -> int:
    style.apply()
    for path in (DELTA, SUMMARY):
        if not path.is_file():
            print(f"missing {path}; run scripts/external_by_model.py first", file=sys.stderr)
            return 2
    delta = pd.read_csv(DELTA)
    delta = delta.loc[(delta["metric"] == "dice") & (delta["comparison"] == "tta8 vs baseline")]
    summary = pd.read_csv(SUMMARY)
    summary = summary.loc[summary["metric"] == "dice"]
    if delta.empty or summary.empty:
        print("no tta8 Dice rows yet", file=sys.stderr)
        return 2

    import matplotlib.pyplot as plt
    import numpy as np

    families = [
        family for family in ("unet_resnet50", "swinunetr_2d") if family in set(delta["family"])
    ]
    datasets = [
        name for name in ("kvasir_seg", "cvc_colondb", "etis_larib") if name in set(delta["dataset"])
    ]
    family_label = {"unet_resnet50": "UNet", "swinunetr_2d": "SwinUNETR"}
    # one column per architecture: three series per bin instead of six in the same axis
    fig, axes = plt.subplots(
        2,
        len(families),
        figsize=(3.6 * len(families), 3.9),
        sharex=True,
        sharey="row",
        squeeze=False,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    positions = np.arange(len(BINS))
    width = 0.8 / max(len(datasets), 1)
    for column, family in enumerate(families):
        top, bottom = axes[0][column], axes[1][column]
        offset = -0.4 + width / 2
        for dataset in datasets:
            block = delta.loc[(delta["family"] == family) & (delta["dataset"] == dataset)]
            if block.empty:
                offset += width
                continue
            colour = style.PALETTE.get(dataset, "#555555")
            marker = style.MARKERS.get(dataset, "o")
            label = style.LABELS.get(dataset, dataset)
            values, lows, highs = [], [], []
            for bin_label in BINS:
                row = block.loc[block["size_bin"] == bin_label]
                values.append(float(row["delta_mean"].iloc[0]) if not row.empty else np.nan)
                lows.append(float(row["delta_mean"].iloc[0] - row["ci_low"].iloc[0]) if not row.empty else np.nan)
                highs.append(float(row["ci_high"].iloc[0] - row["delta_mean"].iloc[0]) if not row.empty else np.nan)
            top.bar(
                positions + offset,
                values,
                width,
                color=colour,
                edgecolor="white",
                linewidth=0.4,
                label=label,
                yerr=[lows, highs],
                error_kw={"elinewidth": 0.6, "capthick": 0.6},
            )
            base = summary.loc[
                (summary["family"] == family)
                & (summary["dataset"] == dataset)
                & (summary["method"] == "baseline")
            ]
            tta = summary.loc[
                (summary["family"] == family)
                & (summary["dataset"] == dataset)
                & (summary["method"] == "tta8")
            ]
            if not base.empty and not tta.empty:
                base = base.set_index("size_bin")["miss_rate"] * 100.0
                tta = tta.set_index("size_bin")["miss_rate"] * 100.0
                base_values = np.array([float(base.get(bin_label, np.nan)) for bin_label in BINS])
                tta_values = np.array([float(tta.get(bin_label, np.nan)) for bin_label in BINS])
                xs = positions + offset
                # paired dot plot: only the two dots of one pair are joined, never two tertiles
                for x, base_value, tta_value in zip(xs, base_values, tta_values):
                    if not np.isfinite(base_value) or not np.isfinite(tta_value):
                        continue
                    bottom.plot(
                        [x, x],
                        [base_value, tta_value],
                        color=colour,
                        linewidth=1.0,
                        zorder=2,
                        solid_capstyle="butt",
                    )
                bottom.plot(
                    xs,
                    base_values,
                    linestyle="none",
                    marker=marker,
                    markersize=4.4,
                    markerfacecolor="white",
                    markeredgecolor=colour,
                    markeredgewidth=0.9,
                    zorder=3,
                )
                bottom.plot(
                    xs,
                    tta_values,
                    linestyle="none",
                    marker=marker,
                    markersize=4.4,
                    markerfacecolor=colour,
                    markeredgecolor=colour,
                    markeredgewidth=0.9,
                    zorder=4,
                )
            offset += width
        top.axhline(0.0, color="#909090", linewidth=0.7)
        top.set_ylabel("TTA-8 Dice gain" if column == 0 else "")
        top.set_title(f"({'ab'[column]}) {family_label[family]}", loc="left")
        top.set_ylim(bottom=min(-0.005, float(top.get_ylim()[0])))
        bottom.set_ylabel("missed by most checkpoints (%)" if column == 0 else "")
        bottom.set_xlabel("reference polyp-area tertile (cut per dataset)")
        bottom.set_xticks(positions)
        bottom.set_xticklabels([f"{label}" for label in BINS])
        bottom.set_ylim(-0.9, float(bottom.get_ylim()[1]) * 1.06)
        block_all = summary.loc[summary["family"] == family]
        if not block_all.empty and float(block_all["miss_rate"].max()) == 0.0:
            bottom.annotate(
                "no majority miss in any bin",
                (0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=6.5,
                color="#606060",
            )
    fig.suptitle(
        "TTA-8 gains Dice in most size bins but adds missed polyps in every bin on ColonDB and ETIS",
        y=1.03,
        fontsize=8.5,
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    extra = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=4.4, markerfacecolor="white",
                   markeredgecolor="#404040", markeredgewidth=0.9, label="baseline (open)"),
        plt.Line2D([], [], linestyle="none", marker="o", markersize=4.4, markerfacecolor="#404040",
                   markeredgecolor="#404040", markeredgewidth=0.9, label="TTA-8 (filled)"),
    ]
    fig.legend(
        handles=handles + extra,
        labels=labels + [item.get_label() for item in extra],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=6.5,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    fig.text(0.5, -0.16, "marker shape = dataset", ha="center", va="top", fontsize=5.5, color="#606060")
    style.save(fig, "fig06_size_subgroups")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

"""Figure 6 - size subgroups: TTA-8 Dice gain and miss rate per size tertile and dataset.

Two panels share the x axis (small / medium / large tertile of the reference polyp area, cut per
dataset).  Top: the paired TTA-8 Dice gain with its 95 % bootstrap CI, so a bar whose interval
crosses zero is visible as such.  Bottom: the share of cases that a majority of the 15 checkpoints
miss, baseline vs TTA-8, i.e. the price paid for the gain.  Both panels come straight from
``results/external_by_model`` and update themselves when the SwinUNETR runs land.
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

    families = sorted(delta["family"].unique())
    datasets = [name for name in delta["dataset"].unique()]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.0, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]}
    )
    positions = np.arange(len(BINS))
    width = 0.8 / max(len(families) * len(datasets), 1)
    offset = -0.4 + width / 2
    for family in families:
        for dataset in datasets:
            block = delta.loc[(delta["family"] == family) & (delta["dataset"] == dataset)]
            if block.empty:
                continue
            key = dataset if family == "unet_resnet50" else "cvc_swin"
            colour = style.PALETTE.get(key, "#555555")
            label = style.LABELS.get(dataset, dataset)
            if family != "unet_resnet50":
                colour, label = style.PALETTE["cvc_swin"], label + ", Swin"
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
                base = base.set_index("size_bin")["miss_rate"]
                tta = tta.set_index("size_bin")["miss_rate"]
                bottom.plot(
                    positions + offset,
                    [float(base.get(bin_label, np.nan)) for bin_label in BINS],
                    linestyle=":",
                    marker="o",
                    markersize=3,
                    color=colour,
                    linewidth=1.0,
                )
                bottom.plot(
                    positions + offset,
                    [float(tta.get(bin_label, np.nan)) for bin_label in BINS],
                    linestyle="-",
                    marker="o",
                    markersize=3,
                    color=colour,
                    linewidth=1.4,
                )
            offset += width
    top.axhline(0.0, color="#909090", linewidth=0.7)
    top.set_ylabel("TTA-8 Dice gain")
    top.set_title("Gain concentrates on small polyps; large ColonDB polyps gain nothing")
    top.legend(loc="upper right", ncol=3)
    top.set_ylim(bottom=min(-0.005, float(top.get_ylim()[0])))
    bottom.set_ylabel("majority-miss rate")
    bottom.set_xlabel("reference polyp-area tertile (cut per dataset)")
    bottom.set_xticks(positions)
    bottom.set_xticklabels([f"{label}" for label in BINS])
    style.annotate(bottom, "dotted: baseline, solid: TTA-8", corner="upper left")
    style.save(fig, "fig06_size_subgroups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Figure 2 - internal vs external: baseline Dice drop against the relative TTA-8 gain.

Every external collection is one point; x is how much the baseline Dice falls from the internal
CVC-ClinicDB result of the same architecture, y is how much TTA with eight views gains *in relative
terms* on that collection.  The point is the message: TTA does not lose value out of distribution,
the relative gain grows with the domain gap.  SwinUNETR points appear automatically once the
external_swin queue has written its runs (the table is rebuilt from the CSVs at run time).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import style  # noqa: E402

COMPARISON = style.RESULTS / "external_by_model" / "comparison.csv"


def main() -> int:
    style.apply()
    if not COMPARISON.is_file():
        print(f"missing {COMPARISON}; run scripts/external_by_model.py first", file=sys.stderr)
        return 2
    table = pd.read_csv(COMPARISON)
    table = table.loc[table["method"] == "tta8"]
    if table.empty:
        print("no tta8 rows in the comparison table", file=sys.stderr)
        return 2

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axhline(0.0, color="#909090", linewidth=0.7, zorder=1)
    ax.axvline(0.0, color="#909090", linewidth=0.7, zorder=1)
    for family, block in table.groupby("family"):
        colour = style.PALETTE.get("cvc_unet", "#1f4e79") if family == "unet_resnet50" else style.PALETTE["cvc_swin"]
        marker = "o" if family == "unet_resnet50" else "s"
        for _, row in block.iterrows():
            dataset = str(row["dataset"])
            ax.scatter(
                row["drop_rel_pct"],
                row["gain_rel_pct_external"],
                s=46,
                marker=marker,
                facecolor=colour,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                label=style.LABELS.get(dataset, dataset),
            )
            ax.annotate(
                style.LABELS.get(dataset, dataset).replace(" (external)", ""),
                (row["drop_rel_pct"], row["gain_rel_pct_external"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=6.5,
                color="#303030",
            )
    # the internal reference of the same architecture
    for family, label in (("unet_resnet50", "CVC-ClinicDB, UNet"), ("swinunetr_2d", "CVC-ClinicDB, SwinUNETR")):
        block = table.loc[table["family"] == family]
        if block.empty:
            continue
        colour = style.PALETTE["cvc_unet"] if family == "unet_resnet50" else style.PALETTE["cvc_swin"]
        ax.scatter(
            0.0,
            float(block["gain_rel_pct_internal"].iloc[0]),
            s=46,
            marker="X",
            facecolor=colour,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label=label.replace(", ", " (internal, ") + ")",
        )
    ax.set_xlabel("baseline Dice drop from internal to external  [%]")
    ax.set_ylabel("relative TTA-8 Dice gain  [%]")
    ax.set_title("TTA pays more, not less, outside the training distribution")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", handletextpad=0.3)
    style.annotate(ax, "circles: UNet, squares: SwinUNETR", corner="lower right")
    style.save(fig, "fig02_internal_vs_external")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

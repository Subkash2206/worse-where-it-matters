"""Figure 2 - does TTA lose value outside the training distribution?

x is the *domain gap*: how far the baseline Dice of one external collection falls below the internal
CVC-ClinicDB baseline of the same architecture, in relative percent.  The internal reference is the
per-case mean baseline Dice of the internal UNet, ``0.7763`` over the 612 test cases of the
cross-validation grid - a constant of the result tables, so the gap can be re-derived by hand as
``(0.7763 - dice_external) / 0.7763 * 100`` and is printed for every collection.  The internal point
itself sits at x = 0.  y is the relative TTA-8 Dice gain on that collection, and the dashed line marks
the internal TTA-8 gain (3.7 %): points above it gain more than on the training distribution.

Reads ``results/external_by_model/comparison.csv``; SwinUNETR points appear automatically once the
external_swin queue has written its runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import style  # noqa: E402

COMPARISON = style.RESULTS / "external_by_model" / "comparison.csv"
INTERNAL_DICE = 0.7763  # per-case mean baseline Dice of the internal CVC-ClinicDB UNet
LABELS = {
    "kvasir_seg": "Kvasir-SEG",
    "cvc_colondb": "CVC-ColonDB",
    "etis_larib": "ETIS-Larib",
}
#: label anchor per collection, so the annotations never sit on top of each other or on the marker
LABEL_OFFSETS = {
    "kvasir_seg": (8, 3, "left"),
    "cvc_colondb": (8, -11, "left"),
    "etis_larib": (-9, 6, "right"),
}
#: gaps the internal UNet must reproduce (relative percent below 0.7763)
EXPECTED_GAPS = {"kvasir_seg": 7.8, "cvc_colondb": 18.7, "etis_larib": 40.4}


def collect(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (family, collection): domain gap, TTA-8 gain, internal TTA-8 gain."""
    baseline = table.loc[
        table["method"] == "baseline", ["family", "dataset", "dice_external"]
    ].copy()
    final = table.loc[
        table["method"] == "tta8",
        ["family", "dataset", "gain_rel_pct_external", "gain_rel_pct_internal"],
    ].copy()
    merged = baseline.merge(final, on=["family", "dataset"], how="inner")
    merged["gap_pct"] = (INTERNAL_DICE - merged["dice_external"].astype(float)) * 100.0 / INTERNAL_DICE
    return merged.sort_values("gap_pct").reset_index(drop=True)



def main() -> int:
    style.apply()
    if not COMPARISON.is_file():
        print(f"missing {COMPARISON}; run scripts/external_by_model.py first", file=sys.stderr)
        return 2
    table = pd.read_csv(COMPARISON)
    table = table.loc[table["method"].isin(["baseline", "tta8"])]
    points = collect(table)
    if points.empty:
        print("no tta8 rows in the comparison table", file=sys.stderr)
        return 2

    print(f"domain gap against the internal UNet baseline Dice {INTERNAL_DICE:.4f}:")
    for _, row in points.iterrows():
        print(
            f"  {str(row['dataset']):12s} ({str(row['family']):14s}) "
            f"baseline Dice {float(row['dice_external']):.4f} -> gap {float(row['gap_pct']):5.1f} % "
            f"(expected {EXPECTED_GAPS.get(str(row['dataset']), float('nan')):5.1f} %), "
            f"TTA-8 gain {float(row['gain_rel_pct_external']):5.1f} %"
        )
        expected = EXPECTED_GAPS.get(str(row["dataset"]))
        if row["family"] == "unet_resnet50" and expected is not None:
            if abs(float(row["gap_pct"]) - expected) > 0.15:
                raise AssertionError(
                    f"{row['dataset']}: gap {float(row['gap_pct']):.2f} % does not match the "
                    f"reported {expected:.1f} %"
                )

    import matplotlib.pyplot as plt

    families = [("unet_resnet50", "o", style.PALETTE["cvc_unet"], "UNet")]
    if (points["family"] == "swinunetr_2d").any():
        families.append(("swinunetr_2d", "s", style.PALETTE["cvc_swin"], "SwinUNETR"))

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    internal_gain = None
    for family, marker, colour, family_label in families:
        block = points.loc[points["family"] == family]
        if block.empty:
            continue
        internal_gain = float(block["gain_rel_pct_internal"].iloc[0])
        for _, row in block.iterrows():
            gap = float(row["gap_pct"])
            gain = float(row["gain_rel_pct_external"])
            ax.scatter(
                gap,
                gain,
                s=44,
                marker=marker,
                facecolor=colour,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                label=family_label,
            )
            dx, dy, alignment = LABEL_OFFSETS.get(str(row["dataset"]), (8, 3, "left"))
            ax.annotate(
                LABELS.get(str(row["dataset"]), str(row["dataset"])),
                (gap, gain),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=alignment,
                fontsize=6.5,
                color="#303030",
                zorder=4,
            )
    if internal_gain is not None:
        ax.axhline(
            internal_gain,
            color="#909090",
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )
        ax.annotate(
            f"internal TTA-8 gain {internal_gain:.1f} %",
            (1.0, internal_gain),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, 4),
            ha="right",
            va="bottom",
            fontsize=6,
            color="#606060",
            zorder=4,
        )
        ax.scatter(
            0.0,
            internal_gain,
            s=44,
            marker="X",
            facecolor=style.PALETTE["cvc_unet"],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            "CVC-ClinicDB (internal)",
            (0.0, internal_gain),
            textcoords="offset points",
            xytext=(9, -10),
            ha="left",
            fontsize=6.5,
            color="#303030",
            zorder=4,
        )
    gaps = points["gap_pct"].astype(float)
    gains = points["gain_rel_pct_external"].astype(float)
    lowest = min([float(gains.min())] + ([internal_gain] if internal_gain is not None else []))
    highest = max([float(gains.max())] + ([internal_gain] if internal_gain is not None else []))
    ax.set_xlim(-2.0, float(gaps.max()) + 6.0)
    ax.set_ylim(lowest - 2.6, highest + 2.0)
    ax.set_xlabel("domain gap: baseline Dice below internal  [%]")
    ax.set_ylabel("relative TTA-8 Dice gain  [%]")
    ax.set_title("TTA gain holds or grows under domain shift")
    if len(families) > 1:
        ax.legend(loc="upper left", fontsize=6)
    style.save(fig, "fig02_internal_vs_external")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

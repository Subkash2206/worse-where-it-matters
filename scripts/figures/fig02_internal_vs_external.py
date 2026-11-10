"""Figure 2 - does TTA lose value outside the training distribution?

x is the *domain gap*: how far the baseline Dice of one external collection falls below the internal
CVC-ClinicDB baseline of the same architecture, in relative percent.  The internal reference is the
per-case mean baseline Dice of the internal UNet, ``0.7763`` over the 612 test cases of the
cross-validation grid - a constant of the result tables, so the gap can be re-derived by hand as
``(0.7763 - dice_external) / 0.7763 * 100`` and is printed for every collection.  The internal point
itself sits at x = 0.  y is the relative TTA-8 Dice gain on that collection, with a **95 % cluster
bootstrap** interval (external: one cluster per case; internal: one cluster per video, the 29
sequences of CVC-ClinicDB), and the dashed line marks the internal TTA-8 gain (3.7 %): points above it
gain more than on the training distribution.

The bootstrap resamples the clusters and recomputes the same statistic the tables report,
``(mean dice_tta8 - mean dice_baseline) / mean dice_baseline * 100`` over the per-case Dice averaged
across the 15 checkpoints, so the point estimate and the interval belong to the same estimator.

Reads ``results/external_by_model/comparison.csv`` for the point estimates and the per-case tables for
the bootstrap; SwinUNETR points appear automatically once the external_swin queue has written its
runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import style  # noqa: E402

COMPARISON = style.RESULTS / "external_by_model" / "comparison.csv"
INTERNAL_RUNS = style.RESULTS / "cvc_clinicdb_unet_resnet50" / "evaluation"
INTERNAL_DICE = 0.7763  # per-case mean baseline Dice of the internal CVC-ClinicDB UNet
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 2024
BOOTSTRAP_CI = 0.95
LABELS = {
    "kvasir_seg": "Kvasir-SEG",
    "cvc_colondb": "CVC-ColonDB",
    "etis_larib": "ETIS-Larib",
}
#: label anchor per collection, so the annotations never sit on top of each other or on the marker
LABEL_OFFSETS = {
    "kvasir_seg": (8, 3, "left"),
    "cvc_colondb": (8, 4, "left"),
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


def _per_case_dice(directory: Path) -> pd.DataFrame:
    """Per-case baseline and TTA-8 Dice, averaged over the checkpoints, with the cluster id."""
    frames = []
    for path in sorted(directory.glob("*_per_case.csv")):
        frame = pd.read_csv(path, usecols=["case_id", "group_id", "method", "metric", "value"])
        frames.append(
            frame.loc[(frame["metric"] == "dice") & (frame["method"].isin(["baseline", "tta8"]))]
        )
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    table["case_id"] = table["case_id"].astype(str)
    table["group_id"] = table["group_id"].astype(str)
    wide = table.pivot_table(
        index=["case_id", "group_id"], columns="method", values="value", aggfunc="mean"
    ).reset_index()
    return wide.rename(columns={"baseline": "dice_baseline", "tta8": "dice_tta8"})


def _relative_gain(base: np.ndarray, treated: np.ndarray) -> float:
    """Relative Dice gain of one draw, the statistic the comparison table divides."""
    return float((treated.mean() - base.mean()) / base.mean() * 100.0)


def bootstrap_gain(table: pd.DataFrame, *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED):
    """Point estimate and 95 % cluster-bootstrap interval of the relative TTA-8 Dice gain.

    Clusters come from ``group_id``: one case per cluster on the external collections, one video per
    cluster internally.  Resampling whole clusters keeps the within-video correlation of neighbouring
    frames out of the interval, which a per-frame bootstrap would ignore.
    """
    base = table["dice_baseline"].to_numpy(dtype=float)
    treated = table["dice_tta8"].to_numpy(dtype=float)
    point = _relative_gain(base, treated)
    groups = table["group_id"].to_numpy()
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    stats = np.empty(int(draws))
    for draw in range(int(draws)):
        picked = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([members[group] for group in picked])
        stats[draw] = _relative_gain(base[index], treated[index])
    low, high = np.percentile(stats, [100 * (1 - BOOTSTRAP_CI) / 2, 100 * (1 + BOOTSTRAP_CI) / 2])
    return point, float(low), float(high), int(unique.size), int(base.size)



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

    # 95 % cluster-bootstrap interval of the relative TTA-8 gain (case for external, video internal)
    print("\n95 % cluster-bootstrap interval of the relative TTA-8 Dice gain:")
    intervals: dict[tuple[str, str], tuple[float, float, float, int, int]] = {}
    for family, dataset in points[["family", "dataset"]].itertuples(index=False):
        directory = (
            INTERNAL_RUNS
            if str(dataset).startswith("cvc_clinicdb")
            else style.RESULTS / f"external_{dataset}" / "evaluation"
        )
        if family != "unet_resnet50" or not directory.is_dir() or str(dataset) == "cvc_clinicdb":
            continue
        case_table = _per_case_dice(directory)
        if case_table.empty:
            continue
        point, low, high, n_clusters, n_cases = bootstrap_gain(case_table)
        intervals[(str(family), str(dataset))] = (point, low, high, n_clusters, n_cases)
        table_gain = float(
            points.loc[
                (points["family"] == family) & (points["dataset"] == dataset),
                "gain_rel_pct_external",
            ].iloc[0]
        )
        print(
            f"  {str(dataset):12s} gain {point:5.2f} % (table {table_gain:5.2f} %) "
            f"95 % CI [{low:5.2f}, {high:5.2f}] "
            f"over {n_cases} cases in {n_clusters} case clusters"
        )
        if abs(point - table_gain) > 0.05:
            raise AssertionError(
                f"{dataset}: bootstrap point estimate {point:.2f} % does not reproduce the table's "
                f"{table_gain:.2f} %"
            )
    internal_table = _per_case_dice(INTERNAL_RUNS)
    internal_ci = None
    if not internal_table.empty:
        point, low, high, n_videos, n_cases = bootstrap_gain(internal_table)
        internal_ci = (low, high)
        videos_per_frame = (internal_table.groupby("group_id").size() > 1).any()
        if not videos_per_frame:
            raise AssertionError("the internal clusters are not videos (one frame per group)")
        print(
            f"  cvc_clinicdb  gain {point:5.2f} % 95 % CI [{low:5.2f}, {high:5.2f}] "
            f"over {n_cases} frames in {n_videos} video clusters"
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
            interval = intervals.get((str(family), str(row["dataset"])))
            lower = upper = 0.0
            if interval is not None:
                _, low, high, _, _ = interval
                lower, upper = gain - low, high - gain
            ax.errorbar(
                gap,
                gain,
                yerr=[[lower], [upper]],
                fmt=marker,
                markersize=6.0,
                markerfacecolor=colour,
                markeredgecolor="white",
                markeredgewidth=0.6,
                ecolor=colour,
                elinewidth=0.8,
                capsize=2.0,
                capthick=0.7,
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
            xytext=(0, -5),
            ha="right",
            va="top",
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
        if internal_ci is not None:
            ax.errorbar(
                0.0,
                internal_gain,
                yerr=[[internal_gain - internal_ci[0]], [internal_ci[1] - internal_gain]],
                fmt="none",
                ecolor=style.PALETTE["cvc_unet"],
                elinewidth=0.8,
                capsize=2.0,
                capthick=0.7,
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
    limits = list(gains.astype(float))
    if internal_gain is not None:
        limits.append(float(internal_gain))
    for _, low, high, _, _ in intervals.values():
        limits.extend([low, high])
    if internal_ci is not None:
        limits.extend(internal_ci)
    ax.set_xlim(-2.0, float(gaps.max()) + 6.0)
    ax.set_ylim(min(limits) - 1.0, max(limits) + 1.4)
    ax.set_xlabel("domain gap: baseline Dice below internal  [%]")
    ax.set_ylabel("relative TTA-8 Dice gain  [%]")
    ax.set_title("TTA gain holds or grows under domain shift")
    if len(families) > 1:
        ax.legend(loc="upper left", fontsize=6)
    style.save(fig, "fig02_internal_vs_external")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

"""Figure 2 - what the domain shift does to the TTA-8 Dice gain.

x is the *domain gap*: how far the baseline Dice of one external collection falls below that
architecture's own internal CVC-ClinicDB baseline, in relative percent (``dice_internal`` in the
comparison table, so the SwinUNETR points are not measured against the UNet).  y is the **absolute**
TTA-8 Dice gain, ``mean dice_tta8 - mean dice_baseline`` over the cases - the same quantity the
paired table reports - with a 95 % cluster bootstrap interval (external: one case per cluster,
internal: one video per cluster).  A dashed line per architecture marks its internal absolute gain
(UNet 0.028, SwinUNETR 0.021): points above it gain more than on the training distribution.

Under the shift the UNet keeps gaining (0.041 / 0.025 / 0.084 against 0.028 internally) while the
SwinUNETR collapses to about a third of its internal gain (0.006 / 0.009 / 0.008 against 0.021).

Reads ``results/external_by_model/comparison.csv`` for the point estimates and the per-case tables for
the bootstrap.
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
#: label anchor per (collection, architecture); the two architectures need different sides
LABEL_OFFSETS = {
    ("kvasir_seg", "unet_resnet50"): (8, 3, "left"),
    ("kvasir_seg", "swinunetr_2d"): (8, 2, "left"),
    ("cvc_colondb", "unet_resnet50"): (8, 10, "left"),
    ("cvc_colondb", "swinunetr_2d"): (-9, -10, "right"),
    ("etis_larib", "unet_resnet50"): (-9, -12, "right"),
    ("etis_larib", "swinunetr_2d"): (-9, 6, "right"),
}
#: gaps the internal UNet must reproduce (relative percent below 0.7763)
EXPECTED_GAPS = {"kvasir_seg": 7.8, "cvc_colondb": 18.7, "etis_larib": 40.4}
#: internal absolute TTA-8 Dice gain each architecture must reproduce
EXPECTED_INTERNAL_GAIN = {"unet_resnet50": 0.028, "swinunetr_2d": 0.021}


def collect(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (family, collection): domain gap and the absolute TTA-8 Dice gains.

    The gap is relative to *that architecture's own* internal baseline Dice (``dice_internal`` in
    the comparison table), so the SwinUNETR points are not measured against the UNet.  The y value
    is the absolute gain, ``gain_vs_baseline_external``, not a percentage of the baseline.
    """
    baseline = table.loc[
        table["method"] == "baseline", ["family", "dataset", "dice_external", "dice_internal"]
    ].copy()
    final = table.loc[
        table["method"] == "tta8",
        ["family", "dataset", "gain_vs_baseline_external", "gain_vs_baseline_internal"],
    ].copy()
    merged = baseline.merge(final, on=["family", "dataset"], how="inner")
    merged["gap_pct"] = (
        (merged["dice_internal"].astype(float) - merged["dice_external"].astype(float))
        * 100.0
        / merged["dice_internal"].astype(float)
    )
    merged["gain_abs"] = merged["gain_vs_baseline_external"].astype(float)
    merged["gain_abs_internal"] = merged["gain_vs_baseline_internal"].astype(float)
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


def family_runs(family: str, dataset: str) -> Path | None:
    """Directory holding the per-case CSVs of one (family, dataset), new layout first."""
    stem = "swinunetr_2d" if family == "swinunetr_2d" else "unet_resnet50"
    candidates = [
        style.RESULTS / f"external_{dataset}" / stem / "evaluation",
        style.RESULTS / f"external_{dataset}" / "evaluation",
    ]
    for path in candidates:
        if path.is_dir() and any(path.glob("*_per_case.csv")):
            return path
    return None


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
    """Absolute Dice gain of one draw, the statistic the comparison table reports."""
    return float(treated.mean() - base.mean())


def bootstrap_gain(table: pd.DataFrame, *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED):
    """Point estimate and 95 % cluster-bootstrap interval of the absolute TTA-8 Dice gain.

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

    print("domain gap against each architecture's own internal baseline Dice (absolute TTA-8 gain):")
    for _, row in points.iterrows():
        print(
            f"  {str(row['dataset']):12s} ({str(row['family']):14s}) "
            f"internal Dice {float(row['dice_internal']):.4f} -> external "
            f"{float(row['dice_external']):.4f} = gap {float(row['gap_pct']):5.1f} % "
            f"(UNet reference {EXPECTED_GAPS.get(str(row['dataset']), float('nan')):5.1f} %), "
            f"TTA-8 gain {float(row['gain_abs']):.4f} (internal {float(row['gain_abs_internal']):.4f})"
        )
        expected = EXPECTED_GAPS.get(str(row["dataset"]))
        if row["family"] == "unet_resnet50" and expected is not None:
            if abs(float(row["gap_pct"]) - expected) > 0.15:
                raise AssertionError(
                    f"{row['dataset']}: gap {float(row['gap_pct']):.2f} % does not match the "
                    f"reported {expected:.1f} %"
                )
        internal_expected = EXPECTED_INTERNAL_GAIN.get(str(row["family"]))
        if internal_expected is not None and abs(float(row["gain_abs_internal"]) - internal_expected) > 0.001:
            raise AssertionError(
                f"{row['family']}: internal TTA-8 gain {float(row['gain_abs_internal']):.4f} does "
                f"not match the documented {internal_expected:.3f}"
            )

    # 95 % cluster-bootstrap interval of the absolute TTA-8 gain (case external, video internal)
    print("\n95 % cluster-bootstrap interval of the absolute TTA-8 Dice gain:")
    intervals: dict[tuple[str, str], tuple[float, float, float, int, int]] = {}
    for family, dataset in points[["family", "dataset"]].itertuples(index=False):
        directory = family_runs(str(family), str(dataset))
        if directory is None:
            continue
        case_table = _per_case_dice(directory)
        if case_table.empty:
            continue
        point, low, high, n_clusters, n_cases = bootstrap_gain(case_table)
        intervals[(str(family), str(dataset))] = (point, low, high, n_clusters, n_cases)
        table_gain = float(
            points.loc[
                (points["family"] == family) & (points["dataset"] == dataset),
                "gain_abs",
            ].iloc[0]
        )
        print(
            f"  {str(dataset):12s} {str(family):14s} gain {point:+.4f} (table {table_gain:+.4f}) "
            f"95 % CI [{low:+.4f}, {high:+.4f}] "
            f"over {n_cases} cases in {n_clusters} case clusters"
        )
        if abs(point - table_gain) > 1e-4:
            raise AssertionError(
                f"{dataset}/{family}: bootstrap point estimate {point:.4f} does not reproduce "
                f"the table's {table_gain:.4f}"
            )
    internal_cis: dict[str, tuple[float, float]] = {}
    for family, runs in (
        ("unet_resnet50", INTERNAL_RUNS),
        ("swinunetr_2d", style.RESULTS / "cvc_clinicdb_swinunetr_2d" / "evaluation"),
    ):
        if not any(runs.glob("*_per_case.csv")):
            continue
        internal_table = _per_case_dice(runs)
        point, low, high, n_videos, n_cases = bootstrap_gain(internal_table)
        internal_cis[family] = (low, high)
        if not (internal_table.groupby("group_id").size() > 1).any():
            raise AssertionError("the internal clusters are not videos (one frame per group)")
        print(
            f"  cvc_clinicdb  {family:14s} gain {point:+.4f} 95 % CI [{low:+.4f}, {high:+.4f}] "
            f"over {n_cases} frames in {n_videos} video clusters"
        )

    import matplotlib.pyplot as plt

    families = [("unet_resnet50", "o", style.PALETTE["cvc_unet"], "UNet")]
    if (points["family"] == "swinunetr_2d").any():
        families.append(("swinunetr_2d", "s", style.PALETTE["cvc_swin"], "SwinUNETR"))

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    internal_gains: dict[str, float] = {}
    for family, marker, colour, family_label in families:
        block = points.loc[points["family"] == family]
        if block.empty:
            continue
        internal_gain = float(block["gain_abs_internal"].iloc[0])
        internal_gains[family] = internal_gain
        labelled = False
        for _, row in block.iterrows():
            gap = float(row["gap_pct"])
            gain = float(row["gain_abs"])
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
                label=f"{family_label} (internal {internal_gain:.3f})" if not labelled else None,
            )
            labelled = True
            dx, dy, alignment = LABEL_OFFSETS.get((str(row["dataset"]), str(family)), (8, 3, "left"))
            ax.annotate(
                LABELS.get(str(row["dataset"]), str(row["dataset"])) + f" ({family_label[:4]})",
                (gap, gain),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=alignment,
                fontsize=6.0,
                color="#303030",
                zorder=4,
            )
        # the architecture's own internal reference at x = 0
        ax.axhline(internal_gain, color=colour, linestyle="--", linewidth=0.8, zorder=1)
        ax.scatter(
            0.0,
            internal_gain,
            s=44,
            marker="X",
            facecolor=colour,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        interval = internal_cis.get(family)
        if interval is not None:
            ax.errorbar(
                0.0,
                internal_gain,
                yerr=[[internal_gain - interval[0]], [interval[1] - internal_gain]],
                fmt="none",
                ecolor=colour,
                elinewidth=0.8,
                capsize=2.0,
                capthick=0.7,
                zorder=3,
            )
        # the architecture's own internal gain is named in the legend, not on the line, so the two
        # nearly identical dashed references cannot collide with the external points
    gaps = points["gap_pct"].astype(float)
    gains = points["gain_abs"].astype(float)
    limits = list(gains.astype(float)) + list(internal_gains.values())
    for _, low, high, _, _ in intervals.values():
        limits.extend([low, high])
    for low, high in internal_cis.values():
        limits.extend([low, high])
    ax.set_xlim(-2.0, float(gaps.max()) + 8.0)
    ax.set_ylim(min(limits) - 0.004, max(limits) + 0.008)
    ax.set_xlabel("domain gap: baseline Dice below its own internal baseline  [%]")
    ax.set_ylabel("absolute TTA-8 Dice gain")
    ax.set_title(
        "Under shift, the TTA gain holds or grows for UNet and shrinks to a third for SwinUNETR",
        fontsize=7.0,
    )
    if len(families) > 1:
        ax.legend(loc="upper left", fontsize=6, title="architecture", title_fontsize=6)

    style.save(fig, "fig02_internal_vs_external")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

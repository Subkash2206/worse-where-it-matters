"""Figure 1 - TTA gain against the number of views (1, 2, 4, 8).

Two panels share the x axis (views on a log-2 scale): the absolute Dice change and the *relative*
HD95 change against the single-view baseline, one line per setting, with the 95 % bootstrap CI of
the paired per-case delta as error bars.  Everything is read from the paired tables the aggregation
already wrote (``paired_seed_averaged.csv`` internally, ``external_by_model/paired.csv`` externally),
so the figure never recomputes a statistic the tables do not contain.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import style  # noqa: E402

VIEWS = {"tta2": 2, "tta4": 4, "tta8": 8}
SOURCES = (
    ("cvc_unet", style.RESULTS / "aggregate_cvc_clinicdb_unet_resnet50" / "paired_seed_averaged.csv"),
    ("cvc_swin", style.RESULTS / "aggregate_cvc_clinicdb_swinunetr_2d" / "paired_seed_averaged.csv"),
    ("brats", style.RESULTS / "aggregate_brats2021_segresnet_reduced" / "paired_seed_averaged.csv"),
)
EXTERNAL = style.RESULTS / "external_by_model" / "paired.csv"
#: the CVC-ColonDB and SwinUNETR HD95 curves nearly coincide, so their x positions are nudged
#: by +/-5 % (the error bars stay on the same scale; the offset is a plotting aid only)
X_OFFSET = {"cvc_colondb": 0.95, "cvc_swin": 1.05}
#: legend order: the two internal 2-D models, BraTS, then the three external collections
_LEGEND_ORDER = {
    label: position
    for position, label in enumerate(
        style.LABELS[key]
        for key in ("cvc_unet", "cvc_swin", "brats", "kvasir_seg", "cvc_colondb", "etis_larib")
    )
}
#: external collections: their UNet and SwinUNETR runs share a colour and differ by line style
EXTERNAL_KEYS = ("kvasir_seg", "cvc_colondb", "etis_larib")


def _setting_style(setting: str) -> tuple[str, str, str, str, tuple[int, int]]:
    """Colour, marker, linestyle, label and legend sort key of one series.

    ``<dataset>_swin`` is the SwinUNETR run of an external collection: same colour and marker as
    the UNet run, but a dashed line and an open marker so both architectures stay visible.
    """
    swin = setting.endswith("_swin") and setting[: -len("_swin")] in EXTERNAL_KEYS
    base = setting[: -len("_swin")] if swin else setting
    colour = style.PALETTE.get(base, "#555555")
    marker = style.MARKERS.get(base, "o")
    label = style.LABELS.get(base, base)
    order = (_LEGEND_ORDER.get(label, 99), 1 if swin else 0)
    if swin:
        return colour, marker, "--", label + " \u00b7 SwinUNETR", order
    if base in EXTERNAL_KEYS:
        return colour, marker, "-", label + " \u00b7 UNet", order
    return colour, marker, "-", label, order



def _rows(frame: pd.DataFrame, key: str, family: str = "unet_resnet50") -> list[dict]:
    """One entry per (views, metric) with the delta, its CI and the relative CI.

    ``key`` is the dataset and ``family`` the external architecture; a SwinUNETR run is stored
    under ``<dataset>_swin`` so the two architectures never share a line.
    """
    rows: list[dict] = []
    control = "mean_control" if "mean_control" in frame.columns else "control_mean"
    if "treatment" not in frame.columns and "comparison" in frame.columns:
        frame = frame.assign(treatment=frame["comparison"].str.split().str[0])
    setting = key if family == "unet_resnet50" else f"{key}_swin"
    for metric in ("dice", "hd95"):
        block = frame.loc[frame["metric"] == metric]
        if block.empty:
            continue
        base = float(block[control].iloc[0])
        for treatment, views in VIEWS.items():
            row = block.loc[block["treatment"] == treatment]
            if row.empty:
                continue
            row = row.iloc[0]
            scale = 100.0 / base if metric == "hd95" and base else 1.0
            rows.append(
                {
                    "setting": setting,
                    "metric": metric,
                    "views": views,
                    "delta": float(row["delta_mean"]) * scale,
                    "low": float(row["ci_low"]) * scale,
                    "high": float(row["ci_high"]) * scale,
                }
            )
    return rows


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    for key, path in SOURCES:
        if path.is_file():
            rows.extend(_rows(pd.read_csv(path), key))
    if EXTERNAL.is_file():
        external = pd.read_csv(EXTERNAL)
        for (key, family), block in external.groupby(["dataset", "family"]):
            rows.extend(_rows(block, str(key), str(family)))
    # the single-view reference is zero by construction
    frame = pd.DataFrame(rows)
    reference = frame.loc[:, ["setting", "metric"]].drop_duplicates()
    zeros = reference.assign(views=1, delta=0.0, low=0.0, high=0.0)
    return pd.concat([frame, zeros], ignore_index=True).sort_values(["metric", "setting", "views"])


def main() -> int:
    style.apply()
    table = collect()
    if table.empty:
        print("no paired tables found", file=sys.stderr)
        return 2
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    handles = []
    for ax, metric, label in zip(axes, ("dice", "hd95"), ("Dice change", "relative HD95 change [%]")):
        block = table.loc[table["metric"] == metric]
        for setting, group in block.groupby("setting"):
            group = group.sort_values("views")
            offset = X_OFFSET.get(str(setting), 1.0)
            colour, marker, linestyle, series_label, order = _setting_style(str(setting))
            line = ax.errorbar(
                group["views"] * offset,
                group["delta"],
                yerr=[group["delta"] - group["low"], group["high"] - group["delta"]],
                marker=marker,
                color=colour,
                linestyle=linestyle,
                markerfacecolor="white" if str(setting).endswith("_swin") else colour,
                label=series_label,
                linewidth=1.2,
                elinewidth=0.7,
                capsize=2,
            )
            if metric == "dice":
                handles.append((order, line))
        ax.axhline(0.0, color="#909090", linewidth=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_xlim(0.9, 8.6)
        ax.set_xlabel("number of TTA views")
        ax.set_ylabel(label)
    axes[0].set_title("(a) Dice", loc="left")
    axes[1].set_title("(b) HD95 (relative)", loc="left")
    axes[1].annotate(
        "±5 % x-jitter: ColonDB, SwinUNETR",
        (0.965, 0.03),
        xycoords="axes fraction",
        ha="right",
        fontsize=5.0,
        color="#707070",
    )
    # one legend for both panels, outside the axes so it cannot cover a curve
    handles = [line for _, line in sorted(handles, key=lambda item: item[0])]
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=6.5,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    style.save(fig, "fig01_gain_vs_views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

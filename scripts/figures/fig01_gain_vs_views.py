"""Figure 1 - TTA gain against the number of views (1, 2, 4, 8).

Two panels share the x axis (views on a log-2 scale): the absolute Dice change and the *relative*
HD95 change against the single-view baseline, one line per setting, with the 95 % bootstrap CI of
the paired per-case delta as error bars.  Everything is read from the paired tables the aggregation
already wrote (``paired_seed_averaged.csv`` internally, ``external_by_model/paired.csv`` externally),
so the figure never recomputes a statistic the tables do not contain.  Colour encodes the dataset and
the line style + marker the architecture - solid + filled circle for the UNet, dash-dot + filled
triangle for the SegFormer, dashed + open square for the SwinUNETR - for every pair, the internal
CVC-ClinicDB grids included; the legend names are ``<dataset> · <model>``.
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
    (
        "cvc_segformer",
        style.RESULTS / "aggregate_cvc_clinicdb_segformer_mit_b2" / "paired_seed_averaged.csv",
    ),
    ("cvc_swin", style.RESULTS / "aggregate_cvc_clinicdb_swinunetr_2d" / "paired_seed_averaged.csv"),
    ("brats", style.RESULTS / "aggregate_brats2021_segresnet_reduced" / "paired_seed_averaged.csv"),
)
EXTERNAL = style.RESULTS / "external_by_model" / "paired.csv"
#: the CVC-ColonDB and SwinUNETR HD95 curves nearly coincide, so their x positions are nudged
#: by +/-5 % (the error bars stay on the same scale; the offset is a plotting aid only)
X_OFFSET = {"cvc_colondb": 0.95, "cvc_swin": 1.05}
#: architecture of every internal source (the external tables carry it in their ``family`` column)
INTERNAL_FAMILY = {
    "cvc_unet": "unet_resnet50",
    "cvc_segformer": "segformer_mit_b2",
    "cvc_swin": "swinunetr_2d",
    "brats": "segresnet",
}
#: colour = dataset: the three internal CVC-ClinicDB models share the blue of that dataset
DATASET_COLOUR_KEY = {
    "cvc_unet": "cvc_unet",
    "cvc_segformer": "cvc_unet",
    "cvc_swin": "cvc_unet",
    "brats": "brats",
    "kvasir_seg": "kvasir_seg",
    "cvc_colondb": "cvc_colondb",
    "etis_larib": "etis_larib",
}
#: marker + line style = architecture (the SegResNet keeps the diamond of the BraTS figures)
MODEL_STYLES = {
    **style.MODEL_STYLES,
    "segresnet": ("-", "D", True),
}
MODEL_TEXT = {**style.MODEL_LABELS, "segresnet": "SegResNet"}
#: legend order: the three internal 2-D models, BraTS, then the three external collections
_LEGEND_ORDER = {
    label: position
    for position, label in enumerate(
        style.LABELS[key]
        for key in ("cvc_unet", "cvc_segformer", "cvc_swin", "brats", "kvasir_seg",
                    "cvc_colondb", "etis_larib")
    )
}
#: short collection names for the "<dataset> · <model>" legend entries
SHORT_NAMES = {
    "cvc_unet": "CVC-ClinicDB",
    "cvc_segformer": "CVC-ClinicDB",
    "cvc_swin": "CVC-ClinicDB",
    "brats": "BraTS 2021",
    "kvasir_seg": "Kvasir-SEG",
    "cvc_colondb": "CVC-ColonDB",
    "etis_larib": "ETIS-Larib",
}
#: rank of the architecture inside one dataset's legend block
MODEL_RANK = {"unet_resnet50": 0, "segformer_mit_b2": 1, "swinunetr_2d": 2, "segresnet": 3}


def _setting_style(setting: str) -> tuple[str, str, str, str, tuple[int, int]]:
    """Colour, marker, linestyle, label and legend sort key of one series.

    Colour encodes the dataset and marker + line style the architecture: every pair - including the
    three internal CVC-ClinicDB grids - is solid + filled circle for the UNet, dash-dot + filled
    triangle for the SegFormer and dashed + open square for the SwinUNETR.  An external setting is
    written ``<dataset>::<family>``; an internal one is the dataset key of ``SOURCES``.
    """
    dataset, _, family = setting.partition("::")
    if not family:
        family = INTERNAL_FAMILY.get(dataset, "unet_resnet50")
    colour_key = DATASET_COLOUR_KEY.get(dataset, dataset)
    colour = style.PALETTE.get(colour_key, "#555555")
    linestyle, marker, filled = MODEL_STYLES.get(family, ("-", "o", True))
    label = f"{SHORT_NAMES.get(dataset, dataset)} · {MODEL_TEXT.get(family, family)}"
    order = (_LEGEND_ORDER.get(style.LABELS.get(dataset, dataset), 99), MODEL_RANK.get(family, 9))
    return colour, marker, linestyle, label, order, filled




def _rows(frame: pd.DataFrame, key: str, family: str = "unet_resnet50") -> list[dict]:
    """One entry per (views, metric) with the delta, its CI and the relative CI.

    ``key`` is the dataset and ``family`` the external architecture; a SwinUNETR run is stored
    under ``<dataset>_swin`` so the two architectures never share a line.
    """
    rows: list[dict] = []
    control = "mean_control" if "mean_control" in frame.columns else "control_mean"
    if "treatment" not in frame.columns and "comparison" in frame.columns:
        frame = frame.assign(treatment=frame["comparison"].str.split().str[0])
    setting = key if family == "unet_resnet50" else f"{key}::{family}"
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
            offset = X_OFFSET.get(str(setting).partition("::")[0], 1.0)
            colour, marker, linestyle, series_label, order, filled = _setting_style(str(setting))
            line = ax.errorbar(
                group["views"] * offset,
                group["delta"],
                yerr=[group["delta"] - group["low"], group["high"] - group["delta"]],
                marker=marker,
                color=colour,
                linestyle=linestyle,
                markerfacecolor=colour if filled else "white",
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

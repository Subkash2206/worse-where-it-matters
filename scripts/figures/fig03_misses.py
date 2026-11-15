"""Figure 3 - how often TTA loses the polyp entirely.

Two families of panels, because the three families define a miss differently and the numbers must
not be pooled:

* (a) internal CVC-ClinicDB (UNet, SegFormer and SwinUNETR) and BraTS: an **empty prediction on a
  case whose reference is non-empty**, read from the seed-averaged detection table (case level, one
  unit per case), with the Holm-adjusted McNemar p of the paired transition;
* (b) and (c) external polyp collections, for the UNet and the SegFormer separately: the share of
  cases that a **majority of the 15 checkpoints** miss, with a Jeffreys interval and the McNemar p of
  the case-majority transition - the definition the external validation reports, and the conservative
  companion of the per-checkpoint count.  The SwinUNETR gets no panel: it never missed a polyp by
  majority on any external collection, so its bars would be a row of zeros (the run prints that).

The methods are encoded as grey shades (one legend for the two panels); the p-value of every paired
comparison sits horizontally above its pair of bars, and the panel keeps enough headroom that neither
the p-values nor the legend touch the title.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import style  # noqa: E402

INTERNAL = (
    ("cvc_unet", style.RESULTS / "aggregate_cvc_clinicdb_unet_resnet50" / "detection_seed_averaged.csv"),
    (
        "cvc_segformer",
        style.RESULTS / "aggregate_cvc_clinicdb_segformer_mit_b2" / "detection_seed_averaged.csv",
    ),
    ("cvc_swin", style.RESULTS / "aggregate_cvc_clinicdb_swinunetr_2d" / "detection_seed_averaged.csv"),
    ("brats", style.RESULTS / "aggregate_brats2021_segresnet_reduced" / "detection_seed_averaged.csv"),
)
EXTERNAL = style.RESULTS / "external_by_model" / "misses.csv"
METHODS = ("baseline", "tta4", "tta8")
INTERNAL_ORDER = ("cvc_unet", "cvc_segformer", "cvc_swin", "brats")
#: panel (b) shows the two 2-D polyp models; the SwinUNETR misses no polyp by majority at all
EXTERNAL_FAMILIES = ("unet_resnet50", "segformer_mit_b2")
EXTERNAL_DATASETS = ("kvasir_seg", "cvc_colondb", "etis_larib")
#: grey shades, light to dark per method (the legend is shared by both panels)
GREY = {"baseline": "#333333", "tta4": "#8a8a8a", "tta8": "#c6c6c6"}
#: the method names as every other figure writes them
METHOD_TEXT = {"baseline": "baseline", "tta4": "TTA-4", "tta8": "TTA-8"}
#: two-line tick labels: the full names are too wide for a two-panel figure
SHORT_LABELS = {
    "cvc_unet": "CVC-ClinicDB\nUNet",
    "cvc_segformer": "CVC-ClinicDB\nSegFormer",
    "cvc_swin": "CVC-ClinicDB\nSwinUNETR",
    "brats": "BraTS 2021\nSegResNet",
    "kvasir_seg::unet_resnet50": "Kvasir-SEG\nUNet",
    "kvasir_seg::segformer_mit_b2": "Kvasir-SEG\nSegFormer",
    "kvasir_seg::swinunetr_2d": "Kvasir-SEG\nSwinUNETR",
    "cvc_colondb::unet_resnet50": "CVC-ColonDB\nUNet",
    "cvc_colondb::segformer_mit_b2": "CVC-ColonDB\nSegFormer",
    "cvc_colondb::swinunetr_2d": "CVC-ColonDB\nSwinUNETR",
    "etis_larib::unet_resnet50": "ETIS-Larib\nUNet",
    "etis_larib::segformer_mit_b2": "ETIS-Larib\nSegFormer",
    "etis_larib::swinunetr_2d": "ETIS-Larib\nSwinUNETR",
    "kvasir_seg": "Kvasir-SEG",
    "cvc_colondb": "CVC-ColonDB",
    "etis_larib": "ETIS-Larib",
}



def jeffreys(successes: float, total: float) -> tuple[float, float]:
    """95 % Jeffreys interval of a binomial rate (the interval the external tables use)."""
    from scipy import stats

    if total <= 0:
        return float("nan"), float("nan")
    low = stats.beta.ppf(0.025, successes + 0.5, total - successes + 0.5)
    high = stats.beta.ppf(0.975, successes + 0.5, total - successes + 0.5)
    return float(low), float(high)


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    for key, path in INTERNAL:
        if not path.is_file():
            continue
        table = pd.read_csv(path).set_index("method")
        for method in METHODS:
            if method not in table.index:
                continue
            row = table.loc[method]
            rows.append(
                {
                    "dataset": key,
                    "method": method,
                    "rate": float(row["missed_rate"]),
                    "low": float(row["ci_low"]),
                    "high": float(row["ci_high"]),
                    "p_value": float(row["mcnemar_p_adjusted"]) if method != "baseline" else np.nan,
                }
            )
    if EXTERNAL.is_file():
        table = pd.read_csv(EXTERNAL)
        for (dataset, family), block in table.groupby(["dataset", "family"]):
            if family not in EXTERNAL_FAMILIES:
                continue  # the SwinUNETR never misses a polyp by majority (see the printed note)
            key = f"{dataset}::{family}"
            for method in METHODS:
                row = block.loc[block["method"] == method]
                if row.empty:
                    continue
                row = row.iloc[0]
                total = float(row["n_cases"])
                missed = float(row["cases_majority_missed"])
                low, high = jeffreys(missed, total)
                rows.append(
                    {
                        "dataset": key,
                        "method": method,
                        "rate": missed / total,
                        "low": low,
                        "high": high,
                        "p_value": float(row["mcnemar_p_case_majority"])
                        if method != "baseline"
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _p_text(p_value: float) -> str:
    """Compact McNemar p for a figure label."""
    if not np.isfinite(p_value):
        return ""
    if p_value > 0.05:
        return "n.s."
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def _panel(
    ax, table: pd.DataFrame, order: tuple[str, ...], title: str, ylabel: str, *, rotate: float = 0.0
) -> None:
    """Grouped bars for the datasets of one panel, p-values horizontally above each pair."""
    datasets = [name for name in order if name in set(table["dataset"])]
    width = 0.24
    rows: dict[str, dict[str, pd.Series]] = {}
    for dataset in datasets:
        block = table.loc[table["dataset"] == dataset]
        rows[dataset] = {
            method: block.loc[block["method"] == method].iloc[0]
            for method in METHODS
            if not block.loc[block["method"] == method].empty
        }
    span = max(
        [float(row["high"]) * 100.0 for block in rows.values() for row in block.values()] or [1.0]
    )
    pad = 0.045 * span
    for index, dataset in enumerate(datasets):
        block = rows[dataset]
        tops: dict[str, float] = {}
        for offset, method in enumerate(METHODS):
            if method not in block:
                continue
            row = block[method]
            rate = float(row["rate"]) * 100.0
            high = float(row["high"]) * 100.0
            low = float(row["low"]) * 100.0
            tops[method] = max(rate, high)
            ax.bar(
                index + (offset - 1) * width,
                rate,
                width,
                color=GREY[method],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            ax.errorbar(
                index + (offset - 1) * width,
                rate,
                yerr=[[max(rate - low, 0.0)], [max(high - rate, 0.0)]],
                color="#303030",
                elinewidth=0.6,
                capsize=1.5,
                zorder=4,
            )
            if rate <= 0.0:
                # an exact zero has no bar to see: say so instead of leaving an empty slot
                ax.annotate(
                    "0",
                    (index + (offset - 1) * width, 0.0),
                    textcoords="offset points",
                    xytext=(0, 1.2),
                    ha="center",
                    va="bottom",
                    fontsize=5.2,
                    color="#505050",
                    zorder=5,
                )
        # one horizontal p label per TTA method, above the pair (baseline, that method); the label
        # sits over its own bar so the two labels of a group never shadow each other
        base_top = tops.get("baseline", 0.0)
        previous = base_top
        for level, method in enumerate(("tta4", "tta8")):
            if method not in block:
                continue
            text = _p_text(float(block[method]["p_value"]))
            if not text:
                continue
            # always at least one pad above every cap of the pair (and above the previous level)
            target = max(base_top, tops[method])
            height = max(target, previous) + pad
            previous = height
            left = index - width
            right = index + (METHODS.index(method) - 1) * width
            ax.plot([left, right], [height, height], color="#303030", linewidth=0.6, zorder=4)
            ax.annotate(
                text,
                ((left + right) / 2.0, height),
                textcoords="offset points",
                xytext=(0, 2.0),
                ha="center",
                va="bottom",
                fontsize=5.4,
                color="#303030",
            )
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(
        [SHORT_LABELS.get(name, name) for name in datasets],
        rotation=rotate,
        ha="right" if rotate else "center",
        fontsize=6.5 if rotate else 7.0,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.set_ylim(0, span * 1.05 + 2.4 * pad)


def main() -> int:
    style.apply()
    table = collect()
    if table.empty:
        print("no detection tables found", file=sys.stderr)
        return 2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (internal, external_unet, external_segformer) = plt.subplots(
        1, 3, figsize=(9.2, 3.1), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]}
    )
    _panel(
        internal,
        table,
        INTERNAL_ORDER,
        "(a) internal: empty prediction",
        "empty predictions [%]",
        rotate=22.0,
    )
    for panel, letter, family, label in (
        (external_unet, "b", "unet_resnet50", "UNet"),
        (external_segformer, "c", "segformer_mit_b2", "SegFormer"),
    ):
        block = table.loc[table["dataset"].str.endswith(f"::{family}")].copy()
        block["dataset"] = block["dataset"].str.split("::").str[0]
        _panel(
            panel,
            block,
            EXTERNAL_DATASETS,
            f"({letter}) external, {label}: majority empty",
            "majority-empty rate [%]",
            rotate=18.0,
        )
    if EXTERNAL.is_file():
        swin = pd.read_csv(EXTERNAL)
        swin = swin.loc[swin["family"] == "swinunetr_2d"]
        if not swin.empty:
            per_dataset = ", ".join(
                f"{row['dataset']} {int(row['cases_majority_missed'])}/{int(row['n_cases'])}"
                for _, row in swin.loc[swin["method"] == "tta8"].iterrows()
            )
            print(
                "caption note: panels (b) and (c) show the UNet and the SegFormer; the SwinUNETR has "
                f"no image empty by majority on any external collection (TTA-8: {per_dataset})"
            )
    handles = [
        Patch(facecolor=GREY[method], edgecolor="white", label=METHOD_TEXT[method])
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(METHODS),
        title="method (grey shade)",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=6.5,
        title_fontsize=6.5,
    )
    style.save(fig, "fig03_misses")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())


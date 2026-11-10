"""Figure 3 - how often TTA loses the polyp entirely.

Two panels, because the two families define a miss differently and the numbers must not be pooled:

* (a) internal CVC-ClinicDB (UNet and SwinUNETR) and BraTS: an **empty prediction on a case whose
  reference is non-empty**, read from the seed-averaged detection table (case level, one unit per
  case), with the Holm-adjusted McNemar p of the paired transition;
* (b) external polyp collections: the share of cases that a **majority of the 15 checkpoints** miss,
  with a Jeffreys interval and the McNemar p of the case-majority transition - the definition the
  external validation reports, and the conservative companion of the per-checkpoint count.

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
    ("cvc_swin", style.RESULTS / "aggregate_cvc_clinicdb_swinunetr_2d" / "detection_seed_averaged.csv"),
    ("brats", style.RESULTS / "aggregate_brats2021_segresnet_reduced" / "detection_seed_averaged.csv"),
)
EXTERNAL = style.RESULTS / "external_by_model" / "misses.csv"
METHODS = ("baseline", "tta4", "tta8")
INTERNAL_ORDER = ("cvc_unet", "cvc_swin", "brats")
EXTERNAL_ORDER = ("kvasir_seg", "cvc_colondb", "etis_larib")
#: grey shades, light to dark per method (the legend is shared by both panels)
GREY = {"baseline": "#333333", "tta4": "#8a8a8a", "tta8": "#c6c6c6"}
#: the method names as every other figure writes them
METHOD_TEXT = {"baseline": "baseline", "tta4": "TTA-4", "tta8": "TTA-8"}
#: two-line tick labels: the full names are too wide for a two-panel figure
SHORT_LABELS = {
    "cvc_unet": "CVC-ClinicDB\nUNet",
    "cvc_swin": "CVC-ClinicDB\nSwinUNETR",
    "brats": "BraTS 2021\nSegResNet",
    "kvasir_seg": "Kvasir-SEG\nexternal",
    "cvc_colondb": "CVC-ColonDB\nexternal",
    "etis_larib": "ETIS-Larib\nexternal",
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
            if family != "unet_resnet50":
                continue  # the SwinUNETR runs land later; the figure picks them up then
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
                        "dataset": dataset,
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


def _panel(ax, table: pd.DataFrame, order: tuple[str, ...], title: str, ylabel: str) -> None:
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
            height = max(base_top, tops[method], previous + pad)
            previous = height
            left = index - width
            right = index + (METHODS.index(method) - 1) * width
            ax.plot([left, right], [height, height], color="#303030", linewidth=0.6, zorder=4)
            ax.annotate(
                text,
                ((left + right) / 2.0, height),
                textcoords="offset points",
                xytext=(0, 1.5),
                ha="center",
                va="bottom",
                fontsize=5.4,
                color="#303030",
            )
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([SHORT_LABELS.get(name, name) for name in datasets])
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

    fig, (internal, external) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    _panel(
        internal,
        table,
        INTERNAL_ORDER,
        "(a) internal: empty prediction",
        "empty predictions [%]",
    )
    _panel(
        external,
        table,
        EXTERNAL_ORDER,
        "(b) external: missed by >= 8 of 15 checkpoints",
        "majority-miss rate [%]",
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


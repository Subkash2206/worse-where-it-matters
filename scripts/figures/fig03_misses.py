"""Figure 3 - how often TTA loses the polyp entirely.

Miss rate per dataset for baseline, TTA-4 and TTA-8 with 95 % intervals, and the Holm-adjusted
McNemar p of the paired transition above each pair of bars.  Definition of a miss, stated in the
figure because the two families differ:

* internal CVC-ClinicDB and BraTS: an empty prediction on a case whose reference is non-empty, read
  from the seed-averaged detection table (case level, one unit per case);
* external polyp collections: the share of cases that a **majority of the 15 checkpoints** miss,
  with a Jeffreys interval - the definition the external validation reports, and the conservative
  companion of the per-checkpoint McNemar count.
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


def main() -> int:
    style.apply()
    table = collect()
    if table.empty:
        print("no detection tables found", file=sys.stderr)
        return 2
    import matplotlib.pyplot as plt

    order = ("cvc_unet", "cvc_swin", "brats", "kvasir_seg", "cvc_colondb", "etis_larib")
    datasets = [name for name in order if name in set(table["dataset"])]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    width = 0.26
    for index, dataset in enumerate(datasets):
        block = table.loc[table["dataset"] == dataset]
        colour = style.PALETTE.get(dataset, "#555555")
        for offset, method in enumerate(METHODS):
            row = block.loc[block["method"] == method]
            if row.empty:
                continue
            row = row.iloc[0]
            rate = float(row["rate"]) * 100.0
            ax.bar(
                index + (offset - 1) * width,
                rate,
                width,
                color=colour,
                alpha=1.0 - 0.28 * offset,
                edgecolor="white",
                linewidth=0.4,
                label=method if index == 0 else None,
            )
            ax.errorbar(
                index + (offset - 1) * width,
                rate,
                yerr=[
                    [max(rate - float(row["low"]) * 100.0, 0.0)],
                    [max(float(row["high"]) * 100.0 - rate, 0.0)],
                ],
                color="#303030",
                elinewidth=0.6,
                capsize=1.5,
            )
            p_value = float(row["p_value"]) if np.isfinite(row["p_value"]) else float("nan")
            if method != "baseline" and np.isfinite(p_value):
                text = "n.s." if p_value > 0.05 else ("p<0.001" if p_value < 0.001 else f"p={p_value:.3f}")
                ax.annotate(
                    text,
                    (index + (offset - 1) * width, float(row["high"]) * 100.0),
                    textcoords="offset points",
                    xytext=(0, 2),
                    ha="center",
                    fontsize=5.5,
                    color="#303030",
                    rotation=90,
                )
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(
        [style.LABELS[name].replace(" (external)", "") for name in datasets], rotation=15, ha="right"
    )
    ax.set_ylabel("miss rate [%]")
    ax.set_title("TTA recovers polyps and silences others - net worse on ETIS and ColonDB")
    ax.legend(title="method", loc="upper left", ncol=3)
    style.annotate(
        ax, "internal/BraTS: empty prediction | external: majority of 15 checkpoints", corner="upper right"
    )
    style.save(fig, "fig03_misses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


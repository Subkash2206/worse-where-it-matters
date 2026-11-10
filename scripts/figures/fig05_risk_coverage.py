"""Figure 5 - can the uncertainty tell which cases failed?  Risk-coverage curves, CVC UNet.

Cases are ranked by the uncertainty signal (most uncertain first), the most uncertain fraction
(coverage removed) is rejected and the failure rate left in the retained set is plotted.  A signal
that knows nothing is flat at the base failure rate; a useful one falls steeply.  AUROC of the same
signal against the failure label is computed here from the per-case values, so the legend and the
curves always agree (``results/aggregate_*/uncertainty_audit_auroc.csv`` is the independent audit).

Signals: single-view ROI entropy (baseline), TTA-4 ROI entropy, TTA-4 view disagreement, TTA-4
pairwise view Dice (1 - mean pairwise Dice) and TTA-4 ROI variance.  ``1 - confidence`` is not among
the stored per-case metrics - the run records entropy, variance, view disagreement and per-view Dice
but not the probability at the predicted pixels - so the variance curve stands in for it; the audit
table lists the same replacement (``tta4_variance_roi``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import style  # noqa: E402

RUNS = style.RESULTS / "cvc_clinicdb_unet_resnet50" / "evaluation"
SIGNALS = (
    ("baseline", "entropy_mean_roi", "single-view ROI entropy", "#1f4e79", "-"),
    ("tta4", "entropy_mean_roi", "TTA-4 ROI entropy", "#c0392b", "-"),
    ("tta4", "view_disagreement", "TTA-4 view disagreement", "#e67e22", "-"),
    ("tta4", "per_view_dice_mean", "1 - pairwise view Dice", "#16a085", "--"),
    ("tta4", "variance_mean_roi", "TTA-4 ROI variance (= 1-confidence proxy)", "#7d3c98", "--"),
)
FAILURE_HD95_PX = 50.0


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with the score oriented so that larger means more uncertain."""
    from scipy.stats import rankdata

    positive = labels.astype(bool)
    n_pos = int(positive.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load() -> pd.DataFrame:
    paths = sorted(RUNS.glob("*_per_case.csv"))
    if not paths:
        raise SystemExit(f"no per-case CSVs in {RUNS}")
    wanted = {"hd95", "dice"} | {metric for _, metric, *_ in SIGNALS}
    frames = [
        pd.read_csv(path, usecols=["case_id", "method", "metric", "value"]) for path in paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame["metric"].isin(wanted)]
    averaged = frame.groupby(["case_id", "method", "metric"])["value"].mean().reset_index()
    wide = averaged.pivot_table(index="case_id", columns=["method", "metric"], values="value")
    wide.columns = [f"{method}:{metric}" for method, metric in wide.columns]
    return wide.dropna(subset=["tta4:hd95"])


def main() -> int:
    style.apply()
    import matplotlib.pyplot as plt

    table = load()
    failed = (table["tta4:hd95"] > FAILURE_HD95_PX).to_numpy()
    base = float(failed.mean())
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    ax.axhline(base * 100.0, color="#909090", linestyle=":", linewidth=0.8)
    ax.annotate(f"no rejection {base * 100.0:.1f} %", (0.02, base * 100.0), textcoords="offset points",
                xytext=(0, 3), fontsize=6, color="#606060")
    for method, metric, label, colour, linestyle in SIGNALS:
        column = f"{method}:{metric}"
        if column not in table:
            print(f"missing {column}; skipped", file=sys.stderr)
            continue
        scores = table[column].to_numpy(dtype=float)
        if metric == "per_view_dice_mean":
            scores = 1.0 - scores  # larger = more uncertain
        order = np.argsort(-scores)
        ordered_failures = failed[order]
        coverage = np.arange(1, failed.size + 1) / failed.size
        remaining = np.cumsum(ordered_failures) / np.arange(1, failed.size + 1)
        value = auroc(scores, failed.astype(int))
        ax.plot(coverage * 100.0, remaining * 100.0, color=colour, linestyle=linestyle,
                label=f"{label} (AUROC {value:.2f})")
    ax.set_xlabel("coverage retained [%]")
    ax.set_ylabel("failure rate in retained set [%]")
    ax.set_title("Uncertainty ranks the HD95 > 50 px failures")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper right")
    style.annotate(ax, "failure = TTA-4 HD95 > 50 px", corner="lower left")
    style.save(fig, "fig05_risk_coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

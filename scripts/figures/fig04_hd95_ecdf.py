"""Figure 4 - HD95 distribution of the CVC-ClinicDB UNet, baseline versus TTA-4.

ECDF over the 15 test cases-folds (5 folds x 3 seeds) x 612 cases, one line per method, with the
50 px clinical threshold and the p95 of each method marked.  Distances are in pixels: the
collection has no calibrated spacing, so the threshold is the round 50 px used in the protocol
section, not a millimetre value.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import style  # noqa: E402

RUNS = style.RESULTS / "cvc_clinicdb_unet_resnet50" / "evaluation"
THRESHOLD_PX = 50.0
METHODS = ("baseline", "tta4")


def load() -> pd.DataFrame:
    paths = sorted(RUNS.glob("*_per_case.csv"))
    if not paths:
        raise SystemExit(f"no per-case CSVs in {RUNS}")
    frames = [pd.read_csv(path, usecols=["case_id", "method", "metric", "value"]) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[(frame["metric"] == "hd95") & (frame["method"].isin(METHODS))]
    return frame.groupby(["case_id", "method"])["value"].mean().reset_index()


def main() -> int:
    style.apply()
    import matplotlib.pyplot as plt

    frame = load()
    wide = frame.pivot(index="case_id", columns="method", values="value")
    paired = wide.dropna()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for method, colour, linestyle in (
        ("baseline", style.PALETTE["cvc_unet"], "-"),
        ("tta4", "#e67e22", "-"),
    ):
        values = np.sort(paired[method].to_numpy(dtype=float))
        cumulative = np.arange(1, values.size + 1) / values.size
        ax.step(values, cumulative, where="post", color=colour, linestyle=linestyle, label=f"{method} (n={values.size})")
        p95 = float(np.percentile(values, 95))
        ax.plot([p95, p95], [0.0, 0.95], color=colour, linestyle=":", linewidth=0.8)
        ax.annotate(
            f"p95 {p95:.0f} px",
            (p95, 0.95),
            textcoords="offset points",
            xytext=(3, -9),
            fontsize=6,
            color=colour,
        )
    ax.axvline(THRESHOLD_PX, color="#909090", linestyle="--", linewidth=0.8)
    ax.annotate("50 px", (THRESHOLD_PX, 0.05), textcoords="offset points", xytext=(3, 0), fontsize=6, color="#606060")
    share = {
        method: float((paired[method] > THRESHOLD_PX).mean()) * 100.0 for method in paired.columns
    }
    ax.annotate(
        f"> 50 px: baseline {share['baseline']:.1f} %, TTA-4 {share['tta4']:.1f} %",
        (0.98, 0.30),
        xycoords="axes fraction",
        ha="right",
        fontsize=6.5,
        color="#303030",
    )
    ax.set_xlabel("HD95 [px]")
    ax.set_ylabel("ECDF")
    ax.set_xlim(0, float(np.percentile(np.concatenate([paired[m] for m in METHODS]), 99.5)))
    ax.set_ylim(0, 1.0)
    ax.set_title("CVC-ClinicDB UNet: TTA-4 pulls the tail in")
    ax.legend(loc="lower right")
    style.save(fig, "fig04_hd95_ecdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

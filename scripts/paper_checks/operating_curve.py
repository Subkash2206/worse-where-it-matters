"""Is TTA a threshold shift for image-level detection? (UNet, external data, no new inference)

The low-threshold rule replaces an empty 0.5 mask by the map thresholded at t whenever the
map peaks at >= t. After the rule, a prediction is empty exactly when its maximum probability
is below t. The rule outputs at t = 0.40, 0.30, 0.20 together with the plain 0.5 mask
therefore give four points of the image-level detection curve, for the single view
(baseline, baseline_lowdet_T) and for the TTA-8 mean map (tta8, tta8_lowdet_T):

  x  false-alarm rate on the 1009 polyp-free cecum frames (mean over 15 checkpoints);
  y  share of empty predictions on CVC-ColonDB + ETIS-Larib pooled
     (8640 image-checkpoint pairs, i.e. also a mean over the 15 checkpoints).

The TTA-8 curve is interpolated linearly at the single view's false-alarm rates.
Reads results/paper_checks/single_view_rule.csv (scripts/paper_checks/single_view_rule.py).
Run from the repository root:  python scripts/paper_checks/operating_curve.py
Writes: results/paper_checks/operating_curve.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_checks"
PAIRS = 380 * 15 + 196 * 15
IMAGES = 380 + 196
ORDER = {"single": ["baseline", "baseline_lowdet_40", "baseline_lowdet_30", "baseline_lowdet_20"],
         "tta": ["tta8", "tta8_lowdet_40", "tta8_lowdet_30", "tta8_lowdet_20"]}
THRESHOLDS = [0.50, 0.40, 0.30, 0.20]


def main() -> None:
    table = pd.read_csv(OUT / "single_view_rule.csv")
    rows = []
    for source, methods in ORDER.items():
        for t, m in zip(THRESHOLDS, methods):
            r = table[(table.source == source) & (table.method == m)].iloc[0]
            rows.append(dict(
                curve="single view" if source == "single" else "TTA-8", threshold=t, method=m,
                false_alarm_pct=float(r.false_alarm_pct),
                empty_pct_pairs=100.0 * (r.cvc_colondb_pairs_empty + r.etis_larib_pairs_empty) / PAIRS,
                empty_pct_majority=100.0 * (r.cvc_colondb_majority_empty + r.etis_larib_majority_empty) / IMAGES,
            ))
    curve = pd.DataFrame(rows)
    tta = curve[curve.curve == "TTA-8"].sort_values("false_alarm_pct")
    for col in ("empty_pct_pairs", "empty_pct_majority"):
        interp = np.interp(curve.false_alarm_pct, tta.false_alarm_pct, tta[col], left=np.nan, right=np.nan)
        curve[f"tta8_{col}_at_same_false_alarm"] = np.where(curve.curve == "single view", interp, np.nan)
    curve = curve.round(2)
    curve.to_csv(OUT / "operating_curve.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()

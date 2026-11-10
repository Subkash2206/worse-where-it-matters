"""Figure 5 - do the uncertainty signals flag the cases that failed?

Primary failure definition (the one the paper uses): **Dice < 0.5 on a non-empty prediction**, i.e.
``non_missed_failures``, and every score is judged against the failures of its own prediction
(``label_source = own``): the single-view entropy of the baseline against baseline failures, the
TTA-4 scores against TTA-4 failures.  Both - the scores and the labels - come from
``results/aggregate_cvc_clinicdb_unet_resnet50/uncertainty_audit_per_case.csv`` and the legend quotes
the AUROC/AURC that ``uncertainty_audit_auroc.csv`` computed from exactly those rows (``label_source
= own``, same ``failure_definition``).  ``AURC n/a`` is printed whenever the audit holds no finite
AURC for that signal, instead of inventing one.

The curves are risk-coverage curves in the selective-prediction sense: cases are kept from the *most
confident* one (lowest uncertainty score) to the least confident, so the failure rate of the retained
set starts at the most confident case and rises to the cohort's own failure rate as coverage reaches
100 %.  Sorting the other way round (most uncertain first) gives the inverted curve that starts near
100 % and falls - that is the rejection curve of the rejected set, not a risk-coverage curve, and it
hides the fact that the good signals rank failures last.  An assertion keeps the direction honest:
the first point must sit below the base rate and the last point must equal it.

The supplementary panel repeats the analysis with the HD95 > 50 px definition, so a reviewer can see
that the ranking does not depend on which failure label is used.

Signals: single-view ROI entropy (baseline), TTA-4 ROI entropy, 1 - confidence (baseline), TTA-4 view
disagreement and 1 - pairwise view Dice of the TTA-4 views.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import style  # noqa: E402

AGGREGATE = style.RESULTS / "aggregate_cvc_clinicdb_unet_resnet50"
PER_CASE = AGGREGATE / "uncertainty_audit_per_case.csv"
AUROC = AGGREGATE / "uncertainty_audit_auroc.csv"
FAILURE_DICE = 0.5
FAILURE_HD95_PX = 50.0

#: per-case score column, method whose failures label it, legend label, colour, linestyle, and the
#: predictor name the uncertainty audit stores the same score under
SIGNALS = (
    (
        "baseline_entropy_roi",
        "baseline",
        "single-view ROI entropy",
        style.OKABE_ITO["blue"],
        "-",
        "baseline_entropy_roi",
    ),
    (
        "tta4_entropy_roi",
        "tta4",
        "TTA-4 ROI entropy",
        style.OKABE_ITO["vermillion"],
        "-",
        "tta4_entropy_roi",
    ),
    (
        "one_minus_confidence",
        "baseline",
        "1 - confidence",
        style.OKABE_ITO["purple"],
        "-",
        "one_minus_confidence",
    ),
    (
        "tta4_view_disagreement",
        "tta4",
        "TTA-4 view disagreement",
        style.OKABE_ITO["orange"],
        "-",
        "tta4_view_disagreement",
    ),
    (
        "tta4_view_pairwise_dice",
        "tta4",
        "1 - pairwise view Dice",
        style.OKABE_ITO["green"],
        "--",
        "tta4_view_dice_disagreement",
    ),
)

#: the audit calls the plotted "1 - pairwise view Dice" score ``tta4_view_dice_disagreement``
METHOD_LABELS = {"baseline": "baseline failures", "tta4": "TTA-4 failures"}
DEFINITIONS = {"dice": "non_missed_failures", "hd95": "all_failures"}



def _curve(scores: np.ndarray, failed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retained coverage against the failure rate of the retained set, most confident first.

    ``scores`` is an uncertainty score (higher = less confident), so an ascending sort keeps the
    most confident cases first and the failure rate of that set climbs from the first case to the
    cohort's base rate at full coverage - the risk-coverage curve of selective prediction.
    """
    order = np.argsort(scores, kind="stable")
    ordered = failed[order].astype(float)
    retained = np.arange(1, ordered.size + 1) / ordered.size * 100.0
    return retained, np.cumsum(ordered) / np.arange(1, ordered.size + 1) * 100.0


def _audit_values(audit: pd.DataFrame, predictor: str, definition: str) -> tuple[float, float]:
    """AUROC and AURC of one signal, from the audit rows of this very definition (own labels)."""
    if audit.empty or predictor not in set(audit["predictor"]):
        return float("nan"), float("nan")
    rows = audit.loc[
        (audit["predictor"] == predictor)
        & (audit["label_source"] == "own")
        & (audit["failure_definition"] == definition)
    ]
    if rows.empty:
        return float("nan"), float("nan")
    return float(rows["auroc"].iloc[0]), float(rows["aurc"].iloc[0])


def _format(auroc_value: float, aurc_value: float) -> str:
    """``AUROC x.xx, AURC y.yyy``; ``n/a`` where the audit has no finite value."""
    auroc_text = f"{auroc_value:.2f}" if np.isfinite(auroc_value) else "n/a"
    aurc_text = f"{aurc_value:.3f}" if np.isfinite(aurc_value) else "n/a"
    return f"AUROC {auroc_text}, AURC {aurc_text}"



def _labels(table: pd.DataFrame, method: str, definition: str) -> np.ndarray:
    """Failure label of one method: Dice < 0.5 on a non-empty prediction, or HD95 > 50 px."""
    if definition == "dice":
        failed = (table[f"dice_{method}"] < FAILURE_DICE).to_numpy()
        column = f"missed_{method}"
        if column in table:
            failed = failed & ~table[column].to_numpy(dtype=bool)  # empty predictions are misses, not failures
        return failed
    return (table[f"hd95_{method}"] > FAILURE_HD95_PX).to_numpy()


def _panel(
    ax, table: pd.DataFrame, definition: str, title: str, audit: pd.DataFrame, tag: str
) -> None:
    """One risk-coverage panel; every curve is asserted to run the right way round."""
    base_rates = {
        method: float(_labels(table, method, definition).mean()) * 100.0
        for method in ("baseline", "tta4")
    }
    for position, (method, rate_value) in enumerate(
        sorted(base_rates.items(), key=lambda item: -item[1])
    ):
        ax.axhline(rate_value, color="#909090", linestyle=":", linewidth=0.8, zorder=1)
        ax.annotate(
            f"{METHOD_LABELS[method]} {rate_value:.1f} %",
            (3.0, rate_value),
            textcoords="offset points",
            xytext=(0, 3 if position == 0 else -9),
            va="bottom" if position == 0 else "top",
            fontsize=5.0,
            color="#606060",
        )
    for column, method, label, colour, linestyle, predictor in SIGNALS:
        if column not in table:
            continue
        scores = table[column].to_numpy(dtype=float)
        if column == "tta4_view_pairwise_dice":
            scores = 1.0 - scores  # 1 - pairwise Dice: larger means more uncertain
        failed = _labels(table, method, definition)
        retained, rate = _curve(scores, failed)
        base_rate = float(failed.mean()) * 100.0
        if not float(rate[0]) < base_rate:
            raise AssertionError(
                f"{tag}/{column}: the first point of the risk-coverage curve is "
                f"{float(rate[0]):.2f} %, not below the base rate {base_rate:.2f} % - the curve "
                "is sorted the wrong way round"
            )
        if abs(float(rate[-1]) - base_rate) > 1e-6:
            raise AssertionError(
                f"{tag}/{column}: full coverage ends at {float(rate[-1]):.6f} %, "
                f"not at the base rate {base_rate:.6f} %"
            )
        auroc_value, aurc_value = _audit_values(audit, predictor, DEFINITIONS[definition])
        ax.plot(
            retained,
            rate,
            color=colour,
            linestyle=linestyle,
            label=f"{label}  {_format(auroc_value, aurc_value)}",
        )
        print(
            f"{tag:4s} {column:28s} n={failed.size:3d} base={base_rate:5.2f} % "
            f"first={float(rate[0]):5.2f} % last={float(rate[-1]):5.2f} % "
            f"{_format(auroc_value, aurc_value)}"
        )
    ax.set_xlim(0, 100)
    # headroom above the highest curve/base rate so the legend never covers a line
    highest = max(
        [float(np.max(np.asarray(line.get_ydata(), dtype=float))) for line in ax.lines if len(line.get_ydata())]
    )
    ax.set_ylim(0, highest * 1.55)
    ax.set_title(title, loc="left")
    ax.set_xlabel("coverage retained [%]")
    ax.set_ylabel("failure rate [%]")


def main() -> int:
    style.apply()
    if not PER_CASE.is_file():
        print(f"missing {PER_CASE}; run scripts/aggregate.py for the CVC UNet first", file=sys.stderr)
        return 2
    table = pd.read_csv(PER_CASE)
    audit = pd.read_csv(AUROC) if AUROC.is_file() else pd.DataFrame()
    import matplotlib.pyplot as plt

    fig, (primary, supplementary) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    _panel(primary, table, "dice", "(a) primary: Dice < 0.5, own prediction", audit, "dice")
    _panel(supplementary, table, "hd95", "(b) supplementary: HD95 > 50 px", audit, "hd95")
    for ax in (primary, supplementary):
        ax.legend(loc="upper left", fontsize=5.2, borderaxespad=0.4, handlelength=1.6)
    style.save(fig, "fig05_risk_coverage")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

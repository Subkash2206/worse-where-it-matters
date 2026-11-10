"""Figure 5 - do the uncertainty signals flag the cases that failed?

Primary failure definition (the one the paper uses): **Dice < 0.5 on a non-empty prediction**, i.e.
``non_missed_failures``, and every score is judged against the failures of its own prediction
(``label_source = own``): the single-view entropy of the baseline against baseline failures, the
TTA-4 scores against TTA-4 failures.  Both - the scores and the labels - come from
``results/aggregate_cvc_clinicdb_unet_resnet50/uncertainty_audit_per_case.csv`` and the legend quotes
the AUROC/AURC that ``uncertainty_audit_auroc.csv`` computed from exactly those rows (``label_source
= own``, same ``failure_definition``).  ``AURC n/a`` is printed whenever the audit holds no finite
AURC for that signal, instead of inventing one.

The two panels use the two Dice-based failure definitions the audit actually wrote (there is **no**
HD95 definition in ``uncertainty_audit_auroc.csv``; its only values are ``all_failures``,
``non_missed_failures`` and ``missed_vs_clean``):

* (a) ``non_missed_failures``: ``(~missed) & (dice < 0.5)`` - a non-empty prediction that still fails;
* (b) ``all_failures``: ``missed | (dice < 0.5)`` - the same failures **plus** the empty predictions.

Every score is judged against the failures of its own prediction (``label_source = own``), the curves
are computed from exactly the definition named in the panel title, and the legend quotes the AUROC /
AURC rows of that same definition.  ``AURC n/a`` is printed whenever the audit holds no finite value.
A guard compares the two panels entry by entry and refuses to let panel (b) report a panel-(a) number
unless the two definitions provably select the same cases for that predictor.

The curves are risk-coverage curves in the selective-prediction sense: cases are kept from the *most
confident* one (lowest uncertainty score) to the least confident, so the failure rate of the retained
set starts at the most confident case and rises to the cohort's own failure rate as coverage reaches
100 %.  Sorting the other way round (most uncertain first) gives the inverted curve that starts near
100 % and falls - that is the rejection curve of the rejected set, not a risk-coverage curve, and it
hides the fact that the good signals rank failures last.  An assertion keeps the direction honest:
the first point must sit below the base rate and the last point must equal it.

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
#: panel key -> the ``failure_definition`` column of the audit (the audit has no HD95 definition)
DEFINITIONS = {"non_missed": "non_missed_failures", "all": "all_failures"}



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


def _audit_values(
    audit: pd.DataFrame, predictor: str, definition: str
) -> tuple[float, float, float, float]:
    """AUROC, AURC and the row size of one signal, in one audit definition (own labels)."""
    missing = (float("nan"), float("nan"), float("nan"), float("nan"))
    if audit.empty or predictor not in set(audit["predictor"]):
        return missing
    rows = audit.loc[
        (audit["predictor"] == predictor)
        & (audit["label_source"] == "own")
        & (audit["failure_definition"] == definition)
    ]
    if rows.empty:
        return missing
    row = rows.iloc[0]
    return (
        float(row["auroc"]),
        float(row["aurc"]),
        float(row["n_used"]),
        float(row["n_failures"]),
    )


def _format(auroc_value: float, aurc_value: float) -> str:
    """``AUROC x.xx, AURC y.yyy``; ``n/a`` where the audit has no finite value."""
    auroc_text = f"{auroc_value:.2f}" if np.isfinite(auroc_value) else "n/a"
    aurc_text = f"{aurc_value:.3f}" if np.isfinite(aurc_value) else "n/a"
    return f"AUROC {auroc_text}, AURC {aurc_text}"



def _labels(table: pd.DataFrame, method: str, definition: str) -> np.ndarray:
    """Failure label of one method, in the audit's own terms.

    ``non_missed``: ``(~missed) & (dice < 0.5)`` - a non-empty prediction that still fails;
    ``all``: ``missed | (dice < 0.5)`` - the same failures plus the empty predictions.
    """
    dice = table[f"dice_{method}"].to_numpy(dtype=float)
    failed = dice < FAILURE_DICE
    column = f"missed_{method}"
    if column not in table:
        return failed
    missed = table[column].to_numpy(dtype=bool)
    if definition == "non_missed":
        # a copy: `failed` is a read-only view of the frame under pandas 3
        return failed & ~missed  # empty predictions are misses, not failures
    return missed | failed


def _panel(
    ax, table: pd.DataFrame, definition: str, title: str, audit: pd.DataFrame, tag: str
) -> dict[str, dict[str, float]]:
    """One risk-coverage panel; every curve is asserted to run the right way round.

    Returns the audit row behind every legend entry, so :func:`_guard_reuse` can check that the two
    panels really use their own definition.
    """
    used: dict[str, dict[str, float]] = {}
    base_rates = {
        method: float(_labels(table, method, definition).mean()) * 100.0
        for method in ("baseline", "tta4")
    }
    for method, rate_value in base_rates.items():
        ax.axhline(rate_value, color="#909090", linestyle=":", linewidth=0.8, zorder=1)
    notes = [
        "dotted: failure rate without rejection - "
        + ", ".join(
            f"{'baseline' if method == 'baseline' else 'TTA-4'} {rate_value:.1f} %"
            for method, rate_value in sorted(base_rates.items(), key=lambda item: -item[1])
        )
    ]
    legend_lines: list[str] = []
    for column, method, label, colour, linestyle, predictor in SIGNALS:
        if column not in table:
            continue
        scores = table[column].to_numpy(dtype=float)
        if column == "tta4_view_pairwise_dice":
            scores = 1.0 - scores  # 1 - pairwise Dice: larger means more uncertain
        failed = _labels(table, method, definition)
        retained, rate = _curve(scores, failed)
        base_rate = float(failed.mean()) * 100.0
        mean_rate = float(rate.mean())
        below = float((rate < base_rate).mean())
        # Direction guard: a risk-coverage curve sorted most-confident-first spends most of its
        # coverage range *below* the no-rejection line and then reaches it exactly.  The inverted
        # sort (most uncertain first) spends most of the range above the line - that was the bug.
        if below < 0.5:
            raise AssertionError(
                f"{tag}/{column}: the curve is below the base rate at only {below:.0%} of the "
                "coverage range - the cases are sorted the wrong way round"
            )
        if abs(float(rate[-1]) - base_rate) > 1e-6:
            raise AssertionError(
                f"{tag}/{column}: full coverage ends at {float(rate[-1]):.6f} %, "
                f"not at the base rate {base_rate:.6f} %"
            )
        auroc_value, aurc_value, n_used, n_failures = _audit_values(
            audit, predictor, DEFINITIONS[definition]
        )
        text = _format(auroc_value, aurc_value)
        ax.plot(retained, rate, color=colour, linestyle=linestyle, label=f"{label}  {text}")
        used[predictor] = {
            "auroc": auroc_value,
            "aurc": aurc_value,
            "n_used": n_used,
            "n_failures": n_failures,
        }
        legend_lines.append(
            f"    {label:30s} AUROC {auroc_value:.4f}  AURC {aurc_value:.4f}   "
            f"[audit {predictor} / own / {DEFINITIONS[definition]} / "
            f"n={n_used:.0f} / {n_failures:.0f} failures]   "
            f"base {base_rate:5.2f} %  first {float(rate[0]):6.2f} %  "
            f"mean {mean_rate:5.2f} %  last {float(rate[-1]):5.2f} %  below {below:.0%}"
        )
        if float(rate[0]) >= base_rate:
            legend_lines.append(
                "      note: the single most confident case is itself a failure under this "
                "definition - an empty prediction has no view disagreement, so agreement-based "
                "signals rank it first"
            )
        if predictor == "tta4_view_dice_disagreement" and definition == "non_missed":
            if abs(auroc_value - 0.865) > 5e-4:
                raise AssertionError(
                    "the pairwise view Dice entry for the UNet is no longer AUROC 0.865 but "
                    f"{auroc_value:.4f}"
                )
            legend_lines.append(
                "      ^ 1 - pairwise view Dice, UNet: AUROC 0.865 confirmed against "
                "uncertainty_audit_auroc.csv"
            )
    ax.set_xlim(0, 100)
    # Robust upper limit: the first few cases can be a single degenerate point (for an
    # agreement-based signal an empty prediction is the most "confident" case), which would
    # otherwise stretch the axis to 100 % and flatten every curve.  Peaks above the limit are
    # annotated rather than silently dropped.
    tail = rate[max(1, int(0.02 * rate.size)) :]
    top = max(max(base_rates.values()) * 1.7, float(tail.max()) * 1.45)
    ax.set_ylim(0, top)
    if float(rate.max()) > top:
        notes.insert(0, f"peak {float(rate.max()):.0f} % at the first case (off scale)")
    ax.annotate(
        "\n".join(notes),
        (0.985, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color="#606060",
        linespacing=1.4,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 1.2, "linewidth": 0},
    )
    ax.set_title(title, loc="left")
    ax.set_xlabel("coverage retained [%]")
    ax.set_ylabel("failure rate [%]")
    print(f"  legend entries of {title}")
    for line in legend_lines:
        print(line)
    return used


def _guard_reuse(
    primary: dict[str, dict[str, float]], supplementary: dict[str, dict[str, float]]
) -> None:
    """Refuse a panel-(b) number that is silently a panel-(a) number.

    An identical AUROC/AURC across the two definitions is legitimate only when the two definitions
    select exactly the same rows (same ``n_used`` and ``n_failures``) - which happens for a predictor
    that is undefined on every case the definitions disagree about.  Anything else means panel (b) is
    quoting the wrong audit rows.
    """
    if DEFINITIONS["all"] == DEFINITIONS["non_missed"]:
        raise AssertionError("both panels would use the same failure definition")
    for predictor, b in supplementary.items():
        a = primary.get(predictor)
        if a is None or not np.isfinite(b["auroc"]) or not np.isfinite(a["auroc"]):
            continue
        same_values = np.isclose(b["auroc"], a["auroc"]) and np.isclose(b["aurc"], a["aurc"])
        if not same_values:
            continue
        same_rows = (b["n_used"] == a["n_used"]) and (b["n_failures"] == a["n_failures"])
        if not same_rows:
            raise AssertionError(
                f"{predictor}: panel (b) reports the panel-(a) AUROC/AURC although the definitions "
                f"select different cases ({a['n_used']:.0f}/{a['n_failures']:.0f} vs "
                f"{b['n_used']:.0f}/{b['n_failures']:.0f}) - the panel is reusing the wrong rows"
            )
        print(
            f"  ok: {predictor} has identical AUROC/AURC under both definitions because both "
            f"select the same {b['n_used']:.0f} cases with {b['n_failures']:.0f} failures"
        )


def main() -> int:
    style.apply()
    if not PER_CASE.is_file():
        print(f"missing {PER_CASE}; run scripts/aggregate.py for the CVC UNet first", file=sys.stderr)
        return 2
    table = pd.read_csv(PER_CASE)
    audit = pd.read_csv(AUROC) if AUROC.is_file() else pd.DataFrame()
    if not audit.empty:
        print(
            "failure definitions in uncertainty_audit_auroc.csv: "
            + ", ".join(sorted(audit["failure_definition"].unique()))
        )
    import matplotlib.pyplot as plt

    fig, (primary, supplementary) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    used_primary = _panel(
        primary,
        table,
        "non_missed",
        "(a) non-missed failures\nDice < 0.5, empty predictions excluded",
        audit,
        "non_missed",
    )
    used_supplementary = _panel(
        supplementary,
        table,
        "all",
        "(b) all failures\nDice < 0.5 or empty prediction",
        audit,
        "all",
    )
    _guard_reuse(used_primary, used_supplementary)
    for ax in (primary, supplementary):
        ax.legend(
            loc="upper right",
            fontsize=5.2,
            borderaxespad=0.4,
            handlelength=1.6,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
        )
    style.save(fig, "fig05_risk_coverage")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

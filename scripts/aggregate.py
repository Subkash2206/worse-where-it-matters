"""Pool the whole cross-validation grid: seeds, folds, paired stats and seed ensembles.

Produces (under ``results/aggregate_<experiment>/`` by default):

* ``pooled_by_seed`` - per method and metric, the mean of the pooled 29-video results
  per seed, the standard deviation across seeds and each seed's value;
* ``paired_by_seed`` - video-level paired tests (Wilcoxon, bootstrap CI, rank-biserial,
  Holm) pooled over the 5 folds, computed inside every seed;
* ``paired_seed_averaged`` - the same tests after averaging metric values per case over
  seeds, so each video stays one independent test unit;
* ``detection_by_seed`` / ``detection_seed_averaged`` - missed-detection rates with
  cluster-bootstrap CIs, transition tables and exact McNemar tests, pooled over folds;
* ``ensemble_per_case`` / ``ensemble_comparison`` / ``ensemble_detection`` - the seed
  ensemble of the baseline maps (``ensemble3``) and of the tta4 maps
  (``ensemble3_tta4``), evaluated with the same metric code as the runs and compared
  with ``baseline``/``tta4`` on the same metrics, missed detections included.

Usage
-----
    python scripts/aggregate.py --config configs/cvc_clinicdb/unet_resnet50.yaml
    python scripts/aggregate.py --config ... --seeds 0 --output-dir results/_agg --no-ensemble
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.eval import (  # noqa: E402
    TailThresholds,
    aggregate_all,
    aggregate_output_dir,
    discover_runs,
    frames_by_seed,
    load_view_agreement,
    per_fold_table,
    pooled_method_table,
    run_uncertainty_audit,
    seed_averaged_frame,
    sensitivity_table,
    string_ids,
    tail_table,
    uncertainty_table,
    view_agreement_dir,
)
from melba.stats import pool_folds  # noqa: E402
from melba.utils import git_revision  # noqa: E402

import pandas as pd  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="experiment YAML in configs/")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="folds (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--per-case-dir", default=None, help="directory with the per-case CSVs")
    parser.add_argument("--maps-dir", default=None, help="root of the probability maps")
    parser.add_argument("--output-dir", default=None, help="default: results/aggregate_<experiment>")
    parser.add_argument("--unit", default=None, choices=["group", "case"], help="pairing unit")
    parser.add_argument("--treatments", nargs="+", default=["tta2", "tta4", "tta8"])
    parser.add_argument("--control", default="baseline")
    parser.add_argument(
        "--n-bootstrap", type=int, default=None, help="override stats.n_bootstrap"
    )
    parser.add_argument(
        "--no-ensemble", action="store_true", help="skip the seed ensembles (maps not needed)"
    )
    parser.add_argument(
        "--view-agreement-dir",
        default=None,
        help=(
            "per-case CSVs of the tta4 view-agreement re-run "
            "(default: results/<experiment>/view_agreement/evaluation)"
        ),
    )
    parser.add_argument(
        "--no-view-agreement",
        action="store_true",
        help="skip the pairwise-Dice detectors even if the view-agreement CSVs exist",
    )
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def _print_table(title: str, table, columns=None, max_rows: int = 40) -> None:
    if table is None or table.empty:
        print(f"\n{title}: (empty)")
        return
    printable = table if columns is None else table[[c for c in columns if c in table.columns]]
    print(f"\n{title}")
    print(
        printable.head(max_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def _uncertainty_tables(by_seed: dict, averaged) -> pd.DataFrame:
    """Spearman/AUROC of the tta4 uncertainty descriptors, per seed and seed-averaged."""
    parts: list[pd.DataFrame] = []
    for seed, frames in sorted(by_seed.items()):
        table = uncertainty_table(pool_folds(frames), method="tta4")
        if not table.empty:
            table.insert(0, "seed", int(seed))
            parts.append(table)
    averaged_table = uncertainty_table(averaged, method="tta4")
    if not averaged_table.empty:
        averaged_table.insert(0, "seed", -1)
        parts.append(averaged_table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _subset(frame: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame[[column for column in columns if column in frame.columns]]


def _write_summary_markdown(
    output_dir: Path,
    tables: dict,
    extra_tables: dict,
    *,
    args: argparse.Namespace,
    runs,
    folds_found,
    seeds_found,
    audit: dict | None = None,
) -> Path:
    """Write ``summary.md`` with the protocol sections (a)-(g)."""
    from melba.eval import write_markdown

    def pick(frame, *columns):
        return _subset(frame, list(columns))

    method_summary = extra_tables["method_summary"]
    frame_level = method_summary.loc[method_summary["level"] == "frame"]
    video_level = method_summary.loc[method_summary["level"] == "video"]
    uncertainty = extra_tables["uncertainty"]
    if not uncertainty.empty:
        uncertainty = uncertainty.loc[uncertainty["seed"] == -1]

    summary_columns = ("method", "metric", "n_cases", "n_videos", "n_seeds", "mean", "std")
    paired_columns = (
        "treatment", "metric", "n_units", "n_cases_used", "restriction", "delta_mean",
        "ci_low", "ci_high", "p_value", "p_value_adjusted", "effect_size", "preferred",
    )
    detection_columns = (
        "method", "n_cases", "n_missed", "missed_rate", "ci_low", "ci_high",
        "nonempty_to_empty", "empty_to_nonempty", "mcnemar_p", "mcnemar_p_adjusted",
    )
    tail_columns = (
        "method", "n_cases", "hd95_p90", "hd95_p95", "hd95_failure_rate",
        "dice_failure_rate", "delta_hd95_p95_estimate", "delta_hd95_p95_ci_low",
        "delta_hd95_p95_ci_high", "delta_hd95_failure_rate_estimate",
        "delta_hd95_failure_rate_ci_low", "delta_hd95_failure_rate_ci_high",
        "delta_dice_failure_rate_estimate", "delta_dice_failure_rate_ci_low",
        "delta_dice_failure_rate_ci_high",
    )
    sensitivity_columns = (
        "treatment", "metric", "n_units", "n_cases_used", "delta_mean", "ci_low", "ci_high",
        "p_value", "p_value_adjusted", "preferred",
    )
    uncertainty_columns = (
        "seed", "predictor", "n_cases", "n_failures", "spearman_vs_hd95",
        "spearman_vs_hd95_p", "spearman_vs_one_minus_dice",
        "spearman_vs_one_minus_dice_p", "auroc_failure",
    )

    preamble = (
        f"{len(runs)} runs: folds {folds_found} x seeds {seeds_found}; config "
        f"`{args.config}`; pairing unit `{args.unit or 'group'}`; bootstrap draws "
        f"`{args.n_bootstrap or 'config'}`; Holm correction; failure thresholds "
        "HD95 > 50 px and Dice < 0.5.\n\n"
        "Frames of one video are not independent, so every CI and test resamples or "
        "pairs at the video level (29 units). Distance metrics are restricted to cases "
        "non-empty under both methods (`n_cases_used`); section (e) repeats the "
        "comparison on all cases with empty predictions penalised by the image "
        "diagonal. `ensemble3` averages the three seeds' baseline maps and "
        "`ensemble3_tta4` their tta4 maps (3 x 4 views); both are a single realisation, "
        "so their across-seed std is undefined (blank)."
    )
    sections = [
        (
            "(a) Per-method summary - frame level",
            pick(frame_level, *summary_columns),
            "Mean over all frames of the fold, pooled over the 5 folds (612 cases), "
            "then mean ± std across the three seeds.",
        ),
        (
            "(a) Per-method summary - video level",
            pick(video_level, *summary_columns),
            "Each video contributes one value (mean over its frames), 29 videos in "
            "total; mean ± std across the three seeds.",
        ),
        (
            "(b) Paired comparisons vs baseline - video level, seeds averaged per case",
            pick(tables.get("paired_seed_averaged"), *paired_columns),
            "Wilcoxon signed-rank (primary), seeded percentile bootstrap CI of the mean "
            "delta, matched-pairs rank-biserial effect size, Holm across metrics. "
            "`restriction=both_nonempty` marks distance metrics and `n_cases_used` is "
            "the number of cases they were computed on.",
        ),
        (
            "(c) Missed detections pooled over folds",
            pick(tables.get("detection_seed_averaged"), *detection_columns),
            "Cluster (video) bootstrap CI of the missed rate; `nonempty_to_empty` counts "
            "new misses and `empty_to_nonempty` recovered cases relative to the "
            "baseline; exact McNemar p on the discordant pairs, Holm-corrected.",
        ),
        (
            "(d) Tail metrics (HD95 percentiles, failure rates) and change vs baseline",
            pick(extra_tables["tail_metrics"], *tail_columns),
            "Percentiles over all cases (an undefined distance counts as a failure); the "
            "change is a video-cluster paired bootstrap CI of the difference.",
        ),
        (
            "(e) Sensitivity: all cases, empty predictions penalised",
            pick(extra_tables["sensitivity_penalized"], *sensitivity_columns),
            "Uses the `*_penalized` columns (empty prediction -> image diagonal), so a "
            "method that misses lesions cannot look good by dropping them.",
        ),
        (
            "(f) Do the TTA uncertainty descriptors rank errors?",
            pick(uncertainty, *uncertainty_columns),
            "Spearman rho against HD95 and 1-Dice, and AUROC for detecting a failure "
            "(Dice < 0.5 or missed detection); row `seed=-1` is the seed-averaged set. "
            "`n_cases` is the cohort size, NOT the number of cases used: the ROI "
            "descriptors are undefined (NaN) for empty predictions and are silently "
            "dropped, which is what the (f-audit) tables below quantify.",
        ),
        (
            "(g) Baseline Dice per fold",
            extra_tables["per_fold_baseline"],
            "Fold-to-fold spread of the control method (seeds averaged).",
        ),
        (
            "Ensemble comparisons - video level, seeds averaged per case",
            pick(tables.get("ensemble_comparison"), *paired_columns),
            "`ensemble3` and `ensemble3_tta4` against `baseline` and `tta4`.",
        ),
    ]
    if audit is not None:
        sections.extend(
            [
                (
                    "(f-audit) ROI diagnostics for failures",
                    audit["roi_diagnostics"],
                    "The ROI descriptors are undefined when the predicted mask is empty "
                    "(`n_entropy_roi_nan`) and can be near zero when it is tiny, so the "
                    "NaNs would silently drop missed detections from any correlation/AUROC.",
                ),
                (
                    "(f-audit) AUROC and AURC for failure detection, three definitions",
                    _subset(
                        audit["auroc"],
                        [
                            "predictor",
                            "label_source",
                            "label_prediction",
                            "failure_definition",
                            "auroc",
                            "ci_low",
                            "ci_high",
                            "aurc",
                            "aurc_ci_low",
                            "aurc_ci_high",
                            "aurc_optimal",
                            "n_used",
                            "n_used_aurc",
                            "n_failures",
                            "n_undefined",
                            "auroc_if_undefined_certain",
                            "auroc_if_undefined_uncertain",
                            "aurc_if_missed_is_worst",
                        ],
                    ),
                    "`label_source=own` scores each predictor against the failures of "
                    "**its own** prediction (baseline descriptor vs baseline failures, tta4 "
                    "vs tta4, ensemble3 vs ensemble3, seed-averaged tta4 map vs "
                    "ensemble3_tta4); `label_source=tta4` repeats the audit against the "
                    "shared tta4 labels for comparison. `all_failures` = Dice < 0.5 or "
                    "missed; `non_missed_failures` excludes missed detections (every "
                    "predictor is defined there); `missed_vs_clean` contrasts missed "
                    "detections with clean cases only. `n_used`/`n_undefined` make the NaN "
                    "drop visible and the two AUROC bounds score undefined cases as "
                    "maximally certain/uncertain. **AURC** is the area under the "
                    "risk-coverage curve with risk = 1 - Dice (lower = better); "
                    "`aurc_optimal` is the same cohort sorted by its own risk (the oracle "
                    "lower bound) and `aurc_if_missed_is_worst` repeats the point estimate "
                    "with missed detections counted as risk 1 instead of being dropped. "
                    "CIs resample videos (29 clusters).",
                ),
                (
                    "(f-audit) Agreement detectors: pairwise Dice between views/seeds",
                    _subset(
                        audit["auroc"].loc[
                            audit["auroc"]["predictor"].isin(
                                [
                                    "tta4_view_dice_disagreement",
                                    "ensemble3_dice_disagreement",
                                    "ensemble3_tta4_dice_disagreement",
                                    "tta4_entropy_roi",
                                ]
                            )
                        ],
                        [
                            "predictor",
                            "label_source",
                            "label_prediction",
                            "failure_definition",
                            "auroc",
                            "ci_low",
                            "ci_high",
                            "aurc",
                            "aurc_ci_low",
                            "aurc_ci_high",
                            "aurc_optimal",
                            "n_used",
                            "n_used_aurc",
                            "n_failures",
                        ],
                    ),
                    "`tta4_view_dice_disagreement` = 1 - mean pairwise Dice between the "
                    "four tta4 views (per seed, averaged over seeds); "
                    "`ensemble3_dice_disagreement` and `ensemble3_tta4_dice_disagreement` "
                    "are the same over the three seeds' baseline / tta4 maps. Higher = the "
                    "members disagree = more uncertain. The tta4 entropy row is kept as the "
                    "reference descriptor.",
                ),
                (
                    "(f-audit) Spearman against HD95 and 1 - Dice (all predictors)",
                    _subset(
                        audit["spearman"],
                        [
                            "predictor",
                            "label_source",
                            "label_prediction",
                            "target",
                            "n_used",
                            "spearman_rho",
                            "p_value",
                        ],
                    ),
                    "Targets also come from the predictor's own prediction "
                    "(`label_source=own`); the `tta4` rows repeat the correlation against "
                    "the tta4 prediction for comparison. `n_used` counts the finite pairs, "
                    "so ROI descriptors lose the missed detections here.",
                ),
                (
                    "(f-audit) detector vs detector on AUROC and AURC (paired bootstrap)",
                    _subset(
                        audit["tests"],
                        [
                            "predictor",
                            "compared_with",
                            "label_prediction",
                            "delta_auroc",
                            "ci_low",
                            "ci_high",
                            "p_value",
                            "delta_aurc",
                            "aurc_ci_low",
                            "aurc_ci_high",
                            "aurc_p_value",
                            "n_used",
                        ],
                    ),
                    "Definition `non_missed_failures`, one shared video-cluster resampling for "
                    "both statistics. `delta_auroc` is positive when the first predictor is "
                    "the better classifier; `delta_aurc = AURC(first) - AURC(second)` is a "
                    "difference of selective risks, so it is **negative** when the first "
                    "predictor is the better detector. The `tta4_entropy_roi` rows compare "
                    "the reference descriptor with every alternative (under the tta4 labels "
                    "and under the alternative's own labels); the "
                    "`tta4_view_dice_disagreement` rows are the required comparisons of the "
                    "view-pairwise-Dice detector against the single-view ROI entropy and "
                    "against 1 - confidence.",
                ),
            ]
        )
        for note in audit.get("notes", []):
            sections.append(("Audit note", note, ""))
    return Path(
        write_markdown(
            str(Path(output_dir) / "summary.md"),
            title="MELBA CVC-ClinicDB - 5-fold x 3-seed TTA results",
            preamble=preamble,
            sections=sections,
        )
    )



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, parse_set_args(args.overrides))
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    stats = config.section("stats")
    unit = args.unit or str(stats.get("unit", "group"))
    n_bootstrap = int(args.n_bootstrap or stats.get("n_bootstrap", 10000))
    ci = float(stats.get("ci", 0.95))
    correction = str(stats.get("correction", "holm"))
    bootstrap_seed = int(stats.get("bootstrap_seed", 0))
    primary_test = str(stats.get("primary_test", "wilcoxon"))
    alternative = str(stats.get("alternative", "two-sided"))

    runs = discover_runs(
        config,
        folds=args.folds,
        seeds=args.seeds,
        output_dir=args.per_case_dir,
        map_root=args.maps_dir,
    )
    if not runs:
        print(
            "ERROR: no per-case results found for the requested folds/seeds; "
            "run scripts/run_grid.py first",
            file=sys.stderr,
        )
        return 2
    folds_found = sorted({run.fold for run in runs})
    seeds_found = sorted({run.seed for run in runs})
    print(f"{len(runs)} run(s) found: folds {folds_found}, seeds {seeds_found}")
    if args.output_dir:
        output_dir: Path | None = Path(args.output_dir)
    else:
        # one directory per experiment, so the UNet and SwinUNETR grids never overwrite
        # each other's tables
        output_dir = _REPO_ROOT / aggregate_output_dir(config)

    try:
        result = aggregate_all(
            config,
            runs=runs,
            unit=unit,
            treatments=args.treatments,
            control=args.control,
            primary_test=primary_test,
            alternative=alternative,
            n_bootstrap=n_bootstrap,
            ci=ci,
            bootstrap_seed=bootstrap_seed,
            correction=correction,
            ensemble=not args.no_ensemble,
            expected_seeds=tuple(int(seed) for seed in args.seeds),
            output_dir=output_dir,
        )
    except (ValueError, ConfigError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    tables = result.tables
    _print_table(
        f"pooled per-video results (mean +/- std across {len(seeds_found)} seed(s))",
        tables.get("pooled_by_seed"),
        columns=["method", "metric", "n_units", "n_cases", "n_seeds", "mean", "std"],
    )
    _print_table(
        "paired tests, seeds averaged per case first (unit = video)",
        tables.get("paired_seed_averaged"),
        columns=[
            "treatment",
            "control",
            "metric",
            "n_units",
            "n_cases_used",
            "restriction",
            "delta_mean",
            "ci_low",
            "ci_high",
            "p_value",
            "p_value_adjusted",
            "preferred",
        ],
    )
    _print_table(
        "missed detections, pooled over folds (seeds averaged per case)",
        tables.get("detection_seed_averaged"),
        columns=[
            "method",
            "n_cases",
            "n_missed",
            "missed_rate",
            "ci_low",
            "ci_high",
            "nonempty_to_empty",
            "empty_to_nonempty",
            "mcnemar_p",
            "mcnemar_p_adjusted",
        ],
    )
    _print_table(
        "seed ensemble vs baseline / tta4",
        tables.get("ensemble_comparison"),
        columns=[
            "treatment",
            "control",
            "metric",
            "n_units",
            "n_cases_used",
            "delta_mean",
            "ci_low",
            "ci_high",
            "p_value",
            "p_value_adjusted",
            "preferred",
        ],
    )
    _print_table(
        "seed ensemble missed detections",
        tables.get("ensemble_detection"),
        columns=[
            "method",
            "n_cases",
            "n_missed",
            "missed_rate",
            "ci_low",
            "ci_high",
            "nonempty_to_empty",
            "empty_to_nonempty",
            "mcnemar_p",
        ],
    )
    for note in result.notes:
        print(f"note: {note}")

    # ---- extra report tables: levels, tails, sensitivity, uncertainty ----------
    from melba.eval.aggregate import write_tables

    by_seed = frames_by_seed(runs)
    averaged = seed_averaged_frame(by_seed)
    ensembles = tables.get("ensemble_per_case")
    pieces = [string_ids(averaged.drop(columns=["n_seeds"], errors="ignore"))]
    if ensembles is not None and not ensembles.empty:
        pieces.append(string_ids(ensembles))
    combined = pd.concat(pieces, ignore_index=True)

    extra_tables = {
        "method_summary": pooled_method_table(by_seed, ensemble_frame=ensembles),
        "per_fold_baseline": per_fold_table(by_seed, method=args.control, metrics=("dice",)),
        "tail_metrics": tail_table(
            combined,
            control=args.control,
            thresholds=TailThresholds(),
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=bootstrap_seed,
        ),
        "sensitivity_penalized": sensitivity_table(
            averaged,
            control=args.control,
            treatments=args.treatments,
            unit=unit,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=bootstrap_seed,
            correction=correction,
        ),
        "uncertainty": _uncertainty_tables(by_seed, averaged),
    }
    tables.update(extra_tables)
    extra_paths = write_tables(extra_tables, output_dir)

    # ---- uncertainty audit: ROI semantics, three failure definitions, baselines ----
    view_agreement = None
    if not args.no_view_agreement:
        agreement_path = (
            Path(args.view_agreement_dir)
            if args.view_agreement_dir
            else _REPO_ROOT / view_agreement_dir(config)
        )
        if agreement_path.exists():
            view_agreement = load_view_agreement([agreement_path])
            print(
                f"view agreement: {len(view_agreement)} case(s) from {agreement_path}"
            )
        else:
            print(f"view agreement: no CSVs at {agreement_path} (detector stays NaN)")
    audit = run_uncertainty_audit(
        runs,
        averaged,
        by_seed=by_seed,
        ensemble_frame=tables.get("ensemble_per_case"),
        view_agreement=view_agreement,
        n_bootstrap=min(1000, n_bootstrap),
        ci=ci,
        seed=bootstrap_seed,
        band_px=int(config.get("evaluation.uncertainty_boundary_band_px", 3)),
    )
    audit_tables = {
        "uncertainty_audit_per_case": audit["per_case"],
        "uncertainty_audit_auroc": audit["auroc"],
        "uncertainty_audit_spearman": audit["spearman"],
        "uncertainty_audit_tests": audit["tests"],
        "uncertainty_audit_roi": audit["roi_diagnostics"],
    }
    extra_tables.update(audit_tables)
    extra_paths.update(write_tables(audit_tables, output_dir))
    for note in audit["notes"]:
        print(f"audit: {note}")
    _print_table(
        "uncertainty audit: AUROC / AURC for failure detection (all / non-missed / missed-only)",
        audit["auroc"].loc[
            audit["auroc"]["predictor"].isin(
                [
                    "tta4_entropy_roi",
                    "pred_area_px",
                    "one_minus_confidence",
                    "baseline_entropy_roi",
                    "ensemble3_entropy_roi",
                    "ensemble3_variance_roi",
                    "tta4_view_dice_disagreement",
                    "ensemble3_dice_disagreement",
                    "ensemble3_tta4_dice_disagreement",
                ]
            )
        ],
        columns=[
            "predictor",
            "label_source",
            "label_prediction",
            "failure_definition",
            "auroc",
            "ci_low",
            "ci_high",
            "aurc",
            "aurc_ci_low",
            "aurc_ci_high",
            "aurc_optimal",
            "n_used",
            "n_used_aurc",
            "n_failures",
            "n_undefined",
        ],
        max_rows=40,
    )
    _print_table(
        "uncertainty audit: detector vs detector (paired bootstrap, AUROC and AURC)",
        audit["tests"].loc[
            audit["tests"]["predictor"].isin(
                ["tta4_entropy_roi", "tta4_view_dice_disagreement"]
            )
        ],
        columns=[
            "predictor",
            "compared_with",
            "label_prediction",
            "delta_auroc",
            "ci_low",
            "ci_high",
            "p_value",
            "delta_aurc",
            "aurc_ci_low",
            "aurc_ci_high",
            "aurc_p_value",
            "n_used",
        ],
        max_rows=40,
    )
    _print_table(
        "uncertainty audit: ROI diagnostics",
        audit["roi_diagnostics"],
    )

    _print_table(
        "per-method summary, frame level (mean +/- std across seeds)",
        extra_tables["method_summary"].loc[lambda frame: frame["level"] == "frame"],
        columns=["method", "metric", "n_cases", "n_videos", "n_seeds", "mean", "std"],
    )
    _print_table(
        "tail metrics (HD95 percentiles and failure rates)",
        extra_tables["tail_metrics"],
        columns=[
            "method",
            "n_cases",
            "hd95_p90",
            "hd95_p95",
            "hd95_failure_rate",
            "dice_failure_rate",
            "delta_hd95_p95_estimate",
            "delta_hd95_p95_ci_low",
            "delta_hd95_p95_ci_high",
            "delta_dice_failure_rate_estimate",
        ],
    )
    _print_table(
        "uncertainty vs error (seed-averaged)",
        extra_tables["uncertainty"].loc[lambda frame: frame["seed"] == -1],
        columns=[
            "predictor",
            "n_cases",
            "n_failures",
            "spearman_vs_hd95",
            "spearman_vs_one_minus_dice",
            "auroc_failure",
        ],
    )
    _print_table("baseline Dice per fold (seeds averaged)", extra_tables["per_fold_baseline"])

    summary_path = _write_summary_markdown(
        output_dir,
        tables,
        extra_tables,
        args=args,
        runs=runs,
        folds_found=folds_found,
        seeds_found=seeds_found,
        audit=audit,
    )

    print(f"\ngit revision: {git_revision()}")
    print("written:")
    for name, path in sorted({**(result.paths or {}), **extra_paths}.items()):
        print(f"  {name:<28} {path}")
    print(f"  {'summary.md':<28} {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


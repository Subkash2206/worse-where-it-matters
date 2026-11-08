"""MC-dropout summary for a finished dropout grid (`mcdropout_T` vs geometric TTA).

Writes under ``results/mc_dropout_<experiment>/``:

* ``methods.csv`` — baseline / tta2 / tta4 / tta8 / ensemble3 / mcdropout_T side by side
  (video level, mean ± std);
* ``deltas.csv`` — paired tests of the MC method against ``baseline`` and ``tta4``
  (Wilcoxon, video bootstrap CI, Holm), i.e. "sampling the weights" against
  "transforming the input" on the same cases;
* ``detection.csv`` — missed detections per method (cluster bootstrap CI, McNemar);
* ``uncertainty_audit_auroc.csv`` — AUROC **and** AURC of every MC descriptor
  (entropy / variance of the samples, in its ROI and over the whole image, disagreement,
  per-sample agreement) against the MC prediction's own failures;
* ``uncertainty_audit_tests.csv`` — each MC descriptor vs the ``tta4`` ROI entropy on the
  same cases (shared video-cluster resampling, AUROC and AURC);
* ``uncertainty_audit_per_case.csv`` and ``summary.md``.

Usage
-----
    python scripts/summarize_mcdropout.py --config configs/cvc_clinicdb/unet_resnet50_dropout.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.eval import (  # noqa: E402
    discover_runs,
    ensemble_frame,
    frames_by_seed,
    load_ground_truth,
    mc_dropout_audit,
    mc_dropout_deltas,
    mc_dropout_method_table,
    seed_averaged_frame,
    write_markdown,
)
from melba.eval.aggregate import write_tables  # noqa: E402
from melba.stats import analyze_detection, pool_folds  # noqa: E402
from melba.utils import git_revision  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="dropout experiment YAML")
    parser.add_argument("--method", default="mcdropout_8", help="the MC method to audit")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, parse_set_args(args.overrides))
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    experiment = str(config.get("experiment.name", "experiment"))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _REPO_ROOT / "results" / f"mc_dropout_{experiment}"
    )

    runs = discover_runs(config, folds=args.folds, seeds=args.seeds)
    if not runs:
        print("ERROR: no per-case results; run the grid first", file=sys.stderr)
        return 2
    frames = frames_by_seed(runs)
    averaged = seed_averaged_frame(frames)
    ensembles: pd.DataFrame | None = None
    try:
        ground_truth = load_ground_truth(config, sorted({run.fold for run in runs}))
        ensembles = ensemble_frame(
            config,
            runs,
            name="ensemble3",
            source_method="baseline",
            ground_truth=ground_truth,
        )
    except (ConfigError, FileNotFoundError) as error:  # maps or masks unavailable
        print(f"note: ensembles skipped ({error})")

    combined = averaged
    if ensembles is not None and not ensembles.empty:
        combined = pd.concat([averaged, ensembles], ignore_index=True)
    pooled = pool_folds([combined])

    audit = mc_dropout_audit(
        averaged, method=args.method, n_bootstrap=args.n_bootstrap, seed=0
    )
    methods = mc_dropout_method_table(
        pooled,
        methods=(
            "baseline",
            "tta2",
            "tta4",
            "tta8",
            "ensemble3",
            "ensemble3_tta4",
            args.method,
        ),
    )
    deltas = mc_dropout_deltas(
        pooled, method=args.method, n_bootstrap=min(10000, args.n_bootstrap * 5)
    )
    detection_methods = [
        name
        for name in ("baseline", "tta4", "tta8", args.method)
        if name in set(combined["method"])
    ]
    detection = analyze_detection(
        combined,
        methods=detection_methods,
        control="baseline",
        unit="group",
    )

    tables = {
        "methods": methods,
        "deltas": deltas,
        "detection": detection,
        "uncertainty_audit_auroc": audit["auroc"],
        "uncertainty_audit_tests": audit["tests"],
        "uncertainty_audit_per_case": audit["per_case"],
    }
    paths = write_tables(
        tables,
        output_dir,
        manifest={
            "config": str(config.path) if config.path else args.config,
            "experiment": experiment,
            "method": args.method,
            "folds": sorted({run.fold for run in runs}),
            "seeds": sorted({run.seed for run in runs}),
            "n_bootstrap": args.n_bootstrap,
            "git_revision": git_revision(),
        },
    )

    print(f"\n({args.method} vs geometric TTA, video level)")
    print(methods.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n{args.method} vs each control (paired, video level)")
    print(deltas.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMC descriptors as failure detectors (own failures, AUROC / AURC)")
    print(audit["auroc"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMC descriptor vs tta4 ROI entropy (paired)")
    print(audit["tests"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for note in audit["notes"]:
        print(f"note: {note}")

    sections = [
        (
            "1. Methods side by side (video level)",
            methods,
            "Mean ± std over the 29 videos; the seed ensembles are a single realisation, "
            "so their across-seed std is undefined (blank).",
        ),
        (
            f"2. {args.method} vs baseline / tta4 (paired, video level)",
            deltas,
            "Wilcoxon signed-rank, video bootstrap CI of the mean delta, Holm correction; "
            "distance metrics on both-non-empty cases.",
        ),
        ("3. Missed detections", detection, "Cluster bootstrap CI and exact McNemar."),
        (
            "4. MC descriptors as failure detectors",
            audit["auroc"],
            "AUROC and AURC (risk = 1 - Dice) against the MC prediction's own failures with "
            "video-cluster bootstrap CIs; the ROI descriptors are undefined for a missed "
            "detection, so `n_used` drops them and the NaN bounds quantify that.",
        ),
        (
            "5. MC descriptor vs tta4 ROI entropy",
            audit["tests"],
            "Paired video-cluster bootstrap on both AUROC and AURC: `delta_aurc < 0` means "
            "the MC descriptor ranks failures better than the TTA view descriptor.",
        ),
    ]
    summary = write_markdown(
        str(output_dir / "summary.md"),
        title=f"MC-dropout summary: {experiment}",
        preamble=(
            f"{len(runs)} run(s); method `{args.method}`; statistics as in "
            "scripts/aggregate.py (video-level pairing, Holm).\n\n"
            + "\n".join(f"* {note}" for note in audit["notes"])
        ),
        sections=sections,
    )
    print("\nwritten:")
    for name, path in sorted(paths.items()):
        print(f"  {name:<28} {path}")
    print(f"  {'summary.md':<28} {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

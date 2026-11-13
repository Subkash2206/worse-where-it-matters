"""Evaluate every checkpoint of an internal CVC grid on an external polyp dataset.

For each external collection x each checkpoint of the internal grid (5 folds x 3 seeds per
architecture) x each TTA method (baseline / tta2 / tta4 / tta8) this runs the *same*
evaluation runner used internally, so the external numbers come from identical code: same
metrics, same uncertainty columns, same missed-detection and empty-reference bookkeeping.

The external cases are the test units (they are scored by every checkpoint), so the pooled
tables average the metrics **per case across the models** before any paired test - the 15
checkpoints do not multiply the sample - and the per-model spread is reported separately.

Usage
-----
    python scripts/evaluate_external.py --config configs/external/polyp_external.yaml \
        --models cvc_clinicdb/unet_resnet50 cvc_clinicdb/swinunetr_2d \
        --folds 0 1 2 3 4 --seeds 0 1 2
    python scripts/evaluate_external.py --config ... --dry-run      # plan only, no GPU

Nothing is written inside the external dataset directories; all outputs go to
``results/external_<dataset>/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from melba.config import Config, ConfigError, load_config, parse_set_args  # noqa: E402
from melba.data import build_dataset  # noqa: E402
from melba.eval import (  # noqa: E402
    evaluate,
    per_case_path,
    read_long_csv,
    string_ids,
    write_markdown,
)
from melba.eval.aggregate import write_tables  # noqa: E402
from melba.eval.external import external_tables  # noqa: E402
from melba.inference import methods_from_config  # noqa: E402
from melba.utils import git_revision  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="external evaluation YAML")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cvc_clinicdb/unet_resnet50", "cvc_clinicdb/swinunetr_2d"],
        help="internal configs (relative to configs/, without the .yaml suffix)",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--datasets", nargs="+", default=None, help="external.datasets keys (default: all)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N cases (smoke run)"
    )
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="skip the evaluations and only re-pool the per-case CSVs that already exist",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-evaluate runs whose per-case CSV already exists (default: resume, skip)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def _metrics(config: Config) -> tuple[str, ...]:
    from melba.metrics import MetricSettings

    settings = MetricSettings.from_config(config)
    # the reference-free descriptors of a polyp-free cohort are pooled exactly like the
    # quality metrics, so the external tables summarise them per case across checkpoints
    return settings.gt_names + settings.prediction_names


def _references(external: Config, entry: dict[str, Any] | None = None) -> str:
    """``"mask"`` or ``"empty"``: does this external dataset ship reference masks?

    Read from ``external.datasets.<name>.references`` and falling back to
    ``dataset.references``, mirroring the loader (``melba.data.polyp_external``).
    """
    source: dict[str, Any] = dict(entry or {})
    raw = source.get("references", external.get("dataset.references", "mask"))
    text = str(raw if raw is not None else "mask").strip().lower()
    return "empty" if text in {"empty", "polyp_free", "polyp-free"} else "mask"


def _external_config(
    external: Config,
    model_config_path: Path,
    *,
    dataset_name: str,
    entry: dict[str, Any] | None = None,
    checkpoint: Path,
    fold: int,
    seed: int,
    output_dir: Path,
    save_maps: bool,
    limit: int | None,
) -> Config:
    """The model's own config, re-pointed at one external dataset and one checkpoint.

    ``entry`` is the ``external.datasets`` mapping of the dataset; it may be omitted when
    only the *paths of the outputs* are wanted (``_frames_by_model``), in which case the
    dataset directories fall back to the config's own ``dataset.images_dir``/``masks_dir``
    - the caller then never touches the images.
    """
    settings = external.section("dataset")
    entry = dict(entry or {})
    images_dir = entry.get("images_dir") or settings.get("images_dir")
    references = _references(external, entry)
    masks_dir = entry["masks_dir"] if "masks_dir" in entry else settings.get("masks_dir")
    if references == "empty":
        # a reference-free cohort: no mask directory at all, every reference is empty
        masks_dir = None
    if images_dir is None:  # pragma: no cover - guarded by the YAML
        raise ConfigError(
            f"external dataset {dataset_name!r} has no images_dir in external.datasets "
            "nor in dataset.images_dir"
        )
    if references != "empty" and masks_dir is None:  # pragma: no cover - guarded by the YAML
        raise ConfigError(
            f"external dataset {dataset_name!r} has no masks_dir in external.datasets nor in "
            "dataset.masks_dir; declare `references: empty` if the dataset has no masks"
        )
    overrides: dict[str, Any] = {
        "experiment.name": f"{dataset_name}__{model_config_path.stem}",
        "dataset.kind": "external_polyp",
        "dataset.name": dataset_name,
        "dataset.images_dir": images_dir,
        "dataset.masks_dir": masks_dir,
        "dataset.references": references,
        "dataset.image_extensions": settings.get("image_extensions"),
        "dataset.mask_threshold": settings.get("mask_threshold"),
        "dataset.image_size": settings.get("image_size"),
        "dataset.spacing_mm": settings.get("spacing_mm"),
        "metrics.names": entry.get("metrics") or external.section("metrics").get("names"),
        "metrics.empty_policy": external.section("metrics").get("empty_policy"),
        "model.checkpoint": str(checkpoint),
        "evaluation.split": str(external.get("evaluation.split", "test")),
        "evaluation.fold": int(fold),
        "dataset.split.fold": int(fold),
        "evaluation.output_dir": str(output_dir),
        "evaluation.save_probability_maps": bool(save_maps),
        "evaluation.probability_map_methods": external.get(
            "evaluation.probability_map_methods", ["baseline", "tta4"]
        ),
        "evaluation.probability_map_cases": external.get(
            "evaluation.probability_map_cases", 10
        ),
        "evaluation.probability_map_case_seed": external.get(
            "evaluation.probability_map_case_seed", 0
        ),
        "runtime.progress": False,
        "seed": int(seed),
    }
    return load_config(model_config_path, overrides)


def _checkpoint_for(model_config: Config, fold: int, seed: int) -> Path:
    pattern = str(model_config.require("model.checkpoint"))
    return model_config.resolve(pattern.format(fold=int(fold), seed=int(seed)))


def _plan(
    model_paths: list[Path],
    datasets: list[str],
    folds: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Every (dataset, model, fold, seed) that will be evaluated, with its checkpoint."""
    steps: list[dict[str, Any]] = []
    for name in datasets:
        for model_path in model_paths:
            base = load_config(model_path)
            for fold in folds:
                for seed in seeds:
                    steps.append(
                        {
                            "dataset": name,
                            "model_path": model_path,
                            "model": f"{model_path.parent.name}/{model_path.stem}",
                            "fold": int(fold),
                            "seed": int(seed),
                            "checkpoint": _checkpoint_for(base, fold, seed),
                        }
                    )
    return steps


def _eval_dir(dataset_dir: Path, model_config_path: Path) -> Path:
    """Per-model evaluation directory of one external dataset.

    The per-case file name carries the dataset, the fold and the seed but not the model, so
    both architectures would write to the same path: the second model was reported as
    ``exists, skipped`` although it had never run, and the pool then held 15 checkpoints
    labelled as 30. One directory per model keeps the names unique.
    """
    return dataset_dir / model_config_path.stem / "evaluation"


def per_case_complete(
    path: Path, *, n_cases: int, methods: Sequence[str], model_stem: str | None = None
) -> tuple[bool, str]:
    """Is this per-case CSV a *finished* run of ``n_cases`` cases and ``methods``?

    A killed process leaves a valid-looking but truncated CSV behind, and the previous resume
    rule ("the file exists") then skipped that checkpoint for ever.  The check therefore reads
    the file and verifies the coverage: every method of the run must be present and every
    method must have one row set per case.  The reason string is printed when a run is redone.

    ``model_stem`` additionally requires the file to have been produced by that model config
    (its ``config_path`` column); the aggregate uses it when it looks in the pre-split layout,
    where every architecture wrote to the same directory.
    """
    if not path.is_file():
        return False, "missing"
    columns = ["case_id", "method"]
    if model_stem is not None:
        columns.append("config_path")
    try:
        frame = pd.read_csv(path, usecols=columns)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, UnicodeDecodeError) as error:
        return False, f"unreadable ({type(error).__name__})"
    if frame.empty:
        return False, "empty"
    if model_stem is not None:
        produced_by = str(frame["config_path"].iloc[0])
        if Path(produced_by).stem != model_stem:
            return False, f"produced by {Path(produced_by).stem}, not {model_stem}"
    per_method = frame.groupby("method")["case_id"].nunique()
    missing = sorted(set(methods) - set(per_method.index))
    if missing:
        return False, f"missing method(s) {missing}"
    short = {name: int(per_method[name]) for name in methods if int(per_method[name]) != int(n_cases)}
    if short:
        return False, f"case coverage {short} instead of {int(n_cases)}"
    return True, f"{int(n_cases)} cases x {len(methods)} method(s)"


def _per_case_candidates(
    dataset_dir: Path, model_config_path: Path, dataset_name: str, fold: int, seed: int
) -> list[Path]:
    """Where the per-case CSV of a run may live, best first.

    The architecture split moved every run into ``<dataset>/<model>/evaluation``; the runs made
    before that split (and the external UNet baseline they hold) are still directly in
    ``<dataset>/evaluation``.  Both are accepted, the new layout first.
    """
    name = f"{dataset_name}_test_fold{fold}_seed{seed}_per_case.csv"
    return [
        dataset_dir / model_config_path.stem / "evaluation" / name,
        dataset_dir / "evaluation" / name,
    ]


def resolve_per_case(
    candidates: Sequence[Path],
    *,
    n_cases: int,
    methods: Sequence[str],
    model_stem: str,
) -> tuple[Path, bool, str]:
    """First candidate that holds a complete run; else the new-layout path and why it failed."""
    reason = "missing"
    for position, path in enumerate(candidates):
        complete, detail = per_case_complete(
            path,
            n_cases=n_cases,
            methods=methods,
            model_stem=model_stem if position else None,
        )
        if complete:
            return path, True, detail
        if path.is_file():
            reason = detail
    return candidates[0], False, reason


def _frames_by_model(
    external: Config,
    steps: list[dict[str, Any]],
    dataset_name: str,
    dataset_dir: Path,
    *,
    n_cases: int,
    expected_methods: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Read the per-case CSV of every finished run of one external dataset."""
    frames: dict[str, pd.DataFrame] = {}
    for step in steps:
        if step["dataset"] != dataset_name:
            continue
        output_dir = _eval_dir(dataset_dir, step["model_path"])
        config = _external_config(
            external,
            step["model_path"],
            dataset_name=dataset_name,
            checkpoint=Path("unused"),
            fold=step["fold"],
            seed=step["seed"],
            output_dir=output_dir,
            save_maps=False,
            limit=None,
        )
        candidates = _per_case_candidates(
            dataset_dir, step["model_path"], dataset_name, step["fold"], step["seed"]
        )
        path, complete, reason = resolve_per_case(
            candidates,
            n_cases=n_cases,
            methods=expected_methods,
            model_stem=step["model_path"].stem,
        )
        if not complete:
            print(f"  incomplete run (skipped): {path.name} - {reason}")
            continue
        label = f"{step['model']}_fold{step['fold']}_seed{step['seed']}"
        frames[label] = string_ids(read_long_csv(path))
    return frames


def _write_external_markdown(
    dataset_dir: Path,
    entry: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    n_models: int,
    *,
    maskless: bool = False,
    metrics: Sequence[str] = (),
) -> None:
    """One markdown report per external dataset (internal vs external kept explicit)."""
    spread = tables.get("external_model_spread", pd.DataFrame())
    paired = tables.get("external_paired", pd.DataFrame())
    detection = tables.get("external_detection", pd.DataFrame())
    false_positive = tables.get("external_false_positive", pd.DataFrame())
    quality = ["dice", "hd95", "boundary_f1", "assd"]
    spread_view = (
        spread.loc[spread["metric"].isin(quality)]
        if not spread.empty and spread["metric"].isin(quality).any()
        else spread
    )
    paired_view = (
        paired.loc[paired["metric"].isin(["dice", "hd95"])] if not paired.empty else paired
    )
    if maskless:
        false_positive_note = (
            "**This is the primary table of this dataset.** Every frame is polyp-free, so "
            "every reference is empty: `false_positive_rate` is the share of frames on which "
            "the method reports a polyp at all, `correct_to_fp`/`fp_to_correct` count the "
            "frames the method invents that `baseline` does not (and vice versa), and "
            "`mcnemar_p` is the exact test on those discordant frames. There is no Dice or "
            "HD95 here: without a lesion they are undefined, not zero."
        )
    else:
        false_positive_note = (
            "None of the three external collections contains an empty reference mask, so this "
            "table is expected to report `n_cases = 0`; it is kept because the internal BraTS "
            "runs use the same code path and the two scopes must stay comparable."
        )
    sections = [
        (
            "Per-model spread over the internal checkpoints (external data)",
            spread_view,
            "Every number comes from the same external cases scored by each checkpoint of the "
            "internal CVC grid; `mean`/`std`/`min`/`max`/`range` describe how much the external "
            "result depends on the checkpoint drawn, not on the dataset.",
        ),
        (
            "Paired TTA comparisons on the external dataset (cases averaged over models)",
            paired_view,
            "Metrics are first averaged per external case across the checkpoints, so each image "
            "is one independent unit; the CI and the Wilcoxon test then describe the TTA effect "
            "on the external cases, not on the models.",
        ),
        (
            "Missed detections on the external dataset",
            detection,
            "An empty prediction on an external polyp is a failure to detect; the rate is over "
            "the external cases (each image is its own unit - these collections have no video "
            "structure, so no clustering is claimed).",
        ),
        (
            "False positives on empty references",
            false_positive,
            false_positive_note,
        ),
    ]
    write_markdown(
        str(dataset_dir / "summary.md"),
        title=f"MELBA external validation - {entry.get('label', dataset_dir.name)}",
        preamble=(
            f"External dataset `{entry.get('label', dataset_dir.name)}` "
            f"(`{entry.get('images_dir')}`) scored by {n_models} internal checkpoints over "
            f"{', '.join(metrics) if metrics else 'the configured metrics'}. Scope: "
            "**external** - never trained on. The internal CVC-ClinicDB tables live in "
            "`results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`."
        ),
        sections=sections,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        external = load_config(args.config, parse_set_args(args.overrides))
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    available = external.section("external").get("datasets") or {}
    if not available:
        print("ERROR: the config declares no external.datasets", file=sys.stderr)
        return 2
    names = list(args.datasets) if args.datasets else list(available)
    unknown = sorted(set(names) - set(available))
    if unknown:
        print(f"ERROR: unknown external dataset(s) {unknown}", file=sys.stderr)
        return 2

    model_paths = [_REPO_ROOT / "configs" / f"{model}.yaml" for model in args.models]
    missing_models = [str(path) for path in model_paths if not path.is_file()]
    if missing_models:
        print(f"ERROR: model config(s) not found: {missing_models}", file=sys.stderr)
        return 2

    output_root = Path(args.output_dir) if args.output_dir else _REPO_ROOT / "results"
    stats = external.section("stats")
    n_bootstrap = int(args.n_bootstrap or stats.get("n_bootstrap", 10000))
    ci = float(stats.get("ci", 0.95))
    bootstrap_seed = int(stats.get("bootstrap_seed", 0))
    correction = str(stats.get("correction", "holm"))
    unit = str(stats.get("unit", "group"))
    primary_test = str(stats.get("primary_test", "wilcoxon"))
    alternative = str(stats.get("alternative", "two-sided"))
    split = str(external.get("evaluation.split", "test"))

    steps = _plan(model_paths, names, list(args.folds), list(args.seeds))
    absent = [step for step in steps if not step["checkpoint"].is_file()]
    print(
        f"{len(steps)} run(s): {len(names)} external dataset(s) x {len(model_paths)} model(s) "
        f"x {len(args.folds)} fold(s) x {len(args.seeds)} seed(s)"
    )
    for step in absent:
        print(
            f"  WARNING no checkpoint: {step['model']} fold{step['fold']} seed{step['seed']} "
            f"({step['checkpoint']})"
        )
    if args.dry_run:
        for step in steps[:12]:
            print(
                f"  {step['dataset']:<14} {step['model']:<32} "
                f"fold{step['fold']} seed{step['seed']}"
            )
        if len(steps) > 12:
            print(f"  ... {len(steps) - 12} more")
        return 0

    started = time.perf_counter()
    for dataset_name in names:
        entry = available[dataset_name]
        dataset_dir = output_root / f"external_{dataset_name}"
        (dataset_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        # a polyp-free cohort (references: empty) has no Dice/HD95 to pool: its metrics are
        # the reference-free descriptors, and its false-positive table is the result
        maskless = _references(external, entry) == "empty"
        dataset_metrics = tuple(entry.get("metrics") or _metrics(external))
        print(f"\n=== {dataset_name} ({entry.get('label', dataset_name)}) -> {dataset_dir}")
        # size and method set of the dataset come from its own config, never from a guess
        expectations: tuple[int, list[str]] | None = None
        for position, step in enumerate(steps):
            if step["dataset"] != dataset_name:
                continue
            eval_dir = _eval_dir(dataset_dir, step["model_path"])
            eval_dir.mkdir(parents=True, exist_ok=True)
            # the ten saved maps belong to the first run of the dataset, not to all fifteen
            save_maps = bool(external.get("evaluation.probability_map_cases")) and position == 0
            config = _external_config(
                external,
                step["model_path"],
                dataset_name=dataset_name,
                entry=entry,
                checkpoint=step["checkpoint"],
                fold=step["fold"],
                seed=step["seed"],
                output_dir=eval_dir,
                save_maps=save_maps,
                limit=args.limit,
            )
            if expectations is None:
                expectations = (
                    len(build_dataset(config, split=split)),
                    [method.name for method in methods_from_config(config)],
                )
            n_cases, expected_methods = expectations
            candidates = _per_case_candidates(
                dataset_dir, step["model_path"], dataset_name, step["fold"], step["seed"]
            )
            expected, complete, reason = resolve_per_case(
                candidates,
                n_cases=n_cases,
                methods=expected_methods,
                model_stem=step["model_path"].stem,
            )
            if args.aggregate_only:
                print(f"[{step['model']} fold{step['fold']} seed{step['seed']}] aggregate only")
                continue
            if complete and not args.force:
                # resume: a *finished* run (all cases, all methods) is never redone
                print(f"[{step['model']} fold{step['fold']} seed{step['seed']}] complete, skipped")
                continue
            if complete is False and candidates[0].is_file():
                print(
                    f"[{step['model']} fold{step['fold']} seed{step['seed']}] "
                    f"incomplete ({reason}), re-evaluating"
                )
            print(
                f"[{step['model']} fold{step['fold']} seed{step['seed']}] evaluating",
                flush=True,
            )
            try:
                result = evaluate(config, limit=args.limit, progress=False)
            except (ConfigError, FileNotFoundError, RuntimeError) as error:
                print(f"ERROR [{dataset_name}/{step['model']}]: {error}", file=sys.stderr)
                return 2
            print(f"  -> {result.n_cases} cases -> {result.per_case_csv.name}")

        frames = _frames_by_model(
            external,
            steps,
            dataset_name,
            dataset_dir,
            n_cases=expectations[0] if expectations else 0,
            expected_methods=expectations[1] if expectations else (),
        )
        if not frames:
            print(f"ERROR: no per-case results for {dataset_name}", file=sys.stderr)
            return 2
        print(f"  pooling {len(frames)} checkpoint run(s)")
        tables = external_tables(
            frames,
            metrics=dataset_metrics,
            control="baseline",
            treatments=("tta2", "tta4", "tta8"),
            unit=unit,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=bootstrap_seed,
            correction=correction,
            primary_test=primary_test,
            alternative=alternative,
            # a reference-free cohort reports detection, not a paired quality gain
            paired_metrics=() if maskless else None,
        )
        write_tables(
            tables,
            dataset_dir,
            manifest={
                "scope": "external",
                "dataset": dataset_name,
                "label": entry.get("label", dataset_name),
                "images_dir": entry.get("images_dir"),
                "masks_dir": entry.get("masks_dir"),
                "references": "empty" if maskless else "mask",
                "metrics": list(dataset_metrics),
                "internal_models": [path.stem for path in model_paths],
                "folds": list(args.folds),
                "seeds": list(args.seeds),
                "n_checkpoints": len(frames),
                "unit": unit,
                "n_bootstrap": n_bootstrap,
                "limit": args.limit,
                "git_revision": git_revision(),
            },
        )
        _write_external_markdown(
            dataset_dir,
            entry,
            tables,
            len(frames),
            maskless=maskless,
            metrics=dataset_metrics,
        )
        print(f"  written to {dataset_dir}")

    print(f"\ntotal wall clock: {(time.perf_counter() - started) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



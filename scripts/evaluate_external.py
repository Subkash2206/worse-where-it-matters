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
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from melba.config import Config, ConfigError, load_config, parse_set_args  # noqa: E402
from melba.eval import (  # noqa: E402
    evaluate,
    per_case_path,
    read_long_csv,
    string_ids,
    write_markdown,
)
from melba.eval.aggregate import write_tables  # noqa: E402
from melba.eval.external import external_tables  # noqa: E402
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

    return MetricSettings.from_config(config).gt_names


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
    masks_dir = entry.get("masks_dir") or settings.get("masks_dir")
    if images_dir is None or masks_dir is None:  # pragma: no cover - guarded by the YAML
        raise ConfigError(
            f"external dataset {dataset_name!r} has no images_dir/masks_dir in "
            "external.datasets nor in dataset.images_dir/dataset.masks_dir"
        )
    overrides: dict[str, Any] = {
        "experiment.name": f"{dataset_name}__{model_config_path.stem}",
        "dataset.kind": "external_polyp",
        "dataset.name": dataset_name,
        "dataset.images_dir": images_dir,
        "dataset.masks_dir": masks_dir,
        "dataset.image_extensions": settings.get("image_extensions"),
        "dataset.mask_threshold": settings.get("mask_threshold"),
        "dataset.image_size": settings.get("image_size"),
        "dataset.spacing_mm": settings.get("spacing_mm"),
        "metrics.names": external.section("metrics").get("names"),
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


def _frames_by_model(
    external: Config,
    steps: list[dict[str, Any]],
    dataset_name: str,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Read the per-case CSV of every finished run of one external dataset.

    ``output_dir`` is the *evaluation* directory the runs wrote into (already checked by the
    caller); it is used as given, so the paths here are the ones ``main`` used when it ran
    the evaluations - rebuilding them one level deeper silently found nothing.
    """
    frames: dict[str, pd.DataFrame] = {}
    for step in steps:
        if step["dataset"] != dataset_name:
            continue
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
        path = per_case_path(config, step["fold"], step["seed"], output_dir=output_dir)
        if not path.is_file():
            print(f"  missing run (skipped): {path.name}")
            continue
        label = f"{step['model']}_fold{step['fold']}_seed{step['seed']}"
        frames[label] = string_ids(read_long_csv(path))
    return frames


def _write_external_markdown(
    dataset_dir: Path,
    entry: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    n_models: int,
) -> None:
    """One markdown report per external dataset (internal vs external kept explicit)."""
    spread = tables.get("external_model_spread", pd.DataFrame())
    paired = tables.get("external_paired", pd.DataFrame())
    detection = tables.get("external_detection", pd.DataFrame())
    false_positive = tables.get("external_false_positive", pd.DataFrame())
    spread_view = (
        spread.loc[spread["metric"].isin(["dice", "hd95", "boundary_f1", "assd"])]
        if not spread.empty
        else spread
    )
    paired_view = (
        paired.loc[paired["metric"].isin(["dice", "hd95"])] if not paired.empty else paired
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
            "None of the three external collections contains an empty reference mask, so this "
            "table is expected to report `n_cases = 0`; it is kept because the internal BraTS "
            "runs use the same code path and the two scopes must stay comparable.",
        ),
    ]
    write_markdown(
        str(dataset_dir / "summary.md"),
        title=f"MELBA external validation - {entry.get('label', dataset_dir.name)}",
        preamble=(
            f"External dataset `{entry.get('label', dataset_dir.name)}` "
            f"(`{entry.get('images_dir')}`) scored by {n_models} internal checkpoints. Scope: "
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
        print(f"\n=== {dataset_name} ({entry.get('label', dataset_name)}) -> {dataset_dir}")
        for position, step in enumerate(steps):
            if step["dataset"] != dataset_name:
                continue
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
                output_dir=dataset_dir / "evaluation",
                save_maps=save_maps,
                limit=args.limit,
            )
            expected = per_case_path(
                config, step["fold"], step["seed"], output_dir=dataset_dir / "evaluation"
            )
            if args.aggregate_only or (expected.is_file() and not args.force):
                # resume: the run already wrote its per-case CSV, do not spend GPU time again
                print(f"[{step['model']} fold{step['fold']} seed{step['seed']}] exists, skipped")
                continue
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

        frames = _frames_by_model(external, steps, dataset_name, dataset_dir / "evaluation")
        if not frames:
            print(f"ERROR: no per-case results for {dataset_name}", file=sys.stderr)
            return 2
        print(f"  pooling {len(frames)} checkpoint run(s)")
        tables = external_tables(
            frames,
            metrics=_metrics(external),
            control="baseline",
            treatments=("tta2", "tta4", "tta8"),
            unit=unit,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=bootstrap_seed,
            correction=correction,
            primary_test=primary_test,
            alternative=alternative,
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
        _write_external_markdown(dataset_dir, entry, tables, len(frames))
        print(f"  written to {dataset_dir}")

    print(f"\ntotal wall clock: {(time.perf_counter() - started) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



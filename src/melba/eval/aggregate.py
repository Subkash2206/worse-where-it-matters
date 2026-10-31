"""Pooled multi-fold / multi-seed aggregation and seed ensembling.

``scripts/run_stats.py`` compares two methods on one run; this module pools the whole
cross-validation grid (5 folds x 3 seeds) instead:

* pooled per-video results per seed, then mean/std across seeds (``pooled_by_seed``);
* video-level paired tests pooled over the folds, once per seed and once with the
  seeds averaged per case first (``paired_by_seed`` / ``paired_seed_averaged``);
* missed-detection rates, transition tables and exact McNemar tests, pooled over
  folds, for both of the above;
* a **seed ensemble**: the mean probability map of the 3 seeds (``ensemble3``) and of
  the 3 seeds x 4 views (``ensemble3_tta4``), thresholded at 0.5 and evaluated with
  exactly the same metric code as the individual runs;
* paired comparisons between ``baseline``, ``tta4``, ``ensemble3`` and
  ``ensemble3_tta4`` on the same metrics, including missed detections.

Frames are only ever pooled across folds when every case and every video appears in
exactly one fold (enforced by :func:`melba.stats.pool_folds`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from melba.config import Config
from melba.eval.records import read_long_csv
from melba.metrics import MetricSettings, compute_case_metrics, grid_diagonal_mm
from melba.stats import analyze_detection, analyze_long, pool_folds

__all__ = [
    "ENSEMBLES",
    "ENSEMBLE_VIEWS",
    "NON_QUALITY_METRICS",
    "PENALIZED_SOURCE_METRICS",
    "AggregateResult",
    "RunFiles",
    "aggregate_all",
    "compare_methods",
    "detection_table",
    "discover_runs",
    "ensemble_frame",
    "frames_by_seed",
    "load_ground_truth",
    "map_dir_for",
    "per_case_path",
    "pooled_seed_table",
    "seed_averaged_frame",
    "string_ids",
    "write_tables",
]

#: distance metrics whose empty-prediction variant is written as ``<metric>_penalized``
PENALIZED_SOURCE_METRICS: tuple[str, ...] = ("hd", "hd95", "hd95_pooled", "assd")

#: ensemble name -> per-view method whose probability maps are averaged over seeds
ENSEMBLES: Mapping[str, str] = {"ensemble3": "baseline", "ensemble3_tta4": "tta4"}

#: number of forward passes per seed that go into each ensemble
ENSEMBLE_VIEWS: Mapping[str, int] = {"baseline": 1, "tta4": 4}

#: per-case metrics that are not qualities (excluded from every quality table)
NON_QUALITY_METRICS: tuple[str, ...] = ("latency_ms", "peak_memory_mb")


@dataclass(frozen=True)
class RunFiles:
    """The artefacts of one (fold, seed) run."""

    fold: int
    seed: int
    per_case: Path
    map_dir: Path


def _dataset_stem(config: Config, split: str | None = None) -> str:
    dataset_name = str(config.get("dataset.name", "dataset"))
    resolved_split = str(split or config.get("evaluation.split", "test"))
    return f"{dataset_name}_{resolved_split}"


def per_case_path(
    config: Config, fold: int, seed: int, *, output_dir: str | Path | None = None
) -> Path:
    """Path written by ``melba.eval.run_evaluation`` for one fold/seed."""
    base = (
        Path(output_dir)
        if output_dir is not None
        else config.resolve(config.require("evaluation.output_dir"))
    )
    return base / f"{_dataset_stem(config)}_fold{int(fold)}_seed{int(seed)}_per_case.csv"


def map_dir_for(
    config: Config,
    fold: int,
    seed: int,
    *,
    output_dir: str | Path | None = None,
    map_root: str | Path | None = None,
) -> Path:
    """Probability-map directory written by the runner for one fold/seed."""
    if map_root is not None:
        root = Path(map_root)
    elif config.get("evaluation.probability_map_dir"):
        root = config.resolve(str(config.get("evaluation.probability_map_dir")))
    else:
        base = (
            Path(output_dir)
            if output_dir is not None
            else config.resolve(config.require("evaluation.output_dir"))
        )
        root = base / "maps"
    return root / f"fold{int(fold)}_seed{int(seed)}"


def discover_runs(
    config: Config,
    *,
    folds: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    output_dir: str | Path | None = None,
    map_root: str | Path | None = None,
) -> list[RunFiles]:
    """Every fold/seed whose per-case CSV exists (missing runs are simply absent)."""
    resolved_folds = (
        list(range(int(config.get("dataset.split.n_folds", 5) or 1)))
        if folds is None
        else [int(fold) for fold in folds]
    )
    resolved_seeds = [0, 1, 2] if seeds is None else [int(seed) for seed in seeds]
    runs: list[RunFiles] = []
    for fold in resolved_folds:
        for seed in resolved_seeds:
            path = per_case_path(config, fold, seed, output_dir=output_dir)
            if path.is_file():
                runs.append(
                    RunFiles(
                        fold=fold,
                        seed=seed,
                        per_case=path,
                        map_dir=map_dir_for(
                            config, fold, seed, output_dir=output_dir, map_root=map_root
                        ),
                    )
                )
    return runs


def frames_by_seed(runs: Iterable[RunFiles]) -> dict[int, list[pd.DataFrame]]:
    """Per-case frames grouped by seed (one frame per fold), ids normalised to ``str``."""
    grouped: dict[int, list[pd.DataFrame]] = {}
    for run in sorted(runs, key=lambda item: (item.seed, item.fold)):
        grouped.setdefault(int(run.seed), []).append(string_ids(read_long_csv(run.per_case)))
    return grouped


def string_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast case/video ids to ``str``.

    The per-case CSV is parsed with numeric ``case_id`` values (CVC frames are named
    like ``1``, ``2``, ...) while ensemble rows key their masks by file stem, i.e. by
    ``str``.  Normalising here keeps a frame from the CSV and a frame from the maps
    mergeable, so the ensemble cannot silently pair with nothing.
    """
    normalised = frame.copy()
    for column in ("case_id", "group_id"):
        if column in normalised.columns:
            normalised[column] = normalised[column].astype(str)
    return normalised


def _quality_subset(frame: pd.DataFrame, metrics: Sequence[str] | None) -> pd.DataFrame:
    subset = frame.loc[~frame["metric"].isin(NON_QUALITY_METRICS)]
    if metrics is not None:
        subset = subset.loc[subset["metric"].isin(list(metrics))]
    return subset


def pooled_seed_table(
    frames_by_seed_map: Mapping[int, Sequence[pd.DataFrame]],
    *,
    unit: str = "group",
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Pooled per-video result per seed, then mean/std across seeds, per method+metric.

    For every seed the folds are pooled (29 videos, each a test unit exactly once) and
    a video's value is the mean over its frames; the table reports the mean of those 29
    values, the standard deviation across seeds, and each seed's mean so the spread can
    be inspected without opening the CSV.
    """
    if not frames_by_seed_map:
        raise ValueError("pooled_seed_table needs at least one seed")
    unit_column = "group_id" if unit == "group" else "case_id"
    per_seed: list[pd.DataFrame] = []
    for seed, frames in sorted(frames_by_seed_map.items()):
        pooled = _quality_subset(pool_folds(frames), metrics)
        per_unit = (
            pooled.groupby(["method", "metric", unit_column], as_index=False)["value"]
            .mean()
            .rename(columns={unit_column: "unit_id"})
        )
        table = per_unit.groupby(["method", "metric"], as_index=False).agg(
            n_units=("unit_id", "nunique"),
            pooled_mean=("value", "mean"),
        )
        counts = pooled.groupby(["method", "metric"], as_index=False).agg(
            n_cases=("case_id", "nunique")
        )
        table = table.merge(counts, on=["method", "metric"], how="left")
        table["seed"] = int(seed)
        per_seed.append(table)

    combined = pd.concat(per_seed, ignore_index=True)
    summary = combined.groupby(["method", "metric"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        n_units=("n_units", "max"),
        n_cases=("n_cases", "max"),
        mean=("pooled_mean", "mean"),
        std=(
            "pooled_mean",
            lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        ),
    )
    pivot = combined.pivot_table(index=["method", "metric"], columns="seed", values="pooled_mean")
    pivot.columns = [f"mean_seed{int(column)}" for column in pivot.columns]
    table = summary.merge(pivot.reset_index(), on=["method", "metric"], how="left")
    return table.sort_values(["method", "metric"]).reset_index(drop=True)


def _majority(values: pd.Series) -> str:
    counts = values.value_counts()
    top = counts[counts == counts.max()].index.tolist()
    return sorted(str(value) for value in top)[0]


def seed_averaged_frame(
    frames_by_seed_map: Mapping[int, Sequence[pd.DataFrame]],
) -> pd.DataFrame:
    """Average metric values per case over seeds; a case is missed when most seeds miss.

    Averaging the *values* per case (instead of pooling seeds as if they were extra
    independent cases) keeps one test unit per video and does not triple the sample.
    """
    if not frames_by_seed_map:
        raise ValueError("seed_averaged_frame needs at least one seed")
    parts: list[pd.DataFrame] = []
    for seed, frames in sorted(frames_by_seed_map.items()):
        for frame in frames:
            part = _quality_subset(frame, None).copy()
            part["seed"] = int(seed)
            parts.append(part)
    combined = pd.concat(parts, ignore_index=True)

    keys = ["case_id", "group_id", "fold", "split", "method", "metric"]
    values = combined.groupby(keys, as_index=False).agg(
        value=("value", "mean"), n_seeds=("seed", "nunique")
    )
    flags = (
        combined.drop_duplicates(["case_id", "method", "seed"])
        .groupby(["case_id", "method"], as_index=False)
        .agg(
            empty_flag=("empty_flag", _majority),
            missed_fraction=("missed_detection", "mean"),
        )
    )
    merged = values.merge(flags, on=["case_id", "method"], how="left")
    merged["missed_detection"] = (merged["missed_fraction"] >= 0.5).astype(int)
    merged = merged.drop(columns=["missed_fraction"])
    merged["seed"] = -1
    merged["run_id"] = "seed_averaged"
    return merged


def compare_methods(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    metrics: Sequence[str] | None = None,
    unit: str = "group",
    primary_test: str = "wilcoxon",
    alternative: str = "two-sided",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
) -> pd.DataFrame:
    """Paired tests for several (treatment, control) pairs on one pooled frame."""
    if not pairs:
        return pd.DataFrame()
    pooled = pool_folds([frame])
    tables: list[pd.DataFrame] = []
    for treatment, control in pairs:
        table = analyze_long(
            pooled,
            treatment=treatment,
            control=control,
            metrics=list(metrics) if metrics is not None else None,
            unit=unit,
            primary_test=primary_test,
            alternative=alternative,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            correction=correction,
        )
        # analyze_long already records the treatment/control columns
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def detection_table(
    frame: pd.DataFrame,
    *,
    methods: Sequence[str],
    control: str = "baseline",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
) -> pd.DataFrame:
    """Missed-detection rates, transitions and exact McNemar tests, pooled over folds."""
    return analyze_detection(
        pool_folds([frame]),
        control=control,
        methods=list(methods),
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        correction=correction,
    )



def load_ground_truth(
    config: Config, folds: Sequence[int], *, split: str | None = None
) -> dict[int, dict[str, dict[str, Any]]]:
    """Reference masks on the native grid, per fold and case (for the ensemble)."""
    from melba.data import build_dataset  # local import: keeps module import light

    resolved_split = str(split or config.get("evaluation.split", "test"))
    ground_truth: dict[int, dict[str, dict[str, Any]]] = {}
    for fold in folds:
        dataset = build_dataset(config, resolved_split, fold=int(fold))
        entries: dict[str, dict[str, Any]] = {}
        for index in range(len(dataset)):
            sample = dataset[index]
            mask = sample["mask_native"][0]
            array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
            entries[str(sample["case_id"])] = {
                "mask": np.asarray(array, dtype=np.float32),
                "spacing": tuple(float(value) for value in sample.get("spacing", (1.0, 1.0))),
                "group_id": str(sample.get("group_id", sample["case_id"])),
            }
        ground_truth[int(fold)] = entries
    return ground_truth


def write_tables(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write every non-empty table as CSV plus an optional ``manifest.json``."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        if table is None or table.empty:
            continue
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    if manifest is not None:
        (output / "manifest.json").write_text(
            json.dumps(dict(manifest), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return paths


def ensemble_frame(
    config: Config,
    runs: Sequence[RunFiles],
    *,
    name: str,
    source_method: str,
    ground_truth: Mapping[int, Mapping[str, Mapping[str, Any]]] | None = None,
    settings: MetricSettings | None = None,
    threshold: float = 0.5,
    split: str | None = None,
) -> pd.DataFrame:
    """Evaluate the seed-averaged probability map as a method of its own.

    For every case the float16 maps written by the runner for ``source_method`` are
    averaged over the seeds (arithmetic mean of probabilities), thresholded and scored
    with the same metric code as the individual runs, so the ensemble is directly
    comparable to ``baseline``/``tta4``.

    The starting point is float16, so a 0.5 decision can differ from the runner's
    float32 computation on borderline pixels: on the real CVC-ClinicDB fold 0 run that
    changed Dice by < 5e-4 on all 118 cases and HD95/ASSD by < 0.6 px.  Every ensemble
    is scored through this same path, so ensembles stay comparable to each other and
    to the individual methods up to that quantisation.
    """
    if not runs:
        raise ValueError("ensemble_frame needs at least one run")
    if config is None:
        raise ValueError("ensemble_frame needs the experiment config")
    resolved_split = str(split or config.get("evaluation.split", "test"))
    settings = settings or MetricSettings.from_config(config)
    by_fold: dict[int, list[RunFiles]] = {}
    for run in sorted(runs, key=lambda item: (item.fold, item.seed)):
        by_fold.setdefault(int(run.fold), []).append(run)

    if ground_truth is None:
        ground_truth = load_ground_truth(config, sorted(by_fold), split=resolved_split)

    dataset_name = str(config.get("dataset.name", "dataset"))
    architecture = str(config.get("model.architecture", "model"))
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict[str, Any]] = []
    for fold, fold_runs in sorted(by_fold.items()):
        entries = ground_truth.get(fold)
        if not entries:
            raise ValueError(f"no ground truth for fold {fold}; re-run load_ground_truth")
        seeds = sorted(run.seed for run in fold_runs)
        views = ENSEMBLE_VIEWS.get(source_method, 1) * len(seeds)
        for case_id, entry in entries.items():
            paths = [
                run.map_dir / f"{case_id}__{source_method}__mean.npy" for run in fold_runs
            ]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"probability maps for {name} are incomplete; missing {missing[:3]}"
                )
            stack = np.stack([np.load(path).astype(np.float32) for path in paths], axis=0)
            probability = stack.mean(axis=0)
            metrics = compute_case_metrics(
                probability,
                entry["mask"],
                spacing=entry["spacing"],
                threshold=threshold,
                settings=settings,
            )
            values: dict[str, float] = {key: metrics[key] for key in settings.gt_names}
            diagonal = grid_diagonal_mm(np.asarray(entry["mask"]).shape, entry["spacing"])
            for source in PENALIZED_SOURCE_METRICS:
                if source in values:
                    value = float(values[source])
                    values[f"{source}_penalized"] = diagonal if np.isnan(value) else value
            spacing_text = ",".join(f"{float(value):g}" for value in entry["spacing"])
            provenance = {
                "run_id": f"ensemble-{source_method}-seeds{''.join(str(seed) for seed in seeds)}",
                "created_utc": created,
                "config_path": str(config.path) if config.path else "",
                "config_hash": config.fingerprint,
                "seed": -1,
                "dataset": dataset_name,
                "model": f"{architecture}+seed_ensemble",
                "split": resolved_split,
                "fold": int(fold),
                "method": name,
                "aggregator": "mean_over_seeds",
                "n_views": int(views),
                "view_names": "",
                "threshold": float(threshold),
                "spacing_mm": spacing_text,
                "case_id": case_id,
                "group_id": entry["group_id"],
                "empty_flag": metrics["empty_flag"],
                "missed_detection": int(metrics["empty_flag"] == "pred_empty"),
            }
            for metric_name, value in values.items():
                rows.append({**provenance, "metric": metric_name, "value": float(value)})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class AggregateResult:
    """Tables produced by :func:`aggregate_all` (plus where they were written)."""

    tables: dict[str, pd.DataFrame]
    notes: tuple[str, ...] = ()
    paths: dict[str, Path] | None = None


def aggregate_all(
    config: Config,
    *,
    runs: Sequence[RunFiles],
    metrics: Sequence[str] | None = None,
    unit: str = "group",
    threshold: float = 0.5,
    treatments: Sequence[str] = ("tta2", "tta4", "tta8"),
    control: str = "baseline",
    primary_test: str = "wilcoxon",
    alternative: str = "two-sided",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    bootstrap_seed: int = 0,
    correction: str = "holm",
    ensemble: bool = True,
    expected_seeds: Sequence[int] = (0, 1, 2),
    ground_truth: Mapping[int, Mapping[str, Mapping[str, Any]]] | None = None,
    output_dir: str | Path | None = None,
) -> AggregateResult:
    """Build every pooled table; write CSVs when ``output_dir`` is given."""
    if not runs:
        raise ValueError(
            "no per-case results found; run scripts/run_grid.py (or point --output-dir at "
            "an existing evaluation directory)"
        )
    settings = MetricSettings.from_config(config)
    quality_metrics = tuple(metrics) if metrics is not None else tuple(settings.gt_names)
    by_seed = frames_by_seed(runs)
    methods = sorted(
        {
            str(method)
            for frames in by_seed.values()
            for frame in frames
            for method in frame["method"].unique()
        }
    )
    pairs = [(treatment, control) for treatment in treatments if treatment in methods]
    notes: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    tables["pooled_by_seed"] = pooled_seed_table(
        by_seed, unit=unit, metrics=quality_metrics
    )

    paired_by_seed: list[pd.DataFrame] = []
    detection_by_seed: list[pd.DataFrame] = []
    for seed, frames in sorted(by_seed.items()):
        pooled = pool_folds(frames)
        if pairs:
            table = compare_methods(
                pooled,
                pairs,
                metrics=quality_metrics,
                unit=unit,
                primary_test=primary_test,
                alternative=alternative,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=bootstrap_seed,
                correction=correction,
            )
            table.insert(0, "seed", int(seed))
            paired_by_seed.append(table)
        detection = detection_table(
            pooled,
            methods=[control] + [treatment for treatment, _ in pairs],
            control=control,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=bootstrap_seed,
            correction=correction,
        )
        detection.insert(0, "seed", int(seed))
        detection_by_seed.append(detection)

    tables["paired_by_seed"] = (
        pd.concat(paired_by_seed, ignore_index=True) if paired_by_seed else pd.DataFrame()
    )
    tables["detection_by_seed"] = (
        pd.concat(detection_by_seed, ignore_index=True)
        if detection_by_seed
        else pd.DataFrame()
    )

    averaged = seed_averaged_frame(by_seed)
    tables["paired_seed_averaged"] = (
        compare_methods(
            averaged,
            pairs,
            metrics=quality_metrics,
            unit=unit,
            primary_test=primary_test,
            alternative=alternative,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=bootstrap_seed,
            correction=correction,
        )
        if pairs
        else pd.DataFrame()
    )
    tables["detection_seed_averaged"] = detection_table(
        averaged,
        methods=[control] + [treatment for treatment, _ in pairs],
        control=control,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=bootstrap_seed,
        correction=correction,
    )


    available_seeds = sorted(by_seed)
    required_seeds = sorted({int(seed) for seed in expected_seeds})
    if ensemble and set(required_seeds) <= set(available_seeds):
        ensemble_runs = [run for run in runs if run.seed in set(required_seeds)]
        frames: list[pd.DataFrame] = []
        for name, source in ENSEMBLES.items():
            if source not in methods:
                notes.append(f"{name} skipped: method {source!r} is not in the results")
                continue
            try:
                frames.append(
                    ensemble_frame(
                        config,
                        ensemble_runs,
                        name=name,
                        source_method=source,
                        ground_truth=ground_truth,
                        settings=settings,
                        threshold=threshold,
                    )
                )
            except FileNotFoundError as error:
                notes.append(f"{name} skipped: {error}")
        if frames:
            ensemble_rows = pd.concat(frames, ignore_index=True)
            tables["ensemble_per_case"] = ensemble_rows
            comparison = pd.concat(
                [averaged.drop(columns=["n_seeds"], errors="ignore"), ensemble_rows],
                ignore_index=True,
            )
            present = set(comparison["method"].unique())
            ensemble_pairs = [
                ("tta4", control),
                ("ensemble3", control),
                ("ensemble3_tta4", control),
                ("ensemble3_tta4", "tta4"),
                ("ensemble3", "tta4"),
            ]
            ensemble_pairs = [
                (treatment, comparison_control)
                for treatment, comparison_control in ensemble_pairs
                if treatment in present and comparison_control in present
            ]
            tables["ensemble_comparison"] = (
                compare_methods(
                    comparison,
                    ensemble_pairs,
                    metrics=quality_metrics,
                    unit=unit,
                    primary_test=primary_test,
                    alternative=alternative,
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                    seed=bootstrap_seed,
                    correction=correction,
                )
                if ensemble_pairs
                else pd.DataFrame()
            )
            detect_methods = sorted(
                {method for pair in ensemble_pairs for method in pair} | {control}
            )
            tables["ensemble_detection"] = detection_table(
                comparison,
                methods=detect_methods,
                control=control,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=bootstrap_seed,
                correction=correction,
            )
    elif ensemble:
        notes.append(
            "ensemble skipped: needs seeds "
            f"{required_seeds}, found {available_seeds}"
        )

    paths: dict[str, Path] | None = None
    if output_dir is not None:
        manifest = {
            "config_path": str(config.path) if config.path else None,
            "config_hash": config.fingerprint,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "folds": sorted({run.fold for run in runs}),
            "seeds": available_seeds,
            "runs": [
                {"fold": run.fold, "seed": run.seed, "per_case": str(run.per_case)}
                for run in runs
            ],
            "unit": unit,
            "metrics": list(quality_metrics),
            "threshold": float(threshold),
            "treatments": list(treatments),
            "control": control,
            "n_bootstrap": int(n_bootstrap),
            "ci": float(ci),
            "correction": correction,
            "notes": notes,
        }
        paths = write_tables(tables, output_dir, manifest=manifest)

    tables = {name: table for name, table in tables.items() if table is not None}
    return AggregateResult(tables=tables, notes=tuple(notes), paths=paths)


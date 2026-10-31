"""Per-case evaluation runner.

The runner turns a config + checkpoint into a long-format per-case CSV.  It never
aggregates before writing: summarisation happens from the per-case file, so a mean
in a paper table can always be traced back to individual cases.

Two scientific guards are enforced here:

* the reported split may not be the split used for checkpoint selection
  (``model.selection_split``), which was one of the bugs of the old code base;
* the split is read from disk (video level), so no frame-level re-splitting can
  happen inside the evaluation.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from melba.config import Config, ConfigError
from melba.data import build_dataset
from melba.inference import (
    TTAMethod,
    aggregate_probabilities,
    methods_from_config,
    predict_per_view,
)
from melba.metrics import (
    MetricSettings,
    compute_case_metrics,
    compute_view_uncertainty,
    grid_diagonal_mm,
)
from melba.models import build_model, checkpoint_sha256, load_checkpoint
from melba.utils import describe_device, environment_info, get_device, seed_everything
from melba.utils.provenance import git_revision

from .records import (
    LongResultWriter,
    make_run_id,
    read_long_csv,
    summarize_long,
    utc_now,
    write_json,
)

__all__ = ["EvaluationResult", "evaluate", "run_evaluation", "TIMING_METRICS"]

#: per-case cost descriptors written next to the quality metrics
TIMING_METRICS: tuple[str, ...] = ("latency_ms", "peak_memory_mb")

#: distance metrics for which a penalised variant (empty prediction -> image
#: diagonal) is written as ``<metric>_penalized`` for the sensitivity table
PENALIZED_SOURCE_METRICS: tuple[str, ...] = ("hd", "hd95", "hd95_pooled", "assd")


@dataclass(frozen=True)
class EvaluationResult:
    """Paths and basic facts about one evaluation run."""

    run_id: str
    split: str
    n_cases: int
    methods: tuple[str, ...]
    per_case_csv: Path
    summary_csv: Path | None
    metadata_json: Path
    fold: int = 0
    seed: int = 0


def _resolve_output_dir(config: Config, override: str | Path | None) -> Path:
    if override is not None:
        path = Path(override)
        path = path if path.is_absolute() else config.resolve(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return config.ensure_dir("evaluation.output_dir")


def _guard_reported_split(config: Config, split: str, allow: bool) -> str:
    selection_split = str(config.get("model.selection_split", "val"))
    if split == selection_split and not allow:
        raise ConfigError(
            f"Refusing to report split '{split}': it is also the checkpoint selection split "
            f"(model.selection_split='{selection_split}'). Select on val and report on test, "
            "or set evaluation.allow_selection_split: true for an explicitly flagged "
            "development run."
        )
    if split == "train":
        raise ConfigError(
            "Refusing to run model evaluation on the training split. "
            "Use split 'val' for model selection or 'test' for reported results."
        )
    return selection_split


def _spacing_text(spacing: Sequence[float]) -> str:
    return ",".join(f"{float(value):g}" for value in spacing)


def _resolve_checkpoint(config: Config, fold: int, seed: int = 0) -> Path | None:
    """Resolve ``model.checkpoint``, substituting the fold and seed when asked for.

    ``melba.training.train`` writes one checkpoint per fold/seed run
    (``.../<experiment>/fold{k}_seed{s}/best.pt``), so the config can declare
    ``model.checkpoint: checkpoints/<experiment>/fold{fold}_seed{seed}/best.pt`` and
    the same config is valid for every run without manual overrides.  Patterns that
    only mention ``{fold}`` keep working.
    """
    setting = config.get("model.checkpoint")
    if not setting:
        return None
    return config.resolve(str(setting).format(fold=int(fold), seed=int(seed)))


def _synchronize(device: torch.device) -> None:
    """Block until all pending CUDA kernels of this device are finished."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    """Peak allocated GPU memory since the last reset (NaN on CPU)."""
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)


def resample_probability_map(
    probability: np.ndarray, target_shape: Sequence[int]
) -> np.ndarray:
    """Resample a ``(C, *spatial)`` probability map onto the reference grid.

    Predictions are made at the configured network resolution but every metric is
    reported on the native grid of the reference mask (bilinear/trilinear
    resampling, no thresholding before resampling).
    """
    array = np.asarray(probability, dtype=np.float32)
    target = tuple(int(size) for size in target_shape)
    if tuple(array.shape[1:]) == target:
        return array
    spatial_dims = array.ndim - 1
    if spatial_dims not in (2, 3):
        raise ValueError(
            f"Expected a (C, H, W) or (C, H, W, D) probability map, got shape {array.shape}"
        )
    mode = "bilinear" if spatial_dims == 2 else "trilinear"
    kwargs: dict[str, Any] = {"align_corners": False}
    if mode == "bilinear":
        kwargs["antialias"] = True
    with torch.inference_mode():
        resized = functional.interpolate(
            torch.from_numpy(array).unsqueeze(0), size=target, mode=mode, **kwargs
        )
    return resized[0].numpy().astype(np.float32)


def run_evaluation(
    config: Config,
    *,
    split: str | None = None,
    limit: int | None = None,
    output_dir: str | Path | None = None,
    model: torch.nn.Module | None = None,
    dataset: Any | None = None,
    methods: Sequence[TTAMethod] | None = None,
    allow_selection_split: bool | None = None,
    progress: bool | None = None,
    fold: int | None = None,
) -> EvaluationResult:
    """Evaluate all configured TTA methods on one split and write per-case results.

    Parameters
    ----------
    config:
        Loaded experiment configuration.
    split:
        Split to evaluate; defaults to ``evaluation.split`` (``test``).
    limit:
        Evaluate only the first ``limit`` cases (smoke tests).
    output_dir:
        Override ``evaluation.output_dir``.
    model, dataset, methods:
        Optional injections for tests; by default they are built from the config.
    allow_selection_split:
        Override ``evaluation.allow_selection_split``.
    progress:
        Show a tqdm progress bar (default: ``runtime.progress``).
    fold:
        Fold index used with ``dataset.split.output_pattern`` (default
        ``evaluation.fold``).
    """
    split = (split or str(config.require("evaluation.split"))).strip().lower()
    resolved_fold = int(
        fold
        if fold is not None
        else (config.get("evaluation.fold", config.get("dataset.split.fold", 0)) or 0)
    )
    if resolved_fold < 0:
        raise ConfigError(f"fold must be non-negative, got {resolved_fold}")
    allow = (
        bool(config.get("evaluation.allow_selection_split", False))
        if allow_selection_split is None
        else bool(allow_selection_split)
    )
    selection_split = _guard_reported_split(config, split, allow)

    seed = seed_everything(
        int(config.require("seed")), deterministic=bool(config.get("deterministic", True))
    )
    device = get_device(config.get("device", "auto"))

    # Timing (3 warm-up passes + one measured pass) is only meaningful for a single
    # reference run; every other run is one forward pass per view.
    timing_mode = str(config.get("evaluation.timing", "auto")).strip().lower()
    if timing_mode == "auto":
        timing_enabled = (resolved_fold, seed) == (0, 0)
    elif timing_mode in {"true", "always", "yes", "1"}:
        timing_enabled = True
    elif timing_mode in {"false", "never", "no", "0"}:
        timing_enabled = False
    else:
        raise ConfigError(
            f"evaluation.timing must be 'auto', 'true' or 'false', got {timing_mode!r}"
        )

    dataset = (
        dataset
        if dataset is not None
        else build_dataset(config, split, fold=resolved_fold)
    )
    methods = list(methods) if methods is not None else methods_from_config(config)
    if not methods:
        raise ConfigError("No TTA methods to evaluate (tta.methods is empty)")

    injected_model = model is not None
    if model is None:
        model = build_model(config)
    resolved_checkpoint: Path | None = _resolve_checkpoint(config, resolved_fold, seed)
    checkpoint_meta: dict[str, Any] = {}
    if not injected_model:
        if resolved_checkpoint is None:
            raise ConfigError(
                "model.checkpoint is null; set it to a checkpoint produced by scripts/train.py "
                "or pass an explicit model for a smoke test."
            )
        checkpoint_meta = load_checkpoint(model, resolved_checkpoint, device)
    model = model.to(device)
    model.eval()

    metric_settings = MetricSettings.from_config(config)
    threshold_value = config.get("inference.threshold", 0.5)
    threshold = 0.5 if threshold_value is None else float(threshold_value)
    activation = str(config.get("inference.activation", "sigmoid"))
    amp = bool(config.get("inference.amp", True))
    dataset_name = str(config.get("dataset.name", "dataset"))
    architecture = str(config.get("model.architecture", "model"))
    run_id = make_run_id(config)

    output_path = _resolve_output_dir(config, output_dir)
    # Results are per fold *and* seed so a cross-validation grid never overwrites itself.
    stem = f"{dataset_name}_{split}_fold{resolved_fold}_seed{seed}"
    per_case_csv = output_path / f"{stem}_per_case.csv"
    summary_csv: Path | None = (
        output_path / f"{stem}_summary.csv"
        if bool(config.get("evaluation.write_summary", True))
        else None
    )
    metadata_json = output_path / f"{stem}_run.json"

    warmup_runs = int(config.get("evaluation.warmup_runs", 3)) if timing_enabled else 0
    if warmup_runs < 0:
        raise ConfigError(f"evaluation.warmup_runs must be non-negative, got {warmup_runs}")
    save_maps = bool(config.get("evaluation.save_probability_maps", False))
    map_methods_setting = config.get("evaluation.probability_map_methods", ["baseline", "tta4"])
    if map_methods_setting is None:
        map_methods_setting = []
    if isinstance(map_methods_setting, str):
        map_methods_setting = [map_methods_setting]
    map_methods = tuple(str(name).strip() for name in map_methods_setting)
    unknown_maps = set(map_methods) - {method.name for method in methods}
    if save_maps and unknown_maps:
        raise ConfigError(
            f"evaluation.probability_map_methods names unknown method(s) {sorted(unknown_maps)}; "
            f"available: {sorted(method.name for method in methods)}"
        )
    save_variance = bool(config.get("evaluation.save_variance_maps", False))
    map_setting = config.get("evaluation.probability_map_dir")
    map_base = config.resolve(map_setting) if map_setting else (output_path / "maps")
    # One sub-directory per run so the ensemble can pick up every fold/seed.
    map_dir = map_base / f"fold{resolved_fold}_seed{seed}"
    if save_maps and map_methods:
        map_dir.mkdir(parents=True, exist_ok=True)
    boundary_band_px = int(config.get("evaluation.uncertainty_boundary_band_px", 3))
    emit_penalized = bool(config.get("evaluation.emit_penalized_metrics", True))

    show_progress = bool(config.get("runtime.progress", True)) if progress is None else progress
    total_cases = len(dataset)
    indices = list(range(total_cases if limit is None else min(int(limit), total_cases)))
    created = utc_now()
    groups: set[str] = set()
    rows_written = 0

    iterator: Any = indices
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(indices, desc=f"{dataset_name}/{split}", unit="case")

    with LongResultWriter(per_case_csv) as writer:
        for index in iterator:
            sample = dataset[index]
            image = sample["image"].unsqueeze(0)
            mask = sample.get("mask_native", sample["mask"])
            target = mask[0].numpy() if hasattr(mask[0], "numpy") else mask[0]
            spacing = tuple(float(value) for value in sample.get("spacing", (1.0,) * target.ndim))
            case_id = str(sample["case_id"])
            group_id = str(sample.get("group_id", case_id))
            groups.add(group_id)

            for method in methods:
                # 3 warm-up passes, then one timed pass (CUDA is synchronised so the
                # measured wall-clock includes kernel execution time).  Every other run
                # skips warm-ups and timing, so it is one forward pass per view.
                if timing_enabled:
                    for _ in range(warmup_runs):
                        predict_per_view(
                            model, image, device, method, activation=activation, amp=amp
                        )
                    _synchronize(device)
                    _reset_peak_memory(device)
                    started = time.perf_counter()
                view_probabilities = predict_per_view(
                    model, image, device, method, activation=activation, amp=amp
                )
                if timing_enabled:
                    _synchronize(device)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    peak_memory_mb = _peak_memory_mb(device)
                else:
                    latency_ms = float("nan")
                    peak_memory_mb = float("nan")

                if view_probabilities.shape[1] != 1:
                    raise ConfigError(
                        "The evaluation runner expects single-channel (binary) predictions; "
                        f"got {view_probabilities.shape[1]} channels for method '{method.name}'."
                    )
                # Every view is mapped back onto the native reference grid before
                # aggregation, so quality metrics and uncertainty share a geometry.
                native_views = np.stack(
                    [
                        resample_probability_map(view_map, target.shape)[0]
                        for view_map in view_probabilities
                    ]
                )
                aggregated = aggregate_probabilities(
                    native_views[:, None], method.aggregator
                )[0]
                metrics = compute_case_metrics(
                    aggregated,
                    target,
                    spacing=spacing,
                    threshold=threshold,
                    settings=metric_settings,
                )
                uncertainty = compute_view_uncertainty(
                    native_views[:, None],
                    threshold=threshold,
                    boundary_band_px=boundary_band_px,
                )

                values: dict[str, float] = {
                    name: metrics[name] for name in metric_settings.gt_names
                }
                values.update(
                    {name: uncertainty[name] for name in metric_settings.uncertainty_names}
                )
                if timing_enabled:
                    values["latency_ms"] = latency_ms
                    values["peak_memory_mb"] = peak_memory_mb
                # Sensitivity values: the same distance metrics with an empty
                # prediction penalised by the image diagonal (all cases, no NaN).
                if emit_penalized:
                    diagonal = grid_diagonal_mm(target.shape, spacing)
                    for name in PENALIZED_SOURCE_METRICS:
                        if name in values:
                            value = values[name]
                            values[f"{name}_penalized"] = (
                                diagonal if math.isnan(value) else value
                            )

                provenance = {
                    "run_id": run_id,
                    "created_utc": created,
                    "config_path": str(config.path) if config.path else "",
                    "config_hash": config.fingerprint,
                    "seed": seed,
                    "dataset": dataset_name,
                    "model": architecture,
                    "split": split,
                    "fold": resolved_fold,
                    "method": method.name,
                    "aggregator": method.aggregator,
                    "n_views": method.n_views,
                    "view_names": method.view_names,
                    "threshold": threshold,
                    "spacing_mm": _spacing_text(spacing),
                    "case_id": case_id,
                    "group_id": group_id,
                    "empty_flag": metrics["empty_flag"],
                    "missed_detection": int(metrics["empty_flag"] == "pred_empty"),
                }
                for metric_name, value in values.items():
                    writer.write({**provenance, "metric": metric_name, "value": value})
                rows_written += len(values)

                if save_maps and method.name in map_methods:
                    np.save(
                        map_dir / f"{case_id}__{method.name}__mean.npy",
                        aggregated.astype(np.float16),
                    )
                    if save_variance:
                        np.save(
                            map_dir / f"{case_id}__{method.name}__var.npy",
                            native_views.var(axis=0, ddof=0).astype(np.float16),
                        )

    return _finalise(
        config=config,
        per_case_csv=per_case_csv,
        summary_csv=summary_csv,
        metadata_json=metadata_json,
        run_id=run_id,
        split=split,
        selection_split=selection_split,
        cases_evaluated=len(indices),
        total_cases=total_cases,
        groups=groups,
        rows_written=rows_written,
        methods=methods,
        metric_settings=metric_settings,
        threshold=threshold,
        activation=activation,
        device=device,
        seed=seed,
        checkpoint_path=resolved_checkpoint,
        checkpoint_meta=checkpoint_meta,
        created=created,
        fold=resolved_fold,
        warmup_runs=warmup_runs,
        timing_enabled=timing_enabled,
        save_maps=save_maps and bool(map_methods),
        map_methods=map_methods,
        map_dir=map_dir,
        emit_penalized=emit_penalized,
    )


evaluate = run_evaluation


def _finalise(
    *,
    config: Config,
    per_case_csv: Path,
    summary_csv: Path | None,
    metadata_json: Path,
    run_id: str,
    split: str,
    selection_split: str,
    cases_evaluated: int,
    total_cases: int,
    groups: set[str],
    rows_written: int,
    methods: Sequence[TTAMethod],
    metric_settings: MetricSettings,
    threshold: float,
    activation: str,
    device: torch.device,
    seed: int,
    checkpoint_path: Path | None,
    checkpoint_meta: dict[str, Any],
    created: str,
    fold: int = 0,
    warmup_runs: int = 0,
    timing_enabled: bool = True,
    save_maps: bool = False,
    map_methods: Sequence[str] = (),
    map_dir: Path | None = None,
    emit_penalized: bool = True,
) -> EvaluationResult:
    """Write the summary and provenance files, then return the run description."""
    frame = read_long_csv(per_case_csv)
    if summary_csv is not None:
        summarize_long(frame).to_csv(summary_csv, index=False)

    checkpoint_info: dict[str, Any] = {"path": None, "sha256": None}
    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint_info = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256(checkpoint_path),
            "metadata": checkpoint_meta,
        }

    payload = {
        "run_id": run_id,
        "created_utc": created,
        "config_path": str(config.path) if config.path else None,
        "config_hash": config.fingerprint,
        "config": config.to_dict(),
        "seed": seed,
        "split": split,
        "fold": fold,
        "selection_split": selection_split,
        "cases_evaluated": cases_evaluated,
        "cases_in_split": total_cases,
        "groups_evaluated": len(groups),
        "rows_written": rows_written,
        "threshold": threshold,
        "activation": activation,
        "timing": {
            "enabled": bool(timing_enabled),
            "warmup_runs": warmup_runs,
            "cuda_synchronize": device.type == "cuda",
            "metrics": list(TIMING_METRICS) if timing_enabled else [],
        },
        "probability_maps": {
            "saved": bool(save_maps),
            "dir": str(map_dir) if save_maps and map_dir is not None else None,
            "dtype": "float16" if save_maps else None,
            "methods": list(map_methods) if save_maps else [],
        },
        "metrics": asdict(metric_settings),
        "normalization": config.get("dataset.normalization"),
        "penalized_metrics": {
            "emitted": emit_penalized,
            "source_metrics": list(PENALIZED_SOURCE_METRICS),
        },
        "methods": [
            {
                "name": method.name,
                "aggregator": method.aggregator,
                "n_views": method.n_views,
                "views": [view.name for view in method.views],
            }
            for method in methods
        ],
        "checkpoint": checkpoint_info,
        "device": describe_device(device),
        "environment": environment_info(device),
        "git_revision": git_revision(),
        "outputs": {
            "per_case_csv": str(per_case_csv),
            "summary_csv": str(summary_csv) if summary_csv else None,
        },
    }
    write_json(metadata_json, payload)

    print(
        f"[evaluate] {run_id}: {cases_evaluated}/{total_cases} cases, "
        f"{len(methods)} method(s), {rows_written} rows -> {per_case_csv}"
    )
    return EvaluationResult(
        run_id=run_id,
        split=split,
        fold=fold,
        seed=seed,
        n_cases=cases_evaluated,
        methods=tuple(method.name for method in methods),
        per_case_csv=per_case_csv,
        summary_csv=summary_csv,
        metadata_json=metadata_json,
    )



"""Test-time augmentation inference.

Each view is applied to the input, the model is run on the transformed input, and
the prediction is mapped back with the view's *derived* inverse before
aggregation.  Aggregation therefore happens in the original image geometry, which
is the whole point of a geometric TTA ensemble.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from melba.config import ConfigError
from melba.models.dropout import disable_mc_dropout, enable_mc_dropout
from melba.transforms import GeometricView, build_views, method_specs_from_config
from melba.utils.device import amp_is_enabled

__all__ = [
    "AGGREGATORS",
    "DEFAULT_MC_SAMPLES",
    "TTAMethod",
    "aggregate_probabilities",
    "logits_to_probabilities",
    "methods_from_config",
    "predict_per_view",
    "predict_probabilities",
    "predict_probabilities_for_methods",
]

AGGREGATORS: tuple[str, ...] = ("mean", "median", "max")

#: default number of stochastic samples for ``mcdropout`` / ``mcdropout_T``
DEFAULT_MC_SAMPLES = 8

#: method names that request Monte-Carlo dropout sampling instead of geometric views
_MC_NAME = re.compile(r"^mcdropout(?:_(\d+))?$", re.IGNORECASE)


@dataclass(frozen=True)
class TTAMethod:
    """A named TTA configuration: one or more views and an aggregator.

    ``stochastic=True`` marks a Monte-Carlo dropout method: the views are identity
    transforms and the forward passes differ only because dropout is switched on at
    inference (see :mod:`melba.models.dropout`).  The per-sample probability maps go
    through exactly the same aggregation and uncertainty code as geometric views.
    """

    name: str
    views: tuple[GeometricView, ...]
    aggregator: str = "mean"
    stochastic: bool = False

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError(f"TTA method {self.name!r} needs at least one view")
        if self.aggregator not in AGGREGATORS:
            raise ValueError(
                f"Unknown aggregator {self.aggregator!r}; expected one of {list(AGGREGATORS)}"
            )
        if self.stochastic and len(self.views) < 2:
            raise ValueError(
                f"Stochastic method {self.name!r} needs at least 2 samples to estimate a spread"
            )

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def view_names(self) -> str:
        return "|".join(view.name for view in self.views)


def _method_entries(config: Any) -> list[dict[str, Any]]:
    """``tta.methods`` plus ``tta.extra_methods`` (names or full specs)."""
    from collections.abc import Mapping

    entries: list[dict[str, Any]] = [dict(spec) for spec in method_specs_from_config(config)]
    for extra in config.get("tta.extra_methods") or []:
        entries.append(dict(extra) if isinstance(extra, Mapping) else {"name": str(extra)})
    return entries


def methods_from_config(config: Any) -> list[TTAMethod]:
    """Build every TTA method declared under ``tta.methods``/``tta.extra_methods``.

    A method named ``mcdropout`` or ``mcdropout_<T>`` without explicit ``views`` is
    expanded into ``T`` identity views with ``stochastic: true`` (default T = 8), so the
    MC-dropout samples are aggregated and scored exactly like geometric views.
    """
    default_aggregator = str(config.get("tta.aggregator", "mean"))
    methods: list[TTAMethod] = []
    for spec in _method_entries(config):
        name = str(spec.get("name") or "").strip()
        if not name:
            raise ConfigError("Every entry of 'tta.methods' needs a 'name'")
        aggregator = str(spec.get("aggregator", default_aggregator)).strip().lower()
        view_specs = list(spec.get("views") or [])
        stochastic = bool(spec.get("stochastic", False))
        match = _MC_NAME.match(name)
        if match:
            stochastic = True
            if not view_specs:
                samples = int(match.group(1) or spec.get("n_samples", DEFAULT_MC_SAMPLES))
                if samples < 2:
                    raise ConfigError(
                        f"TTA method {name!r} needs at least 2 MC samples, got {samples}"
                    )
                view_specs = [
                    {"name": f"mc_sample_{index + 1}", "ops": []} for index in range(samples)
                ]
        views = build_views(view_specs)
        methods.append(
            TTAMethod(name=name, views=tuple(views), aggregator=aggregator, stochastic=stochastic)
        )
    names = [method.name for method in methods]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"Duplicate TTA method name(s): {duplicates}")
    return methods


def logits_to_probabilities(logits: torch.Tensor, activation: str = "sigmoid") -> torch.Tensor:
    """Convert model output to probabilities.

    ``sigmoid`` is used for single-channel (binary) output and ``softmax`` for
    multi-channel output.  ``none`` passes the tensor through unchanged (models
    that already return probabilities).
    """
    activation = (activation or "sigmoid").strip().lower()
    if activation == "sigmoid":
        return torch.sigmoid(logits)
    if activation == "softmax":
        return torch.softmax(logits, dim=1)
    if activation in {"none", "identity"}:
        return logits
    raise ConfigError(f"Unsupported activation {activation!r}; use sigmoid, softmax or none")


def aggregate_probabilities(stacked: np.ndarray, aggregator: str = "mean") -> np.ndarray:
    """Aggregate a ``(n_views, C, *spatial)`` probability stack."""
    if aggregator == "mean":
        return stacked.mean(axis=0)
    if aggregator == "median":
        return np.median(stacked, axis=0)
    if aggregator == "max":
        return stacked.max(axis=0)
    raise ValueError(f"Unknown aggregator {aggregator!r}; expected one of {list(AGGREGATORS)}")


def predict_per_view(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    method: TTAMethod,
    *,
    activation: str = "sigmoid",
    amp: bool = True,
) -> np.ndarray:
    """Return the *individual* view probabilities of a TTA method.

    Parameters
    ----------
    model:
        Segmentation model returning logits of shape ``(1, C, *spatial)``.
    image:
        Pre-processed input of shape ``(1, C, *spatial)`` on any device.
    device:
        Compute device (from :func:`melba.utils.device.get_device`).
    method:
        TTA method whose views are applied.
    activation:
        ``sigmoid`` (binary), ``softmax`` (multi-class) or ``none``.
    amp:
        Use CUDA automatic mixed precision when available.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_views, C, *spatial)`` in the *original* image geometry,
        one slice per view.  Used for the per-view uncertainty descriptors.
    """
    if image.ndim < 4:
        raise ValueError(
            "Expected a batched tensor (1, C, *spatial) with at least two spatial dims, "
            f"got shape {tuple(image.shape)}"
        )
    if image.shape[0] != 1:
        raise ValueError("predict_probabilities expects batch size 1 to keep memory bounded")

    model.eval()
    original_shape = tuple(image.shape[1:])  # (C, *spatial) before any view
    spatial_shape = tuple(image.shape[2:])
    use_amp = amp_is_enabled(amp, device)
    predictions: list[np.ndarray] = []
    stochastic = bool(getattr(method, "stochastic", False))
    if stochastic:
        # Dropout on, BatchNorm untouched: the model stays in eval() mode, so the BN
        # running statistics are used and never updated.
        enable_mc_dropout(model)
    try:
        with torch.inference_mode():
            for view in method.views:
                transformed = view.apply(image).to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(transformed)
                if logits.ndim != transformed.ndim or logits.shape[0] != transformed.shape[0] or tuple(
                    logits.shape[2:]
                ) != tuple(transformed.shape[2:]):
                    raise RuntimeError(
                        f"Model output shape {tuple(logits.shape)} is incompatible with its input "
                        f"{tuple(transformed.shape)} for view {view.name!r}: the batch/spatial "
                        "dimensions must match (channels may differ)."
                    )
                probabilities = logits_to_probabilities(logits, activation)
                restored = view.invert(probabilities[0].float(), original_shape)
                if tuple(restored.shape[1:]) != spatial_shape:
                    raise RuntimeError(
                        f"Inverse of view {view.name!r} produced spatial shape "
                        f"{tuple(restored.shape[1:])}, expected {spatial_shape}"
                    )
                predictions.append(restored.detach().cpu().numpy())
    finally:
        if stochastic:
            disable_mc_dropout(model)

    return np.stack(predictions, axis=0).astype(np.float32)


def predict_probabilities(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    method: TTAMethod,
    *,
    activation: str = "sigmoid",
    amp: bool = True,
) -> np.ndarray:
    """Run a single case through a TTA method and return the aggregated probabilities.

    Returns
    -------
    numpy.ndarray
        Probabilities of shape ``(C, *spatial)`` in the *original* image geometry.
    """
    stacked = predict_per_view(
        model, image, device, method, activation=activation, amp=amp
    )
    return aggregate_probabilities(stacked, method.aggregator).astype(np.float32)


def predict_probabilities_for_methods(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    methods: Sequence[TTAMethod],
    *,
    activation: str = "sigmoid",
    amp: bool = True,
) -> dict[str, np.ndarray]:
    """Convenience wrapper: probabilities for every method of a config."""
    return {
        method.name: predict_probabilities(
            model, image, device, method, activation=activation, amp=amp
        )
        for method in methods
    }


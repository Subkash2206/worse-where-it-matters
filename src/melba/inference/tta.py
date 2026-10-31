"""Test-time augmentation inference.

Each view is applied to the input, the model is run on the transformed input, and
the prediction is mapped back with the view's *derived* inverse before
aggregation.  Aggregation therefore happens in the original image geometry, which
is the whole point of a geometric TTA ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from melba.config import ConfigError
from melba.transforms import GeometricView, build_views, method_specs_from_config
from melba.utils.device import amp_is_enabled

__all__ = [
    "AGGREGATORS",
    "TTAMethod",
    "aggregate_probabilities",
    "logits_to_probabilities",
    "methods_from_config",
    "predict_per_view",
    "predict_probabilities",
    "predict_probabilities_for_methods",
]

AGGREGATORS: tuple[str, ...] = ("mean", "median", "max")


@dataclass(frozen=True)
class TTAMethod:
    """A named TTA configuration: one or more views and an aggregator."""

    name: str
    views: tuple[GeometricView, ...]
    aggregator: str = "mean"

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError(f"TTA method {self.name!r} needs at least one view")
        if self.aggregator not in AGGREGATORS:
            raise ValueError(
                f"Unknown aggregator {self.aggregator!r}; expected one of {list(AGGREGATORS)}"
            )

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def view_names(self) -> str:
        return "|".join(view.name for view in self.views)


def methods_from_config(config: Any) -> list[TTAMethod]:
    """Build every TTA method declared under ``tta.methods`` in a config."""
    default_aggregator = str(config.get("tta.aggregator", "mean"))
    methods: list[TTAMethod] = []
    for spec in method_specs_from_config(config):
        name = str(spec.get("name") or "").strip()
        if not name:
            raise ConfigError("Every entry of 'tta.methods' needs a 'name'")
        aggregator = str(spec.get("aggregator", default_aggregator)).strip().lower()
        views = build_views(spec.get("views") or [])
        methods.append(TTAMethod(name=name, views=tuple(views), aggregator=aggregator))
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


"""Photometric (intensity) TTA transforms: no inverse, no geometry change.

A *geometric* view is inverted after inference because the prediction lives in the
transformed geometry.  An intensity view needs the opposite treatment: the transform acts
only on the network input, and the probability map it produces is already in the original
geometry, so ``invert`` must be the identity.  That is exactly the interface
:class:`melba.transforms.geometric.InvertibleTransform` describes, so intensity ops compose
with geometric ones in the same view specification (``geo4_int4`` = a geometric view times an
intensity change).

All transforms operate on the **normalized** tensor the model receives (z-scores for CVC and
BraTS alike) and per channel, so a four-modality BraTS volume gets the same factor on each
modality - as required for intensity TTA on multi-modal input:

* ``brightness`` - ``x * factor``;
* ``contrast`` - ``(x - mean_c) * factor + mean_c`` per channel;
* ``gamma`` - min-max the channel to [0, 1], raise to ``gamma`` and map back, which keeps the
  operation monotone even though the input has negative values;
* ``gaussian_noise`` - ``x + N(0, sigma * (max_c - min_c))`` with a **fixed seed**, so the
  view is deterministic and reproducible.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .geometric import InvertibleTransform

__all__ = ["Brightness", "Contrast", "Gamma", "GaussianNoise"]


class _IntensityTransform(InvertibleTransform):
    """Intensity change on the input, identity on the prediction."""

    def invert(self, tensor: torch.Tensor, orig_shape: Sequence[int] | None = None) -> torch.Tensor:
        # nothing to undo: the probabilities are already in the original geometry
        return tensor

    @staticmethod
    def _per_channel_range(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel (min, max) over the spatial axes, keeping dims for broadcasting."""
        dims = tuple(range(2, tensor.ndim))
        if not dims:  # pragma: no cover - a bare (C,) tensor is not a valid image
            return tensor.min(), tensor.max()
        return (
            tensor.amin(dim=dims, keepdim=True),
            tensor.amax(dim=dims, keepdim=True),
        )


class Brightness(_IntensityTransform):
    """Multiplicative brightness change (``factor > 0``)."""

    def __init__(self, factor: float = 1.0) -> None:
        factor = float(factor)
        if not factor > 0:
            raise ValueError(f"brightness factor must be positive, got {factor}")
        self.factor = factor
        self.name = f"brightness_x{factor:g}"

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.factor


class Contrast(_IntensityTransform):
    """Contrast change around the per-channel mean."""

    def __init__(self, factor: float = 1.0) -> None:
        factor = float(factor)
        if not factor > 0:
            raise ValueError(f"contrast factor must be positive, got {factor}")
        self.factor = factor
        self.name = f"contrast_x{factor:g}"

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(2, tensor.ndim))
        mean = tensor.mean(dim=dims, keepdim=True) if dims else tensor.mean()
        return (tensor - mean) * self.factor + mean


class Gamma(_IntensityTransform):
    """Gamma change on the min-max normalised channel (monotone, defined for z-scores)."""

    def __init__(self, gamma: float = 1.0) -> None:
        gamma = float(gamma)
        if not gamma > 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = gamma
        self.name = f"gamma_{gamma:g}"

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        low, high = self._per_channel_range(tensor)
        span = (high - low).clamp_min(torch.finfo(tensor.dtype).eps)
        unit = (tensor - low) / span
        return unit.clamp(0.0, 1.0).pow(self.gamma) * span + low


class GaussianNoise(_IntensityTransform):
    """Additive Gaussian noise, ``sigma`` expressed as a fraction of the channel range."""

    def __init__(self, sigma: float = 0.02, *, seed: int = 0) -> None:
        sigma = float(sigma)
        if sigma < 0:
            raise ValueError(f"noise sigma must be non-negative, got {sigma}")
        self.sigma = sigma
        self.seed = int(seed)
        self.name = f"noise_{sigma:g}"

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.sigma == 0:
            return tensor
        low, high = self._per_channel_range(tensor)
        scale = (high - low).clamp_min(torch.finfo(tensor.dtype).eps) * self.sigma
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
        return tensor + noise.to(device=tensor.device, dtype=tensor.dtype) * scale

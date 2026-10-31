"""Invertible geometric transforms used by test-time augmentation.

Every transform exposes ``apply`` and ``invert``; the inverse of a composition is
built by reversing the list, so inverses are never hand-written per view (the
previous code base had a separate, manually maintained inverse table).

Transforms operate on tensors of shape ``(..., *spatial)``, i.e. any number of
leading non-spatial dimensions (channels, batch) followed by the spatial axes.
Dims follow PyTorch conventions and may be negative (``-1`` is the last axis).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn.functional as functional

__all__ = [
    "Compose",
    "Flip",
    "Identity",
    "InvertibleTransform",
    "Rot90",
    "Transpose",
    "Zoom",
]


class InvertibleTransform(ABC):
    """Base class for deterministic, invertible geometric transforms."""

    #: short identifier used in logs and result records
    name: str = "transform"

    @abstractmethod
    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """Map a tensor to the transformed geometry."""

    @abstractmethod
    def invert(self, tensor: torch.Tensor, orig_shape: Sequence[int] | None = None) -> torch.Tensor:
        """Map a tensor back to the original geometry.

        ``orig_shape`` is required by transforms that change the spatial shape
        (``Zoom``) and is ignored by the others.
        """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.apply(tensor)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"


class Identity(InvertibleTransform):
    """No-op transform (the TTA baseline view)."""

    name = "identity"

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def invert(self, tensor: torch.Tensor, orig_shape: Sequence[int] | None = None) -> torch.Tensor:
        return tensor


class Flip(InvertibleTransform):
    """Flip along one or more axes; its own inverse."""

    def __init__(self, dims: Sequence[int]) -> None:
        dims = tuple(int(dim) for dim in dims)
        if not dims:
            raise ValueError("Flip requires at least one dimension")
        if len(set(dims)) != len(dims):
            raise ValueError(f"Flip dimensions must be unique, got {dims}")
        self.dims = dims
        self.name = "flip" + "".join(f"_{dim}" for dim in dims)

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.flip(tensor, dims=self.dims)

    def invert(self, tensor: torch.Tensor, orig_shape: Sequence[int] | None = None) -> torch.Tensor:
        return torch.flip(tensor, dims=self.dims)


class Rot90(InvertibleTransform):
    """Rotation by ``k * 90`` degrees in the plane spanned by ``dims``."""

    def __init__(self, k: int = 1, dims: Sequence[int] = (-2, -1)) -> None:
        dims = tuple(int(dim) for dim in dims)
        if len(dims) != 2:
            raise ValueError(f"Rot90 needs exactly two dims, got {dims}")
        if dims[0] == dims[1]:
            raise ValueError(f"Rot90 dims must differ, got {dims}")
        if int(k) % 4 == 0:
            raise ValueError("Rot90 with k % 4 == 0 is the identity; use Identity() instead")
        self.k = int(k)
        self.dims = dims
        self.name = f"rot90_k{self.k}"

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.rot90(tensor, self.k, dims=self.dims)

    def invert(self, tensor: torch.Tensor, orig_shape: Sequence[int] | None = None) -> torch.Tensor:
        return torch.rot90(tensor, -self.k, dims=self.dims)


class Transpose(InvertibleTransform):
    """Swap two axes; its own inverse."""

    def __init__(self, dims: Sequence[int]) -> None:
        dims = tuple(int(dim) for dim in dims)
        if len(dims) != 2 or dims[0] == dims[1]:
            raise ValueError(f"Transpose needs two distinct dims, got {dims}")
        self.dims = dims
        self.name = f"transpose_{dims[0]}_{dims[1]}"

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.transpose(tensor, self.dims[0], self.dims[1])

    def invert(self, tensor: torch.Tensor, orig_shape: Sequence[int] | None = None) -> torch.Tensor:
        return torch.transpose(tensor, self.dims[0], self.dims[1])


class Zoom(InvertibleTransform):
    """Anisotropic resampling around the image centre (multi-scale TTA).

    ``apply`` resizes the selected axes by ``factors``.  ``invert`` resamples back
    to the original grid; because resampling is not information preserving, the
    inverse is only exact for lossless cases (e.g. nearest-neighbour upsampling
    of a piecewise-constant mask), which is what the round-trip tests check.
    """

    _INTERPOLATION_MODES = ("nearest", "bilinear", "bicubic", "trilinear")

    def __init__(
        self,
        factors: Sequence[float],
        dims: Sequence[int],
        *,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ) -> None:
        factors = tuple(float(factor) for factor in factors)
        dims = tuple(int(dim) for dim in dims)
        if len(factors) != len(dims):
            raise ValueError(f"Zoom needs one factor per dim, got {factors} for {dims}")
        if not dims:
            raise ValueError("Zoom requires at least one dimension")
        if len(set(dims)) != len(dims):
            raise ValueError(f"Zoom dims must be unique, got {dims}")
        if any(factor <= 0 for factor in factors):
            raise ValueError(f"Zoom factors must be positive, got {factors}")
        if mode not in self._INTERPOLATION_MODES:
            raise ValueError(f"Unsupported interpolation mode {mode!r}")
        self.factors = factors
        self.dims = dims
        self.mode = mode
        self.align_corners = bool(align_corners)
        self.antialias = bool(antialias)
        self.name = "zoom_" + "_".join(f"{factor:g}" for factor in factors)

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        sizes = self._scaled_sizes(tensor.shape, self.factors)
        return self._resize(tensor, sizes)

    def invert(self, tensor: torch.Tensor, orig_shape: Sequence[int] | None = None) -> torch.Tensor:
        if orig_shape is None:
            raise ValueError("Zoom.invert requires the original tensor shape")
        ndim = tensor.ndim
        if len(orig_shape) != ndim:
            raise ValueError(
                f"orig_shape has {len(orig_shape)} entries but tensor has {ndim} dims"
            )
        dims = self._normalized_dims(ndim)
        sizes = [int(orig_shape[dim]) for dim in dims]
        return self._resize(tensor, sizes)

    # -- internals ------------------------------------------------------------
    def _normalized_dims(self, ndim: int) -> tuple[int, ...]:
        dims = tuple(dim % ndim for dim in self.dims)
        return tuple(sorted(dims))

    def _scaled_sizes(self, shape: Sequence[int], factors: Sequence[float]) -> list[int]:
        pairs = sorted(
            ((dim % len(shape), float(factor)) for dim, factor in zip(self.dims, factors)),
            key=lambda pair: pair[0],
        )
        return [max(1, int(round(int(shape[dim]) * factor))) for dim, factor in pairs]

    def _resize(self, tensor: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
        ndim = tensor.ndim
        dims = self._normalized_dims(ndim)
        others = tuple(dim for dim in range(ndim) if dim not in dims)
        permutation = others + dims
        permuted = tensor.permute(*permutation)
        lead_shape = tuple(permuted.shape[: len(others)])
        spatial_shape = tuple(permuted.shape[len(others) :])
        flat = permuted.reshape(-1, 1, *spatial_shape)

        kwargs: dict[str, object] = {}
        if self.mode in ("bilinear", "bicubic", "trilinear"):
            kwargs["align_corners"] = self.align_corners
            if self.mode in ("bilinear", "bicubic"):
                kwargs["antialias"] = self.antialias
        resized = functional.interpolate(flat, size=tuple(int(size) for size in sizes), mode=self.mode, **kwargs)
        restored = resized.reshape(*lead_shape, *sizes)

        inverse = [0] * ndim
        for position, dim in enumerate(permutation):
            inverse[dim] = position
        return restored.permute(*inverse)


class Compose(InvertibleTransform):
    """Apply transforms in order; invert them in reverse order."""

    def __init__(self, transforms: Sequence[InvertibleTransform], name: str | None = None) -> None:
        for transform in transforms:
            if not isinstance(transform, InvertibleTransform):
                raise TypeError(f"Compose expects InvertibleTransform instances, got {type(transform)!r}")
        self.transforms = tuple(transforms)
        self.name = name or "compose(" + "|".join(transform.name for transform in self.transforms) + ")"

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            tensor = transform.apply(tensor)
        return tensor

    def invert(self, tensor: torch.Tensor, orig_shape: Sequence[int] | None = None) -> torch.Tensor:
        for transform in reversed(self.transforms):
            tensor = transform.invert(tensor, orig_shape)
        return tensor


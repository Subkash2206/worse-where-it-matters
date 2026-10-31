"""Input normalization for pretrained encoders.

Images are loaded into ``[0, 1]`` and then normalized with the encoder's
preprocessing statistics (``segmentation_models_pytorch.encoders.
get_preprocessing_params``).  For an ImageNet ResNet-50 encoder that is
``mean=(0.485, 0.456, 0.406)`` / ``std=(0.229, 0.224, 0.225)``; without it the
pretrained filters see a differently scaled input and the fine-tuning benefit of
ImageNet weights is largely lost.

The statistics are never hardcoded in the data pipeline: the kind is selected in
YAML (``dataset.normalization.kind``) and, for ``imagenet``, the mean/std are read
from the encoder configuration of the same config, with explicit
``mean``/``std`` overrides allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from melba.config import Config, ConfigError

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NORMALIZATION_KINDS",
    "NormalizationSpec",
]

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
NORMALIZATION_KINDS: tuple[str, ...] = ("imagenet", "custom", "none")


def _encoder_preprocessing_params(config: Config) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mean/std declared by smp for the configured encoder (ImageNet fallback)."""
    encoder = str(config.get("model.encoder", "resnet50"))
    weights = config.get("model.encoder_weights")
    pretrained = None if weights is None else str(weights)
    try:
        import segmentation_models_pytorch as smp

        params = smp.encoders.get_preprocessing_params(encoder, pretrained=pretrained)
    except Exception:  # pragma: no cover - unknown encoder/offline lookup
        params = {}
    mean = params.get("mean") or IMAGENET_MEAN
    std = params.get("std") or IMAGENET_STD
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


@dataclass(frozen=True)
class NormalizationSpec:
    """Channel-wise normalization applied to ``[0, 1]`` images."""

    kind: str = "imagenet"
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in NORMALIZATION_KINDS:
            raise ValueError(
                f"Unknown normalization kind {self.kind!r}; expected one of "
                f"{list(NORMALIZATION_KINDS)}"
            )
        if self.kind == "none":
            return
        if self.mean is None or self.std is None:
            raise ValueError(f"Normalization kind {self.kind!r} needs mean and std")
        mean = tuple(float(value) for value in self.mean)
        std = tuple(float(value) for value in self.std)
        if len(mean) != len(std) or not mean:
            raise ValueError(f"mean and std must have the same length, got {mean} and {std}")
        if any(not np.isfinite(value) for value in mean):
            raise ValueError(f"mean must be finite, got {mean}")
        if any(not np.isfinite(value) or value <= 0 for value in std):
            raise ValueError(f"std must be finite and positive, got {std}")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @property
    def enabled(self) -> bool:
        return self.kind != "none"

    @classmethod
    def from_config(cls, config: Config) -> "NormalizationSpec":
        section: Mapping[str, Any] = config.get("dataset.normalization") or {}
        kind = str(section.get("kind", "imagenet")).strip().lower()
        if kind == "none":
            return cls(kind="none")
        if kind == "custom":
            return cls(
                kind="custom",
                mean=tuple(float(value) for value in (section.get("mean") or [])),
                std=tuple(float(value) for value in (section.get("std") or [])),
            )
        if kind != "imagenet":
            raise ConfigError(
                f"Unknown dataset.normalization.kind {kind!r}; expected one of "
                f"{list(NORMALIZATION_KINDS)}"
            )
        default_mean, default_std = _encoder_preprocessing_params(config)
        mean = section.get("mean") or default_mean
        std = section.get("std") or default_std
        return cls(kind="imagenet", mean=mean, std=std)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Normalize a ``(C, H, W)`` float array that is already in ``[0, 1]``."""
        array = np.asarray(image, dtype=np.float32)
        if not self.enabled:
            return array
        if array.ndim != 3:
            raise ValueError(f"Expected a (C, H, W) image, got shape {array.shape}")
        if array.shape[0] != len(self.mean or ()):  # type: ignore[arg-type]
            raise ValueError(
                f"Image has {array.shape[0]} channels but normalization has "
                f"{len(self.mean or ())} statistics"
            )
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(self.std, dtype=np.float32).reshape(-1, 1, 1)
        return (array - mean) / std

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "mean": list(self.mean) if self.mean is not None else None,
            "std": list(self.std) if self.std is not None else None,
        }

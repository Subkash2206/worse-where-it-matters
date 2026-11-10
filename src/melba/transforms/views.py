"""Config-driven TTA view construction.

A *view* is a named, invertible geometric transform described in YAML::

    views:
      - name: hflip_rot90
        ops:
          - {type: flip, axes: [w]}
          - {type: rot90, k: 1, axes: [h, w]}

Axes are named after the spatial dimensions of a ``(channels, ...)`` tensor:
``w`` is the last axis, ``h`` the one before it and ``d``/``z`` the one before that.
Negative integers are also accepted.  Because the inverse of a composition is
derived automatically, adding a view to a config can never desynchronise an
inverse.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from melba.config import ConfigError

from .geometric import Compose, Flip, Identity, InvertibleTransform, Rot90, Transpose, Zoom
from .intensity import Brightness, Contrast, Gamma, GaussianNoise

__all__ = [
    "AXIS_DIMS",
    "GeometricView",
    "axis_to_dim",
    "build_transform",
    "build_view",
    "build_views",
    "method_specs_from_config",
]

#: named spatial axes -> PyTorch dims (independent of 2D/3D because they are
#: counted from the end of the tensor)
AXIS_DIMS: dict[str, int] = {"w": -1, "h": -2, "d": -3, "z": -3}

_SUPPORTED_TYPES = (
    "identity",
    "flip",
    "rot90",
    "transpose",
    "zoom",
    "brightness",
    "contrast",
    "gamma",
    "gaussian_noise",
)


def axis_to_dim(axis: Any) -> int:
    """Convert a YAML axis spec (``"h"``, ``"w"``, ``"d"`` or an int) to a dim."""
    if isinstance(axis, bool):
        raise ConfigError(f"Invalid axis {axis!r}; use 'h', 'w', 'd' or an integer dim")
    if isinstance(axis, int):
        return axis
    key = str(axis).strip().lower()
    if key in AXIS_DIMS:
        return AXIS_DIMS[key]
    raise ConfigError(f"Unknown axis {axis!r}; expected one of {sorted(AXIS_DIMS)} or an integer")


def _require_axes(options: Mapping[str, Any], *, expected: int | None = None) -> list[int]:
    axes = options.get("axes")
    if axes is None:
        raise ConfigError("Transform spec is missing 'axes'")
    if isinstance(axes, (str, int)):
        axes = [axes]
    dims = [axis_to_dim(axis) for axis in axes]
    if expected is not None and len(dims) != expected:
        raise ConfigError(f"Transform needs exactly {expected} axes, got {len(dims)}")
    if not dims:
        raise ConfigError("Transform spec has an empty 'axes' list")
    return dims


def _reject_unknown_options(options: Mapping[str, Any], allowed: Sequence[str], kind: str) -> None:
    unknown = sorted(set(options) - set(allowed))
    if unknown:
        raise ConfigError(f"Unknown option(s) {unknown} for transform '{kind}'")


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def build_transform(spec: Mapping[str, Any]) -> InvertibleTransform:
    """Instantiate a single transform from its YAML specification."""
    if not isinstance(spec, Mapping):
        raise ConfigError(f"Transform spec must be a mapping, got {type(spec).__name__}")
    options = dict(spec)
    kind = str(options.pop("type", "") or "").strip().lower()
    if not kind:
        raise ConfigError("Transform spec is missing 'type'")
    if kind not in _SUPPORTED_TYPES:
        raise ConfigError(f"Unknown transform type {kind!r}; supported: {list(_SUPPORTED_TYPES)}")

    if kind == "identity":
        _reject_unknown_options(options, (), kind)
        return Identity()

    if kind == "flip":
        dims = _require_axes(options)
        _reject_unknown_options(options, ("axes",), kind)
        return Flip(dims)

    if kind == "rot90":
        dims = _require_axes(options, expected=2)
        k = int(options.pop("k", 1))
        _reject_unknown_options(options, ("axes", "k"), kind)
        return Rot90(k=k, dims=dims)

    if kind == "transpose":
        dims = _require_axes(options, expected=2)
        _reject_unknown_options(options, ("axes",), kind)
        return Transpose(dims)

    if kind == "brightness":
        factor = float(options.pop("factor", 1.0))
        _reject_unknown_options(options, ("factor",), kind)
        return Brightness(factor)

    if kind == "contrast":
        factor = float(options.pop("factor", 1.0))
        _reject_unknown_options(options, ("factor",), kind)
        return Contrast(factor)

    if kind == "gamma":
        value = float(options.pop("gamma", 1.0))
        _reject_unknown_options(options, ("gamma",), kind)
        return Gamma(value)

    if kind == "gaussian_noise":
        sigma = float(options.pop("sigma", 0.02))
        seed = int(options.pop("seed", 0))
        _reject_unknown_options(options, ("sigma", "seed"), kind)
        return GaussianNoise(sigma, seed=seed)

    # zoom / multi-scale
    dims = _require_axes(options)
    factors = options.pop("factors", options.pop("factor", None))
    if factors is None:
        raise ConfigError("Zoom spec is missing 'factors'")
    if _is_scalar(factors):
        factors = [float(factors)] * len(dims)
    factors = list(factors)
    if len(factors) != len(dims):
        raise ConfigError(f"Zoom needs {len(dims)} factors, got {len(factors)}")
    mode = str(options.pop("mode", "bilinear")).strip().lower()
    align_corners = bool(options.pop("align_corners", False))
    antialias = bool(options.pop("antialias", True))
    _reject_unknown_options(
        options, ("axes", "factors", "factor", "mode", "align_corners", "antialias"), kind
    )
    return Zoom(factors, dims, mode=mode, align_corners=align_corners, antialias=antialias)


@dataclass(frozen=True)
class GeometricView:
    """A named, invertible view used to build a TTA prediction."""

    name: str
    transform: InvertibleTransform

    def apply(self, tensor):
        return self.transform.apply(tensor)

    def invert(self, tensor, orig_shape=None):
        return self.transform.invert(tensor, orig_shape)


def build_view(spec: Mapping[str, Any]) -> GeometricView:
    """Build one view (name + ordered list of ``ops``) from YAML."""
    if not isinstance(spec, Mapping):
        raise ConfigError(f"View spec must be a mapping, got {type(spec).__name__}")
    ops = spec.get("ops") or []
    if not isinstance(ops, Sequence) or isinstance(ops, (str, bytes)):
        raise ConfigError(f"'ops' of view {spec.get('name')!r} must be a list")
    unknown = sorted(set(spec) - {"name", "ops"})
    if unknown:
        raise ConfigError(f"Unknown option(s) {unknown} in view {spec.get('name')!r}")

    transforms = [build_transform(op) for op in ops]
    transform: InvertibleTransform = Compose(transforms) if transforms else Identity()
    name = str(spec.get("name") or transform.name)
    return GeometricView(name=name, transform=transform)


def build_views(specs: Sequence[Mapping[str, Any]]) -> list[GeometricView]:
    """Build a list of views, rejecting duplicates and empty specifications."""
    if not specs:
        raise ConfigError("A TTA method needs at least one view")
    views = [build_view(spec) for spec in specs]
    names = [view.name for view in views]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"Duplicate view name(s): {duplicates}")
    return views


def method_specs_from_config(config: Any) -> list[Mapping[str, Any]]:
    """Return the ``tta.methods`` list from a MELBA config."""
    methods = config.get("tta.methods")
    if not methods:
        raise ConfigError("Config is missing 'tta.methods'")
    if not isinstance(methods, Sequence):
        raise ConfigError("'tta.methods' must be a list")
    return list(methods)


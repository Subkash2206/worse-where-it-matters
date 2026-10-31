"""YAML configuration loading.

Design rules enforced here (see README):

* every path, seed and hyper-parameter comes from a YAML file in ``configs/``;
* a config can inherit from one or more parents through ``defaults:``;
* ``${dotted.key}`` references are interpolated inside the YAML tree;
* ``load_config`` accepts dotted ``key=value`` overrides for CLI experiments;
* relative paths are resolved against ``paths.root``, which is itself resolved
  relative to the config file that declares it (``..`` -> repository root).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
    "parse_set_args",
]

_MISSING = object()
_REFERENCE = re.compile(r"\$\{([^}]+)\}")
_MAX_INTERPOLATION_PASSES = 8


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as error:  # pragma: no cover - passthrough
        raise ConfigError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(data, Mapping):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return dict(data)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_with_defaults(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(item) for item in stack + (path,))
        raise ConfigError(f"Circular `defaults` chain: {chain}")
    data = _read_yaml(path)
    parents = data.pop("defaults", []) or []
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, Sequence):
        raise ConfigError(f"`defaults` in {path} must be a list of relative paths")

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (path.parent / str(parent)).resolve()
        merged = _deep_merge(merged, _load_with_defaults(parent_path, stack + (path,)))
    return _deep_merge(merged, data)


def _lookup_dotted(data: Mapping[str, Any], dotted: str, *, source: Path) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigError(f"Unknown reference '${{{dotted}}}' in {source}")
        node = node[part]
    return node


def _resolve_references(node: Any, data: Mapping[str, Any], source: Path) -> Any:
    if isinstance(node, Mapping):
        return {key: _resolve_references(value, data, source) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_references(value, data, source) for value in node]
    if isinstance(node, str):
        whole = _REFERENCE.fullmatch(node.strip())
        if whole is not None:
            # A string that is exactly one reference keeps the referenced type.
            return copy.deepcopy(_lookup_dotted(data, whole.group(1), source=source))
        return _REFERENCE.sub(
            lambda match: str(_lookup_dotted(data, match.group(1), source=source)), node
        )
    return node


def _interpolate(data: Mapping[str, Any], source: Path) -> dict[str, Any]:
    resolved: dict[str, Any] = copy.deepcopy(dict(data))
    for _ in range(_MAX_INTERPOLATION_PASSES):
        updated = _resolve_references(resolved, resolved, source)
        if updated == resolved:
            break
        resolved = updated
    else:  # pragma: no cover - only on pathological configs
        raise ConfigError(f"Could not resolve all '${{...}}' references in {source}")
    return resolved


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = [part for part in dotted.split(".") if part]
    if not parts:
        raise ConfigError("Empty override key")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot apply override '{dotted}': '{part}' is not a mapping")
        node = child
    node[parts[-1]] = value


def parse_set_args(pairs: Sequence[str]) -> dict[str, Any]:
    """Parse ``["a.b=1", "c.d=[1, 2]"]`` into ``{"a.b": 1, "c.d": [1, 2]}``.

    Values are parsed with ``yaml.safe_load`` so that booleans, numbers, lists and
    ``null`` behave as they do inside a YAML file.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"Override '{pair}' must have the form dotted.key=value")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"Override '{pair}' has an empty key")
        try:
            overrides[key] = yaml.safe_load(raw)
        except yaml.YAMLError as error:  # pragma: no cover - passthrough
            raise ConfigError(f"Could not parse value in override '{pair}': {error}") from error
    return overrides


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
class Config(Mapping[str, Any]):
    """Read-only, dotted-key view over a resolved configuration tree."""

    def __init__(self, data: Mapping[str, Any], *, path: Path | None = None) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self.path: Path | None = Path(path).resolve() if path is not None else None

    # -- mapping protocol -----------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._lookup(key))

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        source = self.path.name if self.path else "<memory>"
        return f"Config({source}, keys={sorted(self._data)})"

    # -- dotted access --------------------------------------------------------
    def _lookup(self, key: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in str(key).split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"Missing configuration key '{key}'")
                return default
            node = node[part]
        return node

    def get(self, key: str, default: Any = None) -> Any:
        return copy.deepcopy(self._lookup(key, default))

    def require(self, key: str) -> Any:
        """Return a key's value, raising :class:`ConfigError` when it is unset."""
        value = self._lookup(key, None)
        if value is None:
            raise ConfigError(f"Required configuration key '{key}' is missing or null")
        return copy.deepcopy(value)

    def section(self, key: str) -> dict[str, Any]:
        value = self.require(key)
        if not isinstance(value, Mapping):
            raise ConfigError(f"Configuration key '{key}' must be a mapping")
        return copy.deepcopy(dict(value))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- paths ----------------------------------------------------------------
    @property
    def root(self) -> Path:
        """Absolute repository root.

        ``paths.root: auto`` (the default in ``configs/base.yaml``) anchors the
        root at the closest ancestor of this config file that contains a
        ``pyproject.toml``, so configs can live at any nesting depth.  An explicit
        path is used as-is when absolute, or resolved against the config file
        directory when relative.
        """
        raw = self.get("paths.root", "auto")
        anchor = self.path.parent if self.path is not None else Path.cwd()
        if raw is None or str(raw).strip().lower() == "auto":
            for candidate in (anchor, *anchor.parents):
                if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
                    return candidate.resolve()
            return anchor.resolve()

        root = Path(str(raw)).expanduser()
        return root.resolve() if root.is_absolute() else (anchor / root).resolve()

    def resolve(self, value: str | Path) -> Path:
        """Resolve a (possibly relative) path string against :attr:`root`."""
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()

    def require_path(self, key: str, *, must_exist: bool = False) -> Path:
        path = self.resolve(self.require(key))
        if must_exist and not path.exists():
            raise ConfigError(f"Path from '{key}' does not exist: {path}")
        return path

    def ensure_dir(self, key: str) -> Path:
        path = self.resolve(self.require(key))
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- provenance -----------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        """Stable hash of the resolved configuration (used to tag result rows)."""
        payload = json.dumps(self._data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> "Config":
        if not overrides:
            return self
        data = self.to_dict()
        for key, value in overrides.items():
            _set_dotted(data, key, value)
        source = self.path or Path.cwd()
        updated = Config(_interpolate(data, source), path=self.path)
        _validate(updated)
        return updated


def load_config(
    path: str | Path,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
) -> Config:
    """Load a YAML config, resolve ``defaults``/references and apply overrides.

    Parameters
    ----------
    path:
        Path to a YAML file inside ``configs/``.
    overrides:
        Mapping of dotted keys to values, or a sequence of ``"a.b=value"``
        strings (see :func:`parse_set_args`).
    """
    config_path = Path(path).expanduser().resolve()
    data = _load_with_defaults(config_path, ())

    if overrides is not None and not isinstance(overrides, Mapping):
        overrides = parse_set_args(list(overrides))

    # Interpolate first so overrides can reference existing keys, then apply the
    # overrides and interpolate again so overridden values can be referenced.
    data = _interpolate(data, config_path)
    for key, value in (overrides or {}).items():
        _set_dotted(data, key, value)
    data = _interpolate(data, config_path)

    config = Config(data, path=config_path)
    _validate(config)
    return config


def _validate(config: Config) -> None:
    """Fail fast on problems that would otherwise surface much later."""
    seed = config.get("seed")
    if seed is None:
        raise ConfigError("Missing required key 'seed'")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"'seed' must be a non-negative integer, got {seed!r}")

    if "paths.root" not in config:
        raise ConfigError("Missing required key 'paths.root'")
    if not config.root.is_dir():
        raise ConfigError(f"paths.root does not point to a directory: {config.root}")

    if not config.get("experiment.name"):
        raise ConfigError("Missing required key 'experiment.name'")

    fractions = config.get("dataset.split.fractions")
    if fractions is not None:
        missing = {"train", "val", "test"} - set(fractions)
        if missing:
            raise ConfigError(f"dataset.split.fractions is missing {sorted(missing)}")
        total = float(sum(float(value) for value in fractions.values()))
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"dataset.split.fractions must sum to 1.0, got {total}")
        if any(float(value) <= 0 for value in fractions.values()):
            raise ConfigError("dataset.split.fractions values must be positive")



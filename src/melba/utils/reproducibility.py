"""Seed handling.

Every entry point calls :func:`seed_everything` before building data loaders or
models.  The function seeds ``random``, NumPy and PyTorch (CPU *and* every CUDA
device) and switches cuDNN to deterministic mode, which is what makes a run
reproducible on a given machine + library set.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from melba.config import Config

_UINT32_MAX = 2**32 - 1


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    benchmark: bool = False,
) -> int:
    """Seed all random number generators used by the project.

    Parameters
    ----------
    seed:
        Non-negative integer seed.  Values above ``2**32 - 1`` are reduced
        modulo that range for NumPy compatibility.
    deterministic:
        When ``True`` (default) cuDNN is put into deterministic mode and
        ``torch.use_deterministic_algorithms`` is enabled.  Operations without a
        deterministic implementation raise instead of silently diverging; use
        ``warn_only=True`` semantics (already applied here) to keep them usable.
    benchmark:
        cuDNN benchmark mode.  Only allowed together with ``deterministic=False``
        on CUDA, because benchmarking selects algorithms non-deterministically.

    Returns
    -------
    int
        The validated seed, for convenient logging.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed % _UINT32_MAX)
    # Required by cuBLAS for deterministic matrix multiplications (must be set
    # before the CUDA context is created, hence inside this entry point).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed % _UINT32_MAX)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = bool(benchmark) and not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)

    return seed


def seed_from_config(config: "Config", key: str = "seed") -> int:
    """Read and apply the seed declared in a configuration object."""
    return seed_everything(int(config.require(key)), deterministic=bool(config.get("deterministic", True)))


def make_worker_init_fn(seed: int, *, deterministic: bool = True):
    """Return a DataLoader ``worker_init_fn`` that seeds each worker.

    DataLoader workers reseed NumPy/PyTorch on start-up; without this hook the
    augmented samples differ between runs even when the main process is seeded.
    """

    def _init(worker_id: int) -> None:  # pragma: no cover - runs in subprocess
        worker_seed = (int(seed) + worker_id) % _UINT32_MAX
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        torch.use_deterministic_algorithms(deterministic, warn_only=True)

    return _init


def environment_info(device: torch.device | None = None) -> dict[str, Any]:
    """Collect provenance information to be stored next to every result file."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    for name in ("monai", "segmentation_models_pytorch", "scipy", "pandas"):
        try:  # optional at import time of this helper
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover - defensive
            continue
    if device is not None and device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        info["gpu"] = torch.cuda.get_device_name(index)
        info["gpu_total_memory_gb"] = round(
            torch.cuda.get_device_properties(index).total_memory / 1024**3, 2
        )
    return info


def flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into dotted keys (used for provenance JSON)."""
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_mapping(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat

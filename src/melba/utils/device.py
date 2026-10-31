"""Automatic device selection.

The project must run on a single GPU with 16 GB or less, and on CPU.  Nothing in
the code base may hardcode ``"cuda"``; everything goes through :func:`get_device`,
which is driven by the ``device`` key of the YAML config (``auto`` by default).
"""

from __future__ import annotations

from typing import Any

import torch


def get_device(preference: str | torch.device | None = "auto") -> torch.device:
    """Resolve the compute device.

    Parameters
    ----------
    preference:
        ``"auto"`` (CUDA when available, else CPU), ``"cpu"``, ``"cuda"``,
        ``"cuda:N"``.  A ``torch.device`` is returned unchanged.

    Raises
    ------
    RuntimeError
        If CUDA is explicitly requested but unavailable.
    ValueError
        If an unsupported device string is given.
    """
    if isinstance(preference, torch.device):
        return preference

    requested = (preference or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"Unsupported device '{preference}'; use auto, cpu, cuda or cuda:N"
        ) from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{preference}'; use auto, cpu, cuda or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device: cuda was requested but torch.cuda.is_available() is False. "
            "Use device: auto or device: cpu, or install a CUDA build of PyTorch."
        )
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise ValueError(
                f"cuda:{device.index} requested but only {torch.cuda.device_count()} device(s) visible"
            )
    return device


def amp_is_enabled(requested: bool | str | None, device: torch.device) -> bool:
    """AMP is only used on CUDA and only when the config asks for it."""
    if requested is None:
        return False
    if isinstance(requested, str):
        return device.type == "cuda" and requested.strip().lower() in {"1", "true", "yes", "auto"}
    return bool(requested) and device.type == "cuda"


def describe_device(device: torch.device) -> dict[str, Any]:
    """Human readable device description for logs and run metadata."""
    info: dict[str, Any] = {"device": str(device), "type": device.type}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        info.update(
            {
                "name": properties.name,
                "total_memory_gb": round(properties.total_memory / 1024**3, 2),
                "capability": f"{properties.major}.{properties.minor}",
            }
        )
    return info

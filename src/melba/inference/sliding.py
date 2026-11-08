"""Sliding-window inference for 3D volumes (MONAI inferer, config driven).

A 240x240x155 BraTS volume does not fit in 6 GB of VRAM as one forward pass, so both the
evaluation runner and the training-time validation run the network over overlapping
128^3 windows and stitch the probabilities with MONAI's ``sliding_window_inference``.
Keeping it here (instead of calling the inferer in two places) means the reported
validation number and the reported test number come from the same code path.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from melba.utils.device import amp_is_enabled

__all__ = ["DEFAULT_OVERLAP", "DEFAULT_ROI", "build_sliding_window", "sliding_window_predict"]

#: BraTS convention: 128^3 windows with 50 % overlap
DEFAULT_ROI: tuple[int, int, int] = (128, 128, 128)
DEFAULT_OVERLAP: float = 0.5
SLIDING_WINDOW_KEYS: tuple[str, ...] = ("roi", "overlap", "mode", "sw_batch_size", "padding_mode")


def build_sliding_window(config: Any) -> dict[str, Any] | None:
    """Read ``inference.sliding_window`` from a config (``None``/``false`` disables it)."""
    section = config.get("inference.sliding_window")
    if section is None or section is False:
        return None
    if section is True:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError(
            "inference.sliding_window must be a mapping (roi/overlap/mode) or true/false, "
            f"got {type(section).__name__}"
        )
    unknown = sorted(set(section) - set(SLIDING_WINDOW_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown key(s) {unknown} in inference.sliding_window; allowed: {list(SLIDING_WINDOW_KEYS)}"
        )
    roi = section.get("roi", DEFAULT_ROI)
    if isinstance(roi, int):
        roi = (int(roi),) * 3
    roi = tuple(int(value) for value in roi)
    if len(roi) != 3 or any(value < 16 for value in roi):
        raise ValueError(f"inference.sliding_window.roi must be three sizes >= 16, got {roi}")
    overlap = float(section.get("overlap", DEFAULT_OVERLAP))
    if not 0.0 <= overlap < 1.0:
        raise ValueError(
            f"inference.sliding_window.overlap must be in [0, 1), got {overlap}"
        )
    return {
        "roi": roi,
        "overlap": overlap,
        "mode": str(section.get("mode", "gaussian")),
        "sw_batch_size": int(section.get("sw_batch_size", 1)),
        "padding_mode": str(section.get("padding_mode", "constant")),
    }


def sliding_window_predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    *,
    roi: tuple[int, int, int] = DEFAULT_ROI,
    overlap: float = DEFAULT_OVERLAP,
    mode: str = "gaussian",
    sw_batch_size: int = 1,
    padding_mode: str = "constant",
    amp: bool = True,
) -> torch.Tensor:
    """Logits of a ``(1, C, *spatial)`` volume, computed window by window."""
    from monai.inferers import sliding_window_inference

    use_amp = amp_is_enabled(amp, device)
    inputs = tensor.to(device)

    def forward(patch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, enabled=use_amp):
            return model(patch)

    return sliding_window_inference(
        inputs=inputs,
        roi_size=tuple(int(value) for value in roi),
        sw_batch_size=int(sw_batch_size),
        predictor=forward,
        overlap=float(overlap),
        mode=str(mode),
        padding_mode=str(padding_mode),
        progress=False,
    )

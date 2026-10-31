"""Model factory and checkpoint loading.

Architectures and their hyper-parameters come from the ``model`` section of the
YAML config; nothing is hardcoded here beyond the mapping from architecture name
to library constructor.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from melba.config import Config, ConfigError

__all__ = [
    "MODEL_ARCHITECTURES",
    "build_model",
    "checkpoint_sha256",
    "count_parameters",
    "load_checkpoint",
]

MODEL_ARCHITECTURES: tuple[str, ...] = ("smp_unet", "monai_segresnet", "monai_swinunetr")


def _attach_encoder_dropout(model: nn.Module, dropout_p: float) -> None:
    """Apply spatial dropout to the encoder features during training only.

    ``segmentation_models_pytorch`` has no dropout argument, so the features passed
    from the encoder to the decoder are dropped out in place.  Keys of the state
    dict are unchanged (no wrapper module is introduced), and evaluation is
    unaffected because the hook returns the untouched output in ``eval`` mode.
    """
    if dropout_p <= 0:
        return
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ConfigError("model.dropout_p is only supported for architectures with an encoder")

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        if not module.training:
            return output
        if isinstance(output, (list, tuple)):
            return tuple(
                functional.dropout(feature, dropout_p, training=True) for feature in output
            )
        return functional.dropout(output, dropout_p, training=True)

    encoder.register_forward_hook(hook)


def build_model(config: Config) -> nn.Module:
    """Instantiate the architecture declared in ``model.architecture``."""
    architecture = str(config.require("model.architecture")).strip().lower()
    in_channels = int(config.get("model.in_channels", 3))
    num_classes = int(config.get("model.num_classes", 1))
    dropout_p = float(config.get("model.dropout_p", 0.0) or 0.0)
    if num_classes < 1:
        raise ConfigError(f"model.num_classes must be >= 1, got {num_classes}")
    if not 0.0 <= dropout_p < 1.0:
        raise ConfigError(f"model.dropout_p must be in [0, 1), got {dropout_p}")

    if architecture == "smp_unet":
        import segmentation_models_pytorch as smp

        weights = config.get("model.encoder_weights")
        encoder_weights = (
            None
            if weights is None or str(weights).strip().lower() in {"none", "null", "false"}
            else str(weights)
        )
        model = smp.Unet(
            encoder_name=str(config.get("model.encoder", "resnet50")),
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
        _attach_encoder_dropout(model, dropout_p)
        return model

    if architecture == "monai_segresnet":
        from monai.networks.nets import SegResNet

        spatial_dims = int(config.get("model.spatial_dims", 3))
        return SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_classes,
            init_filters=int(config.get("model.init_filters", 16)),
            blocks_down=tuple(config.get("model.blocks_down", (1, 2, 2, 4))),
            blocks_up=tuple(config.get("model.blocks_up", (1, 1, 1))),
            dropout_prob=dropout_p,
        )

    if architecture == "monai_swinunetr":
        from monai.networks.nets import SwinUNETR

        # SwinUNETR is trained from scratch here: no MONAI pretrained weights are
        # downloaded, so the input statistics come from the training split
        # (dataset.normalization.kind: dataset) instead of ImageNet.
        spatial_dims = int(config.get("model.spatial_dims", 2))
        return SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            spatial_dims=spatial_dims,
            feature_size=int(config.get("model.feature_size", 24)),
            depths=tuple(config.get("model.depths", (2, 2, 2, 2))),
            num_heads=tuple(config.get("model.num_heads", (3, 6, 12, 24))),
            patch_size=int(config.get("model.patch_size", 2)),
            window_size=int(config.get("model.window_size", 7)),
            drop_rate=dropout_p,
            attn_drop_rate=float(config.get("model.attn_drop_rate", 0.0)),
            dropout_path_rate=float(config.get("model.dropout_path_rate", 0.0)),
            use_checkpoint=False,
            use_v2=bool(config.get("model.use_v2", False)),
        )

    raise ConfigError(
        f"Unknown model.architecture '{architecture}'. Available: {list(MODEL_ARCHITECTURES)}"
    )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def checkpoint_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of a checkpoint file, recorded in the run metadata."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load weights into ``model`` and return the checkpoint metadata.

    Wrapped checkpoints (``{"model_state": ..., "epoch": ...}`` or a plain
    ``state_dict`` under ``"state_dict"``) are supported, as are ``module.``
    prefixes from ``DataParallel`` training.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train a model first "
            "(scripts/train.py) or set model.checkpoint to an existing file."
        )

    # Checkpoints produced by ``torch.save`` may embed the version object as
    # metadata; allowlist just that harmless class so ``weights_only=True`` still
    # protects against arbitrary pickled objects.
    try:  # pragma: no cover - depends on the torch version
        from torch.torch_version import TorchVersion

        torch.serialization.add_safe_globals([TorchVersion])
    except Exception:  # pragma: no cover - older/newer torch layouts
        pass

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict) and any(
        key in payload for key in ("model_state", "state_dict", "model_state_dict")
    ):
        metadata = {key: value for key, value in payload.items() if not isinstance(value, dict)}
        state = payload.get("model_state") or payload.get("state_dict") or payload["model_state_dict"]
    else:
        state = payload

    cleaned = {
        (key[7:] if key.startswith("module.") else key): value for key, value in dict(state).items()
    }
    try:
        model.load_state_dict(cleaned, strict=strict)
    except RuntimeError as error:
        raise ConfigError(
            f"Checkpoint {checkpoint_path} does not match the configured architecture: {error}"
        ) from error
    return metadata

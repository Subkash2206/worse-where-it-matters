"""MC-dropout: decoder dropout that can be switched on for inference.

Monte-Carlo dropout turns a deterministic network into a sampler by keeping dropout
active at prediction time.  Two rules matter and are enforced here:

* **only dropout is switched on** — the model stays in ``eval()`` mode, so BatchNorm
  keeps using its running statistics (switching to ``train()`` would both change the
  normalisation and update the BN statistics);
* the dropout has to exist in the trained network, otherwise every "sample" would be
  identical.  ``model.decoder_dropout_p > 0`` therefore adds spatial dropout after every
  decoder block of an ``smp.Unet`` (``decoder_*`` in the config refers to the decoder,
  unlike ``model.dropout_p`` which is the encoder-feature dropout used during training).
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as functional

from melba.config import ConfigError

__all__ = [
    "attach_decoder_dropout",
    "decoder_dropout_modules",
    "disable_mc_dropout",
    "enable_mc_dropout",
    "has_mc_dropout",
]

#: attribute set on every decoder block carrying its dropout probability
_DROPOUT_ATTR = "_melba_mc_dropout_p"
#: attribute holding the list of hooked blocks on the model
_MODULES_ATTR = "_melba_mc_dropout_modules"


def attach_decoder_dropout(model: nn.Module, dropout_p: float) -> int:
    """Add spatial dropout after every decoder block of an ``smp.Unet``.

    The dropout fires when the block is in training mode (normal training) **or** when
    MC sampling was enabled with :func:`enable_mc_dropout`.  Returns the number of
    decoder blocks that were instrumented.
    """
    dropout_p = float(dropout_p)
    if not 0.0 < dropout_p < 1.0:
        raise ConfigError(f"model.decoder_dropout_p must be in (0, 1), got {dropout_p}")
    decoder = getattr(model, "decoder", None)
    blocks = list(getattr(decoder, "blocks", []) or [])
    if not blocks:
        raise ConfigError(
            "model.decoder_dropout_p is only supported for architectures with decoder "
            "blocks (smp_unet)"
        )

    def hook(module: nn.Module, inputs: tuple, output):
        probability = getattr(module, _DROPOUT_ATTR, 0.0)
        if probability <= 0 or not (module.training or getattr(module, "_melba_mc", False)):
            return output
        return functional.dropout2d(output, probability, training=True)

    for block in blocks:
        setattr(block, _DROPOUT_ATTR, dropout_p)
        block.register_forward_hook(hook)
    setattr(model, _MODULES_ATTR, blocks)
    return len(blocks)


def decoder_dropout_modules(model: nn.Module) -> list[nn.Module]:
    """Decoder blocks instrumented by :func:`attach_decoder_dropout` (empty if none)."""
    return list(getattr(model, _MODULES_ATTR, []) or [])


def has_mc_dropout(model: nn.Module) -> bool:
    """True when the model carries switchable decoder dropout."""
    return bool(decoder_dropout_modules(model))


def enable_mc_dropout(model: nn.Module) -> int:
    """Switch decoder dropout on without leaving ``eval()`` mode; returns #modules."""
    modules = decoder_dropout_modules(model)
    if not modules:
        raise ConfigError(
            "MC-dropout requested but the model has no decoder dropout; train with "
            "model.decoder_dropout_p > 0"
        )
    for module in modules:
        setattr(module, "_melba_mc", True)
    return len(modules)


def disable_mc_dropout(model: nn.Module) -> None:
    """Switch decoder dropout off again (idempotent)."""
    for module in decoder_dropout_modules(model):
        setattr(module, "_melba_mc", False)

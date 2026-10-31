"""MELBA: geometry-aware test-time augmentation for medical image segmentation.

The package is organised into small, independently testable modules:

``melba.config``        YAML configuration loading (no hardcoded paths).
``melba.utils``         determinism, device detection, run metadata.
``melba.metrics``       overlap and surface metrics (spacing aware).
``melba.transforms``    invertible geometric transforms and TTA view builders.
``melba.inference``     TTA inference with inverse-transformed aggregation.
``melba.data``          group-level splits and dataset readers.
``melba.models``        model factory built from configuration.
``melba.eval``          per-case long-format result records and the runner.
``melba.stats``         paired, non-parametric analysis of paired results.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

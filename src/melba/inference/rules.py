"""Detection-safe decision rules layered on an existing TTA view stack.

TTA raises the Dice of a polyp segmentation by smoothing the boundary, but averaging
probabilities can also *silence* a lesion: if enough views put a small polyp below 0.5, the
mean map falls below 0.5 everywhere and the case is reported as "no polyp".  The rules here
keep the TTA mask whenever it is non-empty and only replace an *empty* TTA mask by a more
detection-friendly decision made from the same forward passes - no extra inference.

Two families are implemented:

``anyview``
    The lesion is present when any single view (the identity view first) or the mean map is
    non-empty at the decision threshold.  The mask is, in order of preference: the mean-prob
    mask at the threshold if non-empty, else the identity-view mask, else the union of the
    non-empty view masks.

``lowdet_<t>``
    If the peak of the mean probability map reaches ``t`` while the thresholded mean mask is
    empty, the mask is the mean map thresholded at ``t`` (a lower detection threshold); in
    every other case it is the plain thresholded mean mask.  ``t`` is selected on internal
    data only (see ``scripts/select_detection_rule.py``).

Both rules are pure functions of ``(rule, mean_probability, per_view_probabilities,
threshold)`` so they can be unit-tested without a model, and both return a boolean array of
the same shape as the probability map.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "ANYVIEW",
    "DETECTION_RULES",
    "LOWDET_THRESHOLDS",
    "RULE_PATTERN",
    "apply_rule",
    "parse_rule",
    "rule_name",
]

ANYVIEW = "anyview"
#: detection thresholds the ``lowdet`` family exposes (chosen on internal data only)
LOWDET_THRESHOLDS: tuple[float, ...] = (0.2, 0.3, 0.4)
DETECTION_RULES: tuple[str, ...] = (ANYVIEW,) + tuple(
    f"lowdet_{int(round(t * 100)):02d}" for t in LOWDET_THRESHOLDS
)
#: a rule token as it appears in a method name, e.g. ``anyview`` or ``lowdet_03``
RULE_PATTERN = re.compile(r"^(anyview|lowdet_(?P<t>\d{1,2}))$")


def parse_rule(text: str) -> tuple[str, float | None]:
    """Split a rule token into ``(family, lowdet threshold or None)``.

    ``rule_name`` is the inverse of this function, so a rule that parses always produces a
    method name that parses back to the same rule.
    """
    token = str(text).strip().lower()
    match = RULE_PATTERN.match(token)
    if match is None:
        raise ValueError(
            f"Unknown detection rule {text!r}; expected 'anyview' or 'lowdet_<percent>' "
            f"with percent in {[int(round(t * 100)) for t in LOWDET_THRESHOLDS]}"
        )
    percent = match.group("t")
    if percent is None:
        return ANYVIEW, None
    threshold = int(percent) / 100.0
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"lowdet threshold must be in (0, 1), got {threshold}")
    return "lowdet", threshold


def rule_name(family: str, threshold: float | None = None) -> str:
    """Canonical rule token (``anyview`` / ``lowdet_03``) of a family and threshold."""
    if family == ANYVIEW:
        return ANYVIEW
    if family != "lowdet" or threshold is None:
        raise ValueError(f"Unknown detection rule family {family!r}")
    return f"lowdet_{int(round(float(threshold) * 100)):02d}"


def _channel(array: np.ndarray) -> np.ndarray:
    """Drop a leading channel axis of size one (single-channel binary predictions)."""
    array = np.asarray(array)
    return array[0] if array.ndim >= 3 and array.shape[0] == 1 else array


def apply_rule(
    rule: str,
    mean_probability: np.ndarray,
    view_probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    identity_index: int = 0,
) -> np.ndarray:
    """Boolean mask of one case under a detection rule.

    Parameters
    ----------
    rule:
        ``anyview`` or ``lowdet_<percent>`` (see :func:`parse_rule`).
    mean_probability:
        Aggregated probability map, shape ``(C, *spatial)`` or ``(*spatial,)``.
    view_probabilities:
        Per-view probability maps, shape ``(n_views, C, *spatial)``.  The identity view is the
        one at ``identity_index`` (the configs list it first).
    threshold:
        Decision threshold of the ordinary TTA mask (0.5 in this project).
    identity_index:
        Position of the identity view in ``view_probabilities``.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the spatial shape of the probability maps.
    """
    family, low = parse_rule(rule)
    mean = _channel(mean_probability)
    mean_mask = mean >= float(threshold)

    if family == ANYVIEW:
        if mean_mask.any():
            return mean_mask
        views = np.asarray(view_probabilities)
        identity = _channel(views[int(identity_index)])
        identity_mask = identity >= float(threshold)
        if identity_mask.any():
            return identity_mask
        union = np.zeros(mean_mask.shape, dtype=bool)
        for view in views:
            union |= _channel(view) >= float(threshold)
        return union

    # lowdet: only ever *rescues* an empty mask, never re-thresholds a non-empty one
    if not mean_mask.any() and float(mean.max()) >= float(low):
        return mean >= float(low)
    return mean_mask

"""TTA inference: inverse mapping, aggregation and configuration wiring."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from melba.config import ConfigError
from melba.inference import (
    TTAMethod,
    aggregate_probabilities,
    logits_to_probabilities,
    methods_from_config,
    predict_probabilities,
)
from melba.transforms import build_views


class IdentityChannelModel(nn.Module):
    """Returns the first input channel as logits (deterministic and view-equivariant)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


class ConstantModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[:, :1], 0.25)


class ShiftModel(nn.Module):
    """Shifts its prediction one pixel to the left; not invariant to flips."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.roll(x[:, :1], shifts=-1, dims=-1)


class BadShapeModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1, :-1]


def _method(name: str, specs, aggregator: str = "mean") -> TTAMethod:
    return TTAMethod(name=name, views=tuple(build_views(specs)), aggregator=aggregator)


def test_methods_from_config(base_config) -> None:
    methods = methods_from_config(base_config)
    assert [method.name for method in methods] == ["baseline", "tta2", "tta4", "tta8"]
    assert [method.n_views for method in methods] == [1, 2, 4, 8]
    assert all(method.aggregator == "mean" for method in methods)
    assert methods[0].views[0].name == "identity"
    assert methods[2].view_names.split("|")[0] == "identity"


def _view_stack(method: TTAMethod, tensor: torch.Tensor) -> list[tuple[int, ...]]:
    return [tuple(view.apply(tensor).flatten().tolist()) for view in method.views]


def test_d4_membership_is_exact(base_config) -> None:
    """1/2/4/8 views must be the documented D4 subsets, no duplicates."""
    methods = {method.name: method for method in methods_from_config(base_config)}
    tensor = torch.arange(1 * 1 * 5 * 7, dtype=torch.float32).reshape(1, 1, 5, 7)

    def stack(specs):
        return _view_stack(_method("ref", specs), tensor)

    reference_1 = stack([{"name": "identity", "ops": []}])
    reference_2 = stack(
        [
            {"name": "identity", "ops": []},
            {"name": "hflip", "ops": [{"type": "flip", "axes": ["w"]}]},
        ]
    )
    reference_4 = stack(
        [
            {"name": "identity", "ops": []},
            {"name": "rot180", "ops": [{"type": "rot90", "k": 2, "axes": ["h", "w"]}]},
            {"name": "hflip", "ops": [{"type": "flip", "axes": ["w"]}]},
            {"name": "vflip", "ops": [{"type": "flip", "axes": ["h"]}]},
        ]
    )
    reference_8 = stack(
        [
            {"name": "identity", "ops": []},
            {"name": "rot90", "ops": [{"type": "rot90", "k": 1, "axes": ["h", "w"]}]},
            {"name": "rot180", "ops": [{"type": "rot90", "k": 2, "axes": ["h", "w"]}]},
            {"name": "rot270", "ops": [{"type": "rot90", "k": 3, "axes": ["h", "w"]}]},
            {"name": "hflip", "ops": [{"type": "flip", "axes": ["w"]}]},
            {"name": "hflip_rot90", "ops": [{"type": "flip", "axes": ["w"]}, {"type": "rot90", "k": 1, "axes": ["h", "w"]}]},
            {"name": "hflip_rot180", "ops": [{"type": "flip", "axes": ["w"]}, {"type": "rot90", "k": 2, "axes": ["h", "w"]}]},
            {"name": "hflip_rot270", "ops": [{"type": "flip", "axes": ["w"]}, {"type": "rot90", "k": 3, "axes": ["h", "w"]}]},
        ]
    )
    # The test recomputes signatures with the configured method itself so that
    # shape-changing views (rot90 on a non-square grid) are compared as arrays.
    for name, reference in (
        ("baseline", reference_1),
        ("tta2", reference_2),
        ("tta4", reference_4),
        ("tta8", reference_8),
    ):
        method = methods[name]
        signatures = [tuple(view.apply(tensor).flatten().tolist()) for view in method.views]
        assert len(signatures) == len(reference)
        assert len(set(signatures)) == len(signatures), f"{name} contains duplicate views"
        assert signatures == reference, f"{name} does not match the documented D4 subset"


def test_ttamethod_validation() -> None:
    with pytest.raises(ValueError, match="at least one view"):
        TTAMethod(name="empty", views=())
    with pytest.raises(ValueError, match="aggregator"):
        TTAMethod(
            name="bad", views=tuple(build_views([{"name": "id", "ops": []}])), aggregator="mode"
        )


def test_logits_to_probabilities() -> None:
    logits = torch.tensor([[[[0.0, 1.0]]]])
    torch.testing.assert_close(
        logits_to_probabilities(logits, "sigmoid"),
        torch.tensor([[[[0.5, 1 / (1 + math.e**-1)]]]]),
    )
    multi = torch.zeros(1, 3, 1, 1)
    multi[0, 0, 0, 0] = 1.0
    probabilities = logits_to_probabilities(multi, "softmax")
    assert float(probabilities.sum()) == pytest.approx(1.0)
    with pytest.raises(ConfigError):
        logits_to_probabilities(logits, "tanh")


def test_aggregate_probabilities() -> None:
    stack = np.stack([np.zeros((1, 2, 2)), np.full((1, 2, 2), 2.0)])
    np.testing.assert_allclose(aggregate_probabilities(stack, "mean"), 1.0)
    np.testing.assert_allclose(aggregate_probabilities(stack, "median"), 1.0)
    np.testing.assert_allclose(aggregate_probabilities(stack, "max"), 2.0)
    with pytest.raises(ValueError):
        aggregate_probabilities(stack, "mode")


def test_equivariant_model_gives_identical_result_for_any_geometric_view() -> None:
    """If the model commutes with the views, TTA must equal the baseline exactly."""
    image = torch.rand(1, 3, 11, 9)
    baseline = predict_probabilities(
        IdentityChannelModel(),
        image,
        torch.device("cpu"),
        _method("baseline", [{"name": "identity", "ops": []}]),
        amp=False,
    )
    views = [
        {"name": "identity", "ops": []},
        {"name": "hflip", "ops": [{"type": "flip", "axes": ["w"]}]},
        {"name": "rot90", "ops": [{"type": "rot90", "k": 1, "axes": ["h", "w"]}]},
        {"name": "rot180", "ops": [{"type": "rot90", "k": 2, "axes": ["h", "w"]}]},
        {
            "name": "hflip_rot90",
            "ops": [
                {"type": "flip", "axes": ["w"]},
                {"type": "rot90", "k": 1, "axes": ["h", "w"]},
            ],
        },
    ]
    tta = predict_probabilities(
        IdentityChannelModel(), image, torch.device("cpu"), _method("tta", views), amp=False
    )
    expected = torch.sigmoid(image[:, :1])[0].numpy()
    np.testing.assert_allclose(baseline, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tta, expected, rtol=1e-6, atol=1e-6)


def test_inverse_is_applied_before_aggregation() -> None:
    """A translation-equivariant model is not flip invariant: the ensemble must average."""
    image = torch.rand(1, 3, 8, 10)
    baseline_method = _method("baseline", [{"name": "identity", "ops": []}])
    flip_method = _method(
        "flip",
        [
            {"name": "identity", "ops": []},
            {"name": "hflip", "ops": [{"type": "flip", "axes": ["w"]}]},
        ],
    )
    model = ShiftModel()
    baseline = predict_probabilities(
        model, image, torch.device("cpu"), baseline_method, amp=False
    )
    ensembled = predict_probabilities(
        model, image, torch.device("cpu"), flip_method, amp=False
    )

    logits = image[:, :1]
    expected = (
        torch.sigmoid(torch.roll(logits, -1, dims=-1))
        + torch.sigmoid(torch.roll(logits, 1, dims=-1))
    ) / 2
    np.testing.assert_allclose(ensembled, expected[0].numpy(), rtol=1e-6, atol=1e-6)
    assert not np.allclose(baseline, ensembled)


def test_aggregators_on_a_constant_model() -> None:
    image = torch.rand(1, 1, 5, 6)
    views = [
        {"name": "identity", "ops": []},
        {"name": "hflip", "ops": [{"type": "flip", "axes": ["w"]}]},
    ]
    for aggregator in ("mean", "median", "max"):
        probabilities = predict_probabilities(
            ConstantModel(),
            image,
            torch.device("cpu"),
            _method("constant", views, aggregator=aggregator),
            amp=False,
        )
        np.testing.assert_allclose(probabilities, torch.sigmoid(torch.tensor(0.25)).item())


def test_output_shape_and_dtype() -> None:
    image = torch.rand(1, 3, 7, 9)
    method = _method(
        "zoom",
        [
            {"name": "identity", "ops": []},
            {
                "name": "scale",
                "ops": [
                    {
                        "type": "zoom",
                        "axes": ["h", "w"],
                        "factors": [0.8, 1.25],
                        "mode": "bilinear",
                    }
                ],
            },
        ],
    )
    probabilities = predict_probabilities(
        IdentityChannelModel(), image, torch.device("cpu"), method, amp=False
    )
    assert probabilities.shape == (1, 7, 9)
    assert probabilities.dtype == np.float32
    assert np.isfinite(probabilities).all()


def test_predict_per_view_returns_every_view(base_config) -> None:
    from melba.inference import predict_per_view

    methods = {method.name: method for method in methods_from_config(base_config)}
    image = torch.rand(1, 3, 8, 10)
    view_stack = predict_per_view(
        ShiftModel(), image, torch.device("cpu"), methods["tta4"], amp=False
    )
    assert view_stack.shape == (4, 1, 8, 10)
    assert np.isfinite(view_stack).all()
    # the mean of the views is what the mean aggregator reports
    aggregated = predict_probabilities(
        ShiftModel(), image, torch.device("cpu"), methods["tta4"], amp=False
    )
    np.testing.assert_allclose(view_stack.mean(axis=0), aggregated, rtol=1e-6, atol=1e-6)
    # a shift-equivariant model gives genuinely different views
    assert not np.allclose(view_stack[0], view_stack[1])


def test_invalid_model_output_and_inputs() -> None:
    image = torch.rand(1, 1, 6, 6)
    method = _method("baseline", [{"name": "identity", "ops": []}])
    with pytest.raises(RuntimeError, match="incompatible"):
        predict_probabilities(BadShapeModel(), image, torch.device("cpu"), method, amp=False)
    with pytest.raises(ValueError, match="batch size 1"):
        predict_probabilities(
            ConstantModel(), torch.rand(2, 1, 6, 6), torch.device("cpu"), method, amp=False
        )
    with pytest.raises(ValueError, match="batched tensor"):
        predict_probabilities(
            ConstantModel(), torch.rand(1, 6, 6), torch.device("cpu"), method, amp=False
        )


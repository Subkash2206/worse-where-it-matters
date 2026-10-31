"""Every transform/inverse pair must round-trip (the old code kept hand-written inverses)."""

from __future__ import annotations

import pytest
import torch

from melba.config import ConfigError, load_config
from melba.transforms import (
    Compose,
    Flip,
    Identity,
    Rot90,
    Transpose,
    Zoom,
    axis_to_dim,
    build_transform,
    build_view,
    build_views,
    method_specs_from_config,
)

SHAPE_PRESERVING_2D = [
    Identity(),
    Flip([-1]),
    Flip([-2]),
    Flip([-2, -1]),
    Rot90(1),
    Rot90(2),
    Rot90(3),
    Transpose([-2, -1]),
    Compose([Flip([-1]), Rot90(1)]),
    Compose([Rot90(3), Flip([-2]), Transpose([-2, -1])]),
]

SHAPE_PRESERVING_3D = [
    Identity(),
    Flip([-1]),
    Flip([-3]),
    Flip([-3, -1]),
    Rot90(1, (-2, -1)),
    Rot90(2, (-3, -1)),
    Transpose([-3, -2]),
    Compose([Rot90(1, (-2, -1)), Flip([-1]), Rot90(3, (-3, -1))]),
]


@pytest.mark.parametrize("transform", SHAPE_PRESERVING_2D, ids=lambda t: t.name)
def test_roundtrip_2d(transform) -> None:
    tensor = torch.arange(1 * 3 * 6 * 8, dtype=torch.float32).reshape(1, 3, 6, 8)
    transformed = transform.apply(tensor)
    restored = transform.invert(transformed, tensor.shape)
    assert restored.shape == tensor.shape
    torch.testing.assert_close(restored, tensor, rtol=0, atol=0)


@pytest.mark.parametrize("transform", SHAPE_PRESERVING_3D, ids=lambda t: t.name)
def test_roundtrip_3d(transform) -> None:
    tensor = torch.arange(2 * 1 * 5 * 6 * 7, dtype=torch.float32).reshape(2, 1, 5, 6, 7)
    transformed = transform.apply(tensor)
    restored = transform.invert(transformed, tensor.shape)
    assert restored.shape == tensor.shape
    torch.testing.assert_close(restored, tensor, rtol=0, atol=0)


def test_rot90_changes_and_restores_shape_for_non_square_input() -> None:
    tensor = torch.zeros(1, 1, 6, 8)
    transform = Rot90(1, (-2, -1))
    transformed = transform.apply(tensor)
    assert tuple(transformed.shape) == (1, 1, 8, 6)
    assert tuple(transform.invert(transformed, tensor.shape).shape) == (1, 1, 6, 8)


def test_inverse_of_inverse_is_the_forward_transform() -> None:
    transform = Compose([Flip([-1]), Rot90(3), Transpose([-2, -1])])
    tensor = torch.rand(2, 3, 7, 5)
    transformed = transform.apply(tensor)
    again = transform.apply(transform.invert(transformed, tensor.shape))
    torch.testing.assert_close(again, transformed, rtol=0, atol=0)


def test_zoom_restores_shape_and_is_exact_for_constants_and_nearest_upsampling() -> None:
    tensor = torch.rand(2, 3, 10, 12)
    for factor in (0.8, 1.25, 2.0):
        zoom = Zoom([factor, factor], [-2, -1], mode="bilinear")
        transformed = zoom.apply(tensor)
        restored = zoom.invert(transformed, tensor.shape)
        assert tuple(restored.shape) == tuple(tensor.shape)

    constant = torch.full((1, 1, 9, 11), 0.37)
    zoom = Zoom([1.7, 0.6], [-2, -1])
    torch.testing.assert_close(
        zoom.invert(zoom.apply(constant), constant.shape), constant, atol=1e-6, rtol=0
    )

    mask = (torch.rand(1, 1, 9, 11) > 0.5).float()
    nearest = Zoom([3, 3], [-2, -1], mode="nearest")
    torch.testing.assert_close(
        nearest.invert(nearest.apply(mask), mask.shape), mask, rtol=0, atol=0
    )


def test_zoom_roundtrip_is_close_for_smooth_fields() -> None:
    coordinates = torch.arange(16, dtype=torch.float32)
    smooth = torch.sin(coordinates / 3.0).reshape(1, 1, 16, 1) * torch.cos(
        coordinates / 4.0
    ).reshape(1, 1, 1, 16)
    zoom = Zoom([1.25, 1.25], [-2, -1], mode="bilinear")
    restored = zoom.invert(zoom.apply(smooth), smooth.shape)
    assert float((restored - smooth).abs().max()) < 0.1


def test_zoom_requires_original_shape() -> None:
    with pytest.raises(ValueError, match="original tensor shape"):
        Zoom([1.5, 1.5], [-2, -1]).invert(torch.rand(1, 1, 7, 7))


def test_zoom_works_on_3d_tensors() -> None:
    volume = torch.rand(1, 1, 6, 8, 10)
    zoom = Zoom([1.25, 0.75, 1.25], [-3, -2, -1], mode="trilinear")
    transformed = zoom.apply(volume)
    assert tuple(transformed.shape) == (1, 1, 8, 6, 12)
    restored = zoom.invert(transformed, volume.shape)
    assert tuple(restored.shape) == tuple(volume.shape)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Flip([]),
        lambda: Flip([-1, -1]),
        lambda: Rot90(1, (-1,)),
        lambda: Rot90(1, (-1, -1)),
        lambda: Rot90(4),
        lambda: Transpose([-1]),
        lambda: Transpose([-1, -1]),
        lambda: Zoom([1.0], [-1, -2]),
        lambda: Zoom([0.0, 1.0], [-2, -1]),
        lambda: Zoom([1.0, 1.0], [-2, -1], mode="lanczos"),
    ],
)
def test_transform_argument_validation(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_compose_rejects_non_transforms() -> None:
    with pytest.raises(TypeError):
        Compose([Flip([-1]), "not a transform"])


def test_axis_names() -> None:
    assert axis_to_dim("w") == -1
    assert axis_to_dim("h") == -2
    assert axis_to_dim("d") == -3
    assert axis_to_dim("z") == -3
    assert axis_to_dim(2) == 2
    with pytest.raises(ConfigError):
        axis_to_dim("q")


def test_build_transform_from_yaml_specs() -> None:
    flip = build_transform({"type": "flip", "axes": ["w"]})
    assert isinstance(flip, Flip) and flip.dims == (-1,)
    rot = build_transform({"type": "rot90", "k": 3, "axes": ["h", "w"]})
    assert isinstance(rot, Rot90) and rot.k == 3 and rot.dims == (-2, -1)
    transpose = build_transform({"type": "transpose", "axes": ["h", "w"]})
    assert isinstance(transpose, Transpose)
    zoom = build_transform(
        {"type": "zoom", "axes": ["h", "w"], "factors": [1.2, 0.8], "mode": "bilinear"}
    )
    assert isinstance(zoom, Zoom) and zoom.factors == (1.2, 0.8)
    scalar_zoom = build_transform({"type": "zoom", "axes": ["h", "w"], "factor": 1.5})
    assert scalar_zoom.factors == (1.5, 1.5)
    assert isinstance(build_transform({"type": "identity"}), Identity)


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"type": "shear"},
        {"type": "flip"},
        {"type": "flip", "axes": ["w"], "unexpected": 1},
        {"type": "rot90", "axes": ["h", "w", "d"]},
        {"type": "zoom", "axes": ["h", "w"]},
        {"type": "zoom", "axes": ["h", "w"], "factors": [1.0]},
        {"type": "zoom", "axes": ["w"], "factors": [1.0, 2.0]},
    ],
)
def test_build_transform_rejects_bad_specs(spec) -> None:
    with pytest.raises(ConfigError):
        build_transform(spec)


def test_build_view_and_view_list() -> None:
    view = build_view({"name": "flip_w", "ops": [{"type": "flip", "axes": ["w"]}]})
    assert view.name == "flip_w"
    unnamed = build_view({"ops": [{"type": "flip", "axes": ["h"]}]})
    assert isinstance(unnamed.transform, Compose)
    assert unnamed.name.startswith("compose(")
    no_ops = build_view({"name": "identity"})
    assert isinstance(no_ops.transform, Identity)

    views = build_views(
        [
            {"name": "a", "ops": []},
            {"name": "b", "ops": [{"type": "flip", "axes": ["w"]}]},
        ]
    )
    assert [view.name for view in views] == ["a", "b"]

    with pytest.raises(ConfigError, match="Duplicate"):
        build_views([{"name": "x", "ops": []}, {"name": "x", "ops": []}])
    with pytest.raises(ConfigError, match="at least one view"):
        build_views([])
    with pytest.raises(ConfigError):
        build_view({"name": "bad", "ops": [{"type": "flip"}], "extra": 1})


def _contains_zoom(transform) -> bool:
    if isinstance(transform, Zoom):
        return True
    if isinstance(transform, Compose):
        return any(_contains_zoom(child) for child in transform.transforms)
    return False


def test_every_configured_view_round_trips(cvc_config_path) -> None:
    """The YAML is the source of truth: every view it declares must invert correctly."""
    config = load_config(cvc_config_path)
    tensor = torch.rand(1, 3, 7, 9)
    checked = 0
    for method in method_specs_from_config(config):
        for view in build_views(method["views"]):
            transformed = view.apply(tensor)
            restored = view.invert(transformed, tensor.shape)
            assert tuple(restored.shape) == tuple(tensor.shape)
            if _contains_zoom(view.transform):
                # resampling is not lossless, only the geometry must be restored
                assert restored.shape == tensor.shape
            else:
                torch.testing.assert_close(restored, tensor, rtol=0, atol=0)
            checked += 1
    assert checked >= 15


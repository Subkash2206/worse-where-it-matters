"""MC-dropout: switchable decoder dropout, frozen BatchNorm and the ``mcdropout_T`` method."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from melba.config import ConfigError
from melba.eval import evaluate, read_long_csv
from melba.inference import TTAMethod, methods_from_config, predict_per_view
from melba.models import build_model, enable_mc_dropout, has_mc_dropout
from melba.models.dropout import disable_mc_dropout


def _dropout_config(config, **overrides):
    settings = {
        "model.encoder": "resnet18",
        "model.encoder_weights": None,
        "model.decoder_dropout_p": 0.2,
        "dataset.image_size": [64, 64],
        "runtime.progress": False,
    }
    settings.update(overrides)
    return config.with_overrides(settings)


def _bn_state(model: nn.Module) -> dict[str, tuple[torch.Tensor, ...]]:
    return {
        name: (
            module.running_mean.clone(),
            module.running_var.clone(),
            module.num_batches_tracked.clone(),
        )
        for name, module in model.named_modules()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    }


def _assert_bn_unchanged(model: nn.Module, before: dict[str, tuple[torch.Tensor, ...]]) -> None:
    after = _bn_state(model)
    assert set(after) == set(before)
    for name, (mean, var, count) in before.items():
        assert torch.equal(after[name][0], mean), f"running_mean of {name} changed"
        assert torch.equal(after[name][1], var), f"running_var of {name} changed"
        assert int(after[name][2]) == int(count), f"num_batches_tracked of {name} changed"


def test_decoder_dropout_is_switchable_and_off_by_default(toy_dataset_config) -> None:
    config = _dropout_config(toy_dataset_config)
    model = build_model(config)
    assert has_mc_dropout(model)

    image = torch.rand(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        first = model(image)
        second = model(image)
    # a plain eval() forward is deterministic: dropout is inactive
    assert torch.allclose(first, second)


def test_mc_samples_differ_while_batch_norm_statistics_are_frozen(toy_dataset_config) -> None:
    config = _dropout_config(toy_dataset_config)
    model = build_model(config)
    image = torch.rand(1, 3, 64, 64)
    model.eval()
    before = _bn_state(model)
    assert before, "the decoder is expected to contain BatchNorm layers"

    enable_mc_dropout(model)
    assert model.training is False, "MC sampling must not put the model into training mode"
    with torch.no_grad():
        samples = [model(image) for _ in range(8)]
    disable_mc_dropout(model)

    stacked = torch.stack(samples)
    spread = stacked.max(dim=0).values - stacked.min(dim=0).values
    assert float(spread.max()) > 0.0, "MC-dropout samples should not be identical"
    assert len({tuple(sample.flatten()[:8].tolist()) for sample in samples}) > 1
    # BatchNorm keeps using (and never updates) its running statistics
    _assert_bn_unchanged(model, before)
    assert model.training is False

    with torch.no_grad():
        first = model(image)
        second = model(image)
    assert torch.allclose(first, second), "dropout must be off again after sampling"


def test_predict_per_view_restores_dropout_state(toy_dataset_config) -> None:
    config = _dropout_config(toy_dataset_config)
    model = build_model(config)
    model.eval()
    views = methods_from_config(
        config.with_overrides({"tta.extra_methods": ["mcdropout_4"]})
    )[-1].views
    method = TTAMethod(name="mcdropout_4", views=views, stochastic=True)
    before = _bn_state(model)
    stacked = predict_per_view(model, torch.rand(1, 3, 64, 64), torch.device("cpu"), method)
    assert stacked.shape[0] == 4
    assert np.isfinite(stacked).all()
    assert float(stacked.std(axis=0).max()) > 0.0, "the samples should not be identical"
    _assert_bn_unchanged(model, before)
    assert all(
        getattr(module, "_melba_mc", False) is False
        for module in model.modules()
        if hasattr(module, "_melba_mc")
    )


def test_methods_from_config_expands_mcdropout(toy_dataset_config) -> None:
    methods = methods_from_config(
        toy_dataset_config.with_overrides({"tta.extra_methods": ["mcdropout_4"]})
    )
    method = methods[-1]
    assert method.name == "mcdropout_4"
    assert method.stochastic is True
    assert method.n_views == 4
    assert method.view_names == "mc_sample_1|mc_sample_2|mc_sample_3|mc_sample_4"
    # the geometric methods are still there
    assert [entry.name for entry in methods[:4]] == ["baseline", "tta2", "tta4", "tta8"]
    assert all(entry.stochastic is False for entry in methods[:4])

    default = methods_from_config(
        toy_dataset_config.with_overrides({"tta.extra_methods": ["mcdropout"]})
    )[-1]
    assert default.n_views == 8  # T = 8 by default


def test_explicit_stochastic_spec_is_honoured(toy_dataset_config) -> None:
    config = toy_dataset_config.with_overrides(
        {
            "tta.methods": [
                {
                    "name": "mc",
                    "stochastic": True,
                    "views": [{"name": "s1", "ops": []}, {"name": "s2", "ops": []}],
                }
            ]
        }
    )
    method = methods_from_config(config)[0]
    assert method.stochastic is True
    assert method.n_views == 2


def test_runner_rejects_stochastic_methods_without_dropout(toy_dataset_config) -> None:
    config = _dropout_config(
        toy_dataset_config,
        **{
            "model.decoder_dropout_p": 0.0,
            "tta.extra_methods": ["mcdropout_4"],
            "evaluation.timing": False,
        },
    )
    model = build_model(config)
    assert not has_mc_dropout(model)
    with pytest.raises(ConfigError, match="MC-dropout"):
        evaluate(config, model=model, limit=1, progress=False)


def test_evaluate_writes_mc_dropout_uncertainty_columns(toy_dataset_config, tmp_path) -> None:
    config = _dropout_config(
        toy_dataset_config,
        **{
            "tta.extra_methods": ["mcdropout_4"],
            "evaluation.timing": False,
            "evaluation.output_dir": str(tmp_path / "mc_results"),
        },
    )
    model = build_model(config)
    result = evaluate(config, model=model, limit=2, progress=False)

    frame = read_long_csv(result.per_case_csv)
    mc_rows = frame.loc[frame["method"] == "mcdropout_4"]
    assert not mc_rows.empty
    assert set(mc_rows["n_views"]) == {4}

    wide = mc_rows.pivot_table(index="case_id", columns="metric", values="value", aggfunc="mean")
    for column in ("dice", "entropy_mean", "variance_mean", "view_disagreement"):
        assert column in wide.columns, column
    assert np.isfinite(wide["variance_mean"]).all()
    # dropout is switched off again once the run is finished
    assert all(
        getattr(module, "_melba_mc", False) is False
        for module in model.modules()
        if hasattr(module, "_melba_mc")
    )


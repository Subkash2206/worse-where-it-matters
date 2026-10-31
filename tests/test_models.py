"""Model factory and checkpoint handling."""

from __future__ import annotations

import pytest
import torch

from melba.config import ConfigError
from melba.models import (
    build_model,
    checkpoint_sha256,
    count_parameters,
    load_checkpoint,
)


def test_build_model_from_config(toy_dataset_config) -> None:
    config = toy_dataset_config.with_overrides({"model.encoder_weights": None})
    model = build_model(config)
    assert count_parameters(model) > 0
    logits = model(torch.rand(1, 3, 64, 64))
    assert tuple(logits.shape) == (1, 1, 64, 64)

    with pytest.raises(ConfigError, match="Unknown model.architecture"):
        build_model(config.with_overrides({"model.architecture": "nope"}))


def test_checkpoint_roundtrip(toy_dataset_config, tmp_path) -> None:
    config = toy_dataset_config.with_overrides({"model.encoder_weights": None})
    model = build_model(config)
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": 3,
            "val_loss": 0.25,
            "val_dice": 0.8,
            "seed": 0,
            "torch_version": str(torch.__version__),
        },
        path,
    )

    reloaded = build_model(config)
    metadata = load_checkpoint(reloaded, path, torch.device("cpu"))
    assert metadata["epoch"] == 3
    assert metadata["val_dice"] == pytest.approx(0.8)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], tensor)

    # plain state_dict (and DataParallel "module." prefixes) are also accepted
    plain = tmp_path / "plain.pt"
    torch.save({f"module.{key}": value for key, value in model.state_dict().items()}, plain)
    load_checkpoint(build_model(config), plain, torch.device("cpu"))

    assert checkpoint_sha256(path) == checkpoint_sha256(path)
    assert checkpoint_sha256(path) != checkpoint_sha256(plain)


def test_checkpoint_errors(toy_dataset_config, tmp_path) -> None:
    config = toy_dataset_config.with_overrides({"model.encoder_weights": None})
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(build_model(config), tmp_path / "missing.pt", torch.device("cpu"))

    wrong = tmp_path / "wrong.pt"
    torch.save({"model_state": {"unknown.weight": torch.zeros(1)}}, wrong)
    with pytest.raises(ConfigError, match="does not match"):
        load_checkpoint(build_model(config), wrong, torch.device("cpu"))


def test_dropout_only_affects_training(toy_dataset_config) -> None:
    plain_config = toy_dataset_config.with_overrides(
        {"model.encoder_weights": None, "model.dropout_p": 0.0}
    )
    dropped_config = toy_dataset_config.with_overrides(
        {"model.encoder_weights": None, "model.dropout_p": 0.5}
    )
    torch.manual_seed(0)
    plain = build_model(plain_config)
    torch.manual_seed(0)
    dropped = build_model(dropped_config)
    # a forward hook is used, so the state dict keys stay identical
    assert set(plain.state_dict()) == set(dropped.state_dict())

    inputs = torch.rand(2, 3, 32, 32)
    plain.train()
    torch.manual_seed(1)
    first = plain(inputs)
    torch.manual_seed(2)
    second = plain(inputs)
    torch.testing.assert_close(first, second)

    dropped.train()
    torch.manual_seed(3)
    third = dropped(inputs)
    torch.manual_seed(4)
    fourth = dropped(inputs)
    assert not torch.allclose(third, fourth)

    dropped.eval()
    with torch.no_grad():
        torch.testing.assert_close(dropped(inputs), dropped(inputs))

    with pytest.raises(ConfigError, match="dropout_p"):
        build_model(plain_config.with_overrides({"model.dropout_p": 1.0}))

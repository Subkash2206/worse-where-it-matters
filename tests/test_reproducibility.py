"""Determinism helpers and automatic device selection."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from melba.utils import (
    amp_is_enabled,
    describe_device,
    environment_info,
    get_device,
    seed_everything,
)
from melba.utils.provenance import git_revision


def _draw() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.rand()), torch.rand(3)


def test_seed_everything_makes_every_stream_repeatable() -> None:
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    second = _draw()
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])

    seed_everything(124)
    third = _draw()
    assert third[0] != first[0]


def test_seed_everything_rejects_invalid_seeds() -> None:
    with pytest.raises(TypeError):
        seed_everything(True)
    with pytest.raises(TypeError):
        seed_everything("0")
    with pytest.raises(ValueError):
        seed_everything(-3)
    assert seed_everything(np.int64(7)) == 7
    assert torch.backends.cudnn.deterministic is True
    assert torch.are_deterministic_algorithms_enabled() is True


def test_cublas_workspace_config_is_set_for_determinism() -> None:
    import os

    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    seed_everything(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_worker_init_fn_is_picklable_and_seeds() -> None:
    """Windows/macOS workers use spawn, so the hook must survive pickling."""
    import pickle
    import random

    import numpy as np

    from melba.utils.reproducibility import make_worker_init_fn

    hook = make_worker_init_fn(123, deterministic=True)
    restored = pickle.loads(pickle.dumps(hook))

    random.seed(0)
    np.random.seed(0)
    restored(4)
    first = (random.random(), float(np.random.rand()))
    random.seed(0)
    np.random.seed(0)
    restored(4)
    second = (random.random(), float(np.random.rand()))
    assert first == second

    random.seed(0)
    np.random.seed(0)
    restored(5)
    assert (random.random(), float(np.random.rand())) != first


def test_get_device_auto_and_cpu() -> None:
    assert get_device("cpu").type == "cpu"
    auto = get_device("auto")
    assert auto.type in {"cpu", "cuda"}
    assert get_device(torch.device("cpu")).type == "cpu"
    with pytest.raises(ValueError):
        get_device("tpu")


def test_get_device_cuda_requests(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        get_device("cuda")
    assert get_device("auto").type == "cpu"


def test_amp_is_enabled() -> None:
    cpu = torch.device("cpu")
    assert amp_is_enabled(True, cpu) is False
    assert amp_is_enabled(False, cpu) is False
    assert amp_is_enabled(None, cpu) is False
    assert amp_is_enabled(True, torch.device("cuda", 0)) is True
    assert amp_is_enabled("false", torch.device("cuda", 0)) is False


def test_describe_device_and_environment() -> None:
    description = describe_device(torch.device("cpu"))
    assert description["type"] == "cpu"
    info = environment_info(torch.device("cpu"))
    assert {"python", "torch", "numpy", "cuda_available"} <= set(info)


def test_git_revision_is_a_string_or_none() -> None:
    revision = git_revision()
    assert revision is None or isinstance(revision, str)

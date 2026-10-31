"""Configuration loading: defaults, interpolation, overrides, validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from melba.config import Config, ConfigError, load_config, parse_set_args


def test_defaults_are_merged_and_references_resolved(base_config: Config, repo_root: Path) -> None:
    assert base_config["experiment.name"] == "cvc_clinicdb_unet_resnet50"
    # from base.yaml
    assert base_config["seed"] == 1234
    assert base_config["metrics.empty_policy"] == "nan"
    # ${experiment.name} interpolation
    assert base_config["evaluation.output_dir"] == "results/cvc_clinicdb_unet_resnet50/evaluation"
    # `auto` root anchors at the repository root
    assert base_config.root == repo_root
    assert base_config.resolve(base_config.require("dataset.images_dir")) == (
        repo_root / "cvc_clinicdb" / "original"
    ).resolve()


def test_dotted_access_and_mapping_protocol(base_config: Config) -> None:
    assert base_config.get("does.not.exist") is None
    assert base_config.get("does.not.exist", 5) == 5
    with pytest.raises(KeyError):
        base_config["does.not.exist"]
    with pytest.raises(ConfigError):
        base_config.require("does.not.exist")
    assert "seed" in base_config
    assert isinstance(dict(base_config)["experiment"], dict)
    # returning copies protects the config from accidental mutation
    metrics = base_config["metrics"]
    metrics["names"].append("bogus")
    assert "bogus" not in base_config["metrics"]["names"]


def test_fingerprint_is_stable_and_sensitive(base_config: Config) -> None:
    reloaded = load_config(base_config.path)
    assert base_config.fingerprint == reloaded.fingerprint
    changed = base_config.with_overrides({"seed": 99})
    assert changed.fingerprint != base_config.fingerprint
    assert changed["seed"] == 99
    assert base_config["seed"] == 1234


def test_overrides_from_mapping_and_cli_strings(cvc_config_path: Path) -> None:
    config = load_config(cvc_config_path, {"dataset.image_size": [128, 128], "seed": 7})
    assert config.get("dataset.image_size") == [128, 128]
    assert config["seed"] == 7

    config = load_config(cvc_config_path, ["dataset.image_size=[96, 96]", "device=cpu"])
    assert config.get("dataset.image_size") == [96, 96]
    assert config["device"] == "cpu"


def test_parse_set_args_types() -> None:
    parsed = parse_set_args(["a.b=1", "c=true", "d=null", "e=[1, 2]", "f=text"])
    assert parsed == {"a.b": 1, "c": True, "d": None, "e": [1, 2], "f": "text"}
    with pytest.raises(ConfigError):
        parse_set_args(["missing_equals"])


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_circular_defaults(tmp_path: Path) -> None:
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text("defaults: [b.yaml]\nseed: 1\n", encoding="utf-8")
    second.write_text("defaults: [a.yaml]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Circular"):
        load_config(first)


def test_unknown_reference_is_reported(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(
        "seed: 1\npaths: {root: auto}\nexperiment: {name: x}\nout: ${missing.key}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Unknown reference"):
        load_config(config_file)


def test_validation_errors(tmp_path: Path) -> None:
    bad_fractions = tmp_path / "fractions.yaml"
    bad_fractions.write_text(
        "seed: 1\npaths: {root: auto}\nexperiment: {name: x}\n"
        "dataset:\n  split:\n    fractions: {train: 0.8, val: 0.3, test: 0.1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(bad_fractions)

    negative_seed = tmp_path / "seed.yaml"
    negative_seed.write_text(
        "seed: -1\npaths: {root: auto}\nexperiment: {name: x}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="non-negative"):
        load_config(negative_seed)


def test_all_shipped_configs_load(repo_root: Path) -> None:
    config_files = sorted((repo_root / "configs").rglob("*.yaml"))
    assert config_files, "no configs found"
    for path in config_files:
        if path.name == "base.yaml":
            continue
        config = load_config(path)
        assert config.require("experiment.name")
        assert config.require("tta.methods")

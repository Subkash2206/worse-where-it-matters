"""Detection-safe decision rules: parsing, mask semantics and the runner integration.

The rules must (a) never change a non-empty TTA mask, (b) rescue only what a view actually
saw, and (c) cost no extra forward passes.  The tests below pin all three properties, the
last one by counting the model calls of a full evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from melba.config import ConfigError
from melba.eval import evaluate, read_long_csv
from melba.inference import (
    DETECTION_RULES,
    TTAMethod,
    apply_rule,
    parse_rule,
    rule_name,
)
from melba.metrics import compute_case_metrics


def _probability(shape: tuple[int, ...] = (8, 12)) -> np.ndarray:
    return np.zeros((1, *shape), dtype=np.float32)


# --------------------------------------------------------------------------- parsing


def test_parse_rule_round_trip() -> None:
    assert parse_rule("anyview") == ("anyview", None)
    for percent in (20, 30, 40):
        family, threshold = parse_rule(f"lowdet_{percent:02d}")
        assert family == "lowdet"
        assert threshold == pytest.approx(percent / 100.0)
        assert rule_name(family, threshold) == f"lowdet_{percent:02d}"
    assert set(DETECTION_RULES) == {"anyview", "lowdet_20", "lowdet_30", "lowdet_40"}


@pytest.mark.parametrize("text", ["", "any", "lowdet", "lowdet_100", "low", "1.2", "lowdet_0"])
def test_parse_rule_rejects_unknown_tokens(text: str) -> None:
    with pytest.raises(ValueError, match="Unknown detection rule|threshold must be"):
        parse_rule(text)


def test_parse_rule_normalises_a_short_threshold() -> None:
    # ``lowdet_5`` is accepted (0.05) but is not canonical: a method name must use
    # ``lowdet_05``, which TTAMethod enforces for derived entries
    assert parse_rule("lowdet_5") == ("lowdet", pytest.approx(0.05))
    assert rule_name(*parse_rule("lowdet_5")) == "lowdet_05"


def test_rule_name_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="Unknown detection rule family"):
        rule_name("threshold", 0.3)


# ------------------------------------------------------------------- mask semantics


def test_anyview_keeps_a_non_empty_mean_mask() -> None:
    mean = _probability()
    mean[0, 2:5, 2:5] = 0.9
    views = np.zeros((8, 1, 8, 12), dtype=np.float32)
    views[:, 0, 2:5, 2:5] = 0.9  # every view agrees
    mask = apply_rule("anyview", mean, views)
    assert mask.sum() == 9
    np.testing.assert_array_equal(mask, mean[0] >= 0.5)


def test_anyview_falls_back_to_the_identity_mask() -> None:
    # the mean and every view mask are empty at 0.5, but the model drove the identity view
    # over the threshold: that is the detection the rule must keep
    mean = _probability()
    mean[0] = 0.45
    views = np.full((8, 1, 8, 12), 0.45, dtype=np.float32)
    views[0, 0, 3:6, 4:7] = 0.8
    mask = apply_rule("anyview", mean, views)
    assert mask.sum() == 9
    assert not (mean[0] >= 0.5).any()


def test_anyview_falls_back_to_the_union_of_view_masks() -> None:
    mean = _probability()
    views = np.full((8, 1, 8, 12), 0.45, dtype=np.float32)
    views[3, 0, 1:3, 1:3] = 0.9
    views[5, 0, 5:7, 8:10] = 0.9
    mask = apply_rule("anyview", mean, views, identity_index=0)
    assert mask.sum() == 8
    assert mask[1:3, 1:3].all() and mask[5:7, 8:10].all()


def test_anyview_returns_an_empty_mask_when_nothing_fires() -> None:
    mean = _probability()
    views = np.full((4, 1, 8, 12), 0.2, dtype=np.float32)
    assert not apply_rule("anyview", mean, views).any()


def test_lowdet_rescues_only_a_high_peak() -> None:
    mean = _probability()
    mean[0, 4, 4] = 0.42  # peak above 0.4 but below the 0.5 decision threshold
    views = np.zeros((8, 1, 8, 12), dtype=np.float32)
    rescued = apply_rule("lowdet_40", mean, views)
    assert rescued[4, 4] and rescued.sum() == 1
    assert apply_rule("lowdet_30", mean, views).sum() == 1
    # a peak below the rule threshold leaves the plain (empty) mask alone
    mean_small = _probability()
    mean_small[0, 4, 4] = 0.15
    assert not apply_rule("lowdet_20", mean_small, views).any()


def test_lowdet_never_changes_a_non_empty_mask() -> None:
    mean = _probability()
    mean[0, 2:5, 2:5] = 0.55
    views = np.zeros((4, 1, 8, 12), dtype=np.float32)
    plain = mean[0] >= 0.5
    for rule in ("lowdet_20", "lowdet_30", "lowdet_40"):
        np.testing.assert_array_equal(apply_rule(rule, mean, views), plain)


def test_rule_ignores_the_channel_axis() -> None:
    mean = _probability()
    mean[0, 1, 1] = 0.9
    views = np.zeros((4, 1, 8, 12), dtype=np.float32)
    mask = apply_rule("anyview", mean[0], views[:, 0])
    assert mask.shape == (8, 12) and mask[1, 1]


# ------------------------------------------------------------------ metric override


def test_pred_mask_overrides_the_probability_threshold() -> None:
    probability = np.zeros((1, 8, 12), dtype=np.float32)
    probability[0, 2:5, 2:5] = 0.9
    target = np.zeros((8, 12), dtype=np.float32)
    target[2:5, 2:5] = 1.0
    decided = compute_case_metrics(probability[0], target, spacing=(1.0, 1.0))
    empty_decision = np.zeros((8, 12), dtype=bool)
    metrics = compute_case_metrics(
        probability[0], target, spacing=(1.0, 1.0), pred_mask=empty_decision
    )
    assert decided["dice"] == pytest.approx(1.0)
    assert metrics["empty_flag"] == "pred_empty"
    assert metrics["dice"] == 0.0  # empty prediction on a non-empty reference


# ---------------------------------------------------------------- config validation


IDENTITY = {"name": "identity", "ops": []}
HFLIP = {"name": "hflip", "ops": [{"type": "flip", "axes": ["w"]}]}


def _method_specs(methods: list[dict]) -> dict:
    return {"tta.methods": methods}


def test_derived_method_inherits_views_and_aggregator(toy_dataset_config) -> None:
    from melba.inference import methods_from_config

    config = toy_dataset_config.with_overrides(
        _method_specs(
            [
                {"name": "tta2", "views": [IDENTITY, HFLIP]},
                {"name": "tta2_anyview", "derived_from": "tta2", "rule": "anyview"},
            ]
        )
    )
    parent, derived = methods_from_config(config)
    assert derived.derived_from == "tta2"
    assert derived.rule == "anyview"
    assert derived.views == parent.views
    assert derived.aggregator == parent.aggregator == "mean"
    assert derived.is_derived and not parent.is_derived


def test_derived_method_order_does_not_matter(toy_dataset_config) -> None:
    """A rule may be written before its parent; the list returned puts parents first."""
    from melba.inference import methods_from_config

    config = toy_dataset_config.with_overrides(
        _method_specs(
            [
                {"name": "tta2_anyview", "derived_from": "tta2", "rule": "anyview"},
                {"name": "tta2", "views": [IDENTITY, HFLIP]},
            ]
        )
    )
    methods = methods_from_config(config)
    assert [method.name for method in methods] == ["tta2", "tta2_anyview"]
    assert methods[1].views == methods[0].views


def test_derived_method_with_unknown_parent_fails(toy_dataset_config) -> None:
    from melba.inference import methods_from_config

    config = toy_dataset_config.with_overrides(
        _method_specs(
            [
                {"name": "tta2", "views": [IDENTITY, HFLIP]},
                {"name": "tta8_anyview", "derived_from": "tta8", "rule": "anyview"},
            ]
        )
    )
    with pytest.raises(ConfigError, match="unknown"):
        methods_from_config(config)


def test_derived_method_rejects_its_own_views(toy_dataset_config) -> None:
    from melba.inference import methods_from_config

    config = toy_dataset_config.with_overrides(
        _method_specs(
            [
                {"name": "tta2", "views": [IDENTITY, HFLIP]},
                {
                    "name": "tta2_anyview",
                    "derived_from": "tta2",
                    "rule": "anyview",
                    "views": [IDENTITY],
                },
            ]
        )
    )
    with pytest.raises(ConfigError, match="may not declare its own views"):
        methods_from_config(config)


@pytest.mark.parametrize(
    ("name", "rule", "message"),
    [
        ("tta2_mask", "anyview", "must be named"),
        ("tta2_anyview", "lowdet_30", "must be named"),
        ("tta2_lowdet_3", "lowdet_3", "write it as"),
    ],
)
def test_derived_method_name_must_encode_the_rule(name, rule, message, toy_dataset_config) -> None:
    from melba.inference import methods_from_config

    config = toy_dataset_config.with_overrides(
        _method_specs(
            [
                {"name": "tta2", "views": [IDENTITY, HFLIP]},
                {"name": name, "derived_from": "tta2", "rule": rule},
            ]
        )
    )
    with pytest.raises((ConfigError, ValueError), match=message):
        methods_from_config(config)


def test_ttamethod_rejects_a_rule_without_a_parent() -> None:
    from melba.transforms import build_views

    views = tuple(build_views([IDENTITY]))
    with pytest.raises(ValueError, match="without 'derived_from'"):
        TTAMethod(name="tta1_lowdet_30", views=views, rule="lowdet_30")



# ------------------------------------------------------------- runner integration


class PlateauModel(nn.Module):
    """Smooth, parameter-free model that counts its forward passes.

    It puts a plateau just under the 0.5 decision threshold in the image centre, so the
    plain TTA mask is empty while the lowdet rules must rescue it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        # background p = 0.047, plateau p = 0.455: below the 0.5 decision threshold, but
        # above the 0.20/0.30/0.40 lower thresholds of the lowdet rules
        signal = torch.full_like(x[:, :1], -3.0)
        height, width = signal.shape[-2:]
        signal[
            :, :, height // 3 : 2 * height // 3, width // 3 : 2 * width // 3
        ] = -0.18
        return signal


def _derived_config(toy_dataset_config):
    return toy_dataset_config.with_overrides(
        {
            "metrics.names": ["dice", "boundary_f1", "hd95"],
            "evaluation.timing": False,
            "evaluation.save_probability_maps": False,
            **_method_specs(
                [
                    {"name": "baseline", "views": [IDENTITY]},
                    {"name": "tta2", "views": [IDENTITY, HFLIP]},
                    {"name": "tta4", "views": [IDENTITY, HFLIP]},
                    {"name": "tta8", "views": [IDENTITY, HFLIP]},
                    {"name": "tta8_anyview", "derived_from": "tta8", "rule": "anyview"},
                    {"name": "tta8_lowdet_30", "derived_from": "tta8", "rule": "lowdet_30"},
                ]
            ),
        }
    )


def test_derived_rules_cost_no_forward_passes(toy_dataset_config, tmp_path) -> None:
    config = _derived_config(toy_dataset_config)
    model = PlateauModel()
    result = evaluate(config, model=model, limit=2, output_dir=tmp_path / "rules", progress=False)
    frame = read_long_csv(result.per_case_csv)
    assert frame["method"].nunique() == 6
    n_cases = frame["case_id"].nunique()
    # 1 + 2 + 2 + 2 forward passes for the four parents, nothing for the two rules
    assert model.calls == n_cases * 7


def test_plain_tta8_misses_and_the_rules_rescue(toy_dataset_config, tmp_path) -> None:
    config = _derived_config(toy_dataset_config)
    result = evaluate(
        config, model=PlateauModel(), limit=2, output_dir=tmp_path / "rules2", progress=False
    )
    frame = read_long_csv(result.per_case_csv)
    flags = frame.drop_duplicates(["case_id", "method"]).set_index(["case_id", "method"])
    tta8 = flags.xs("tta8", level="method")["empty_flag"]
    lowdet = flags.xs("tta8_lowdet_30", level="method")["empty_flag"]
    anyview = flags.xs("tta8_anyview", level="method")["empty_flag"]
    assert set(tta8) == {"pred_empty"}
    assert set(lowdet) == {"both_nonempty"}  # the peak is rescued
    # every view sees the same plateau, and none of them reaches 0.5: nothing to rescue
    assert set(anyview) == {"pred_empty"}


def test_rule_dice_matches_an_independent_mask_computation(toy_dataset_config, tmp_path) -> None:
    """The rule's mask is what the runner scores, not the plain threshold of the mean map.

    The check is non-circular: the *parent's* saved mean map is re-decided here with
    ``apply_rule`` and the resulting mask must reproduce both the saved rule mask and the
    Dice the runner wrote.
    """
    from melba.data import build_dataset

    config = _derived_config(toy_dataset_config).with_overrides(
        {
            "evaluation.save_probability_maps": True,
            "evaluation.probability_map_methods": ["tta8", "tta8_lowdet_30"],
        }
    )
    result = evaluate(
        config, model=PlateauModel(), limit=1, output_dir=tmp_path / "rules3", progress=False
    )
    frame = read_long_csv(result.per_case_csv)
    sample = build_dataset(config, split="test")[0]
    target = np.asarray(sample.get("mask_native", sample["mask"])[0])

    seed = int(config["seed"])
    map_dir = (tmp_path / "rules3" / "maps" / f"fold0_seed{seed}")
    mean_map = np.load(map_dir / f"{sample['case_id']}__tta8__mean.npy").astype(np.float32)
    rule_mask = np.load(map_dir / f"{sample['case_id']}__tta8_lowdet_30__mean.npy") > 0.5
    expected_mask = apply_rule("lowdet_30", mean_map, mean_map[None])
    np.testing.assert_array_equal(rule_mask, expected_mask)

    expected = compute_case_metrics(
        mean_map, target, spacing=(1.0, 1.0), pred_mask=expected_mask
    )
    rows = frame.loc[
        (frame["case_id"] == sample["case_id"])
        & (frame["method"] == "tta8_lowdet_30")
        & (frame["metric"] == "dice")
    ]
    assert len(rows) == 1
    assert rows["value"].iloc[0] == pytest.approx(expected["dice"], abs=1e-6)
    # the rule produced a real mask where the plain 0.5 threshold had none
    flags = frame.drop_duplicates(["case_id", "method"]).set_index(["case_id", "method"])
    assert flags.xs("tta8", level="method").loc[sample["case_id"], "empty_flag"] == "pred_empty"
    assert flags.xs("tta8_lowdet_30", level="method").loc[sample["case_id"], "empty_flag"] != (
        "pred_empty"
    )


def test_metadata_stamps_the_rule_and_its_parent(toy_dataset_config, tmp_path) -> None:
    import json

    config = _derived_config(toy_dataset_config)
    result = evaluate(
        config, model=PlateauModel(), limit=1, output_dir=tmp_path / "rules4", progress=False
    )
    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    rules = {entry["name"]: entry for entry in metadata["methods"] if entry["derived_from"]}
    assert set(rules) == {"tta8_anyview", "tta8_lowdet_30"}
    assert rules["tta8_lowdet_30"]["rule"] == "lowdet_30"
    assert rules["tta8_lowdet_30"]["derived_from"] == "tta8"


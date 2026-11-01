"""Cross-model tables and the residual-error test, on synthetic per-case frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melba.eval.compare import (
    ModelRuns,
    cluster_bootstrap_correlation,
    deltas_table,
    gain_difference_table,
    method_table,
    missed_detection_table,
    paired_swap_null,
    residual_error_table,
    tail_table_for_models,
)
from melba.eval.report import spearman

N_VIDEOS = 6
N_FOLDS = 2
N_UNITS = N_VIDEOS * N_FOLDS  # every video is a test unit exactly once


def _case_frames(gain: float, *, seed: int, missed: set[str] | None = None) -> pd.DataFrame:
    """Two folds of per-case rows where the TTA gain is larger on worse baselines."""
    missed = missed or set()
    rows = []
    for fold in range(N_FOLDS):
        for index in range(N_VIDEOS):
            case = f"c{fold}_{index}"
            group = f"v{fold}_{index}"
            baseline_dice = 0.40 + 0.08 * index
            baseline_hd95 = 48.0 - 6.0 * index
            delta_dice = gain * (0.95 - baseline_dice)
            delta_hd95 = -gain * baseline_hd95 * 0.2
            for method, dice, hd95 in (
                ("baseline", baseline_dice, baseline_hd95),
                ("tta4", baseline_dice + delta_dice, baseline_hd95 + delta_hd95),
            ):
                empty_flag = "pred_empty" if (method == "tta4" and case in missed) else "both_nonempty"
                values = {"dice": dice, "hd95": hd95}
                if empty_flag == "pred_empty":
                    values["dice"] = 0.0
                    values["hd95"] = np.nan
                for metric, value in values.items():
                    rows.append(
                        {
                            "run_id": f"run-{seed}",
                            "seed": seed,
                            "fold": fold,
                            "split": "test",
                            "case_id": case,
                            "group_id": group,
                            "method": method,
                            "metric": metric,
                            "value": float(value),
                            "empty_flag": empty_flag,
                            "missed_detection": int(empty_flag == "pred_empty"),
                        }
                    )
    return pd.DataFrame(rows)


def _model(name: str, gain: float, *, missed: set[str] | None = None) -> ModelRuns:
    frames = {seed: [_case_frames(gain, seed=seed, missed=missed)] for seed in (0, 1, 2)}
    ensemble = _case_frames(gain * 1.1, seed=0).copy()
    ensemble = ensemble.loc[ensemble["method"] == "tta4"].assign(
        method="ensemble3", seed=-1, run_id="ensemble-baseline"
    )
    return ModelRuns(name=name, frames_by_seed=frames, ensemble_frame=ensemble)


def test_method_table_lists_both_models_side_by_side() -> None:
    models = [_model("unet", 1.0), _model("swin", 0.5)]
    table = method_table(models)
    assert set(table["model"]) == {"unet", "swin"}
    assert set(table["method"]) == {"baseline", "tta4", "ensemble3"}
    assert set(table["metric"]) == {"dice", "hd95"}
    # one value per video, no frame-level double counting
    assert set(table["n_units"]) == {N_UNITS}
    assert set(table["n_cases"]) == {N_UNITS}

    dice = table.loc[table["metric"] == "dice"].set_index(["model", "method"])["mean"]
    assert dice[("unet", "tta4")] > dice[("unet", "baseline")]
    assert dice[("unet", "tta4")] > dice[("swin", "tta4")]


def test_deltas_table_reports_paired_gain_per_model() -> None:
    table = deltas_table([_model("unet", 1.0), _model("swin", 0.4)], n_bootstrap=200)
    assert set(table["model"]) == {"unet", "swin"}
    assert set(table["treatment"]) == {"tta4", "ensemble3"}
    assert {"delta_mean", "ci_low", "ci_high", "p_value", "p_value_adjusted"} <= set(table.columns)
    dice = table.loc[table["metric"] == "dice"]
    assert (dice["delta_mean"] > 0).all()
    # `preferred` names the winning method, so it echoes the treatment here
    assert set(dice["preferred"]) == set(dice["treatment"])


def test_missed_detection_table_counts_misses_per_model() -> None:
    table = missed_detection_table([_model("unet", 1.0), _model("swin", 1.0, missed={"c0_0"})])
    unet = table.loc[(table["model"] == "unet") & (table["method"] == "tta4")].iloc[0]
    swin = table.loc[(table["model"] == "swin") & (table["method"] == "tta4")].iloc[0]
    assert unet["n_missed"] == 0
    assert swin["n_missed"] == 1
    assert swin["missed_rate"] == pytest.approx(1 / N_UNITS)
    assert swin["nonempty_to_empty"] == 1


def test_tail_table_reports_failure_rates_per_model() -> None:
    table = tail_table_for_models([_model("unet", 1.0), _model("swin", 1.0, missed={"c0_0"})])
    assert set(table["model"]) == {"unet", "swin"}
    assert {"hd95_p95", "hd95_failure_rate", "dice_failure_rate"} <= set(table.columns)
    unet = table.loc[(table["model"] == "unet") & (table["method"] == "tta4")].iloc[0]
    swin = table.loc[(table["model"] == "swin") & (table["method"] == "tta4")].iloc[0]
    # an empty prediction makes HD95 undefined and therefore counts as a failure
    assert unet["hd95_failure_rate"] == 0.0
    assert swin["hd95_failure_rate"] == pytest.approx(1 / N_UNITS)
    assert np.isfinite(swin["delta_hd95_failure_rate_estimate"])


def test_residual_error_test_is_negative_when_gains_track_baseline_error() -> None:
    table = residual_error_table(
        [_model("unet", 1.0), _model("swin", 0.4)], n_bootstrap=200, n_permutations=500
    )
    assert set(table["scope"]) == {
        "model:unet",
        "model:swin",
        "pooled_models",
    }
    assert set(table["metric"]) == {"dice", "hd95"}
    per_model = table.loc[table["scope"] != "pooled_models"]
    # synthetic data: the gain grows as the control gets worse, for both metrics
    assert (per_model["rho_oldham"] < 0).all()
    assert (per_model["ci_high_oldham"] < 0).all()
    assert (per_model["p_permutation_oldham"] < 0.05).all()
    assert (table["rho_naive"] < 0).all()  # kept as the secondary, biased column
    assert (table["ci_low_naive"] <= table["rho_naive"]).all()
    assert (table["rho_naive"] <= table["ci_high_naive"]).all()
    assert (table["n_permutations"] == 500).all()

    pooled = table.loc[table["scope"] == "pooled_models"].set_index("metric")
    # both models contribute their cases to the pooled analysis
    assert pooled.loc["dice", "n_cases"] == 2 * N_UNITS
    assert pooled.loc["dice", "n_clusters"] == N_UNITS
    # stacking two models with different effect sizes attenuates the rank correlation,
    # so the pooled rows are only required to keep the sign
    assert (pooled["rho_oldham"] < 0).all()


def _latent_frames(
    *, seed: int = 0, effect: float = 0.0, n_videos: int = 24, n_frames: int = 4
) -> pd.DataFrame:
    """Two methods that measure the same latent quality, with or without a real gain.

    ``effect=0`` means TTA has **no true effect**: both predictions are the latent value
    plus independent noise of equal variance, which is exactly the situation in which the
    naive residual-error rho is negative for purely statistical reasons.  ``effect>0``
    moves each prediction towards a ceiling of ``0.8``, so the gain is positive on
    average *and* larger for the videos the control handled badly.
    """
    rng = np.random.default_rng(2024 + seed)
    videos_per_fold = n_videos // N_FOLDS
    rows = []
    for fold in range(N_FOLDS):
        for video in range(videos_per_fold):
            group = f"g{fold}_{video}"
            latent = float(rng.normal(0.6, 0.12))
            for frame in range(n_frames):
                case = f"k{fold}_{video}_{frame}"
                control = latent + float(rng.normal(0.0, 0.05))
                treatment = latent + float(rng.normal(0.0, 0.05)) + effect * (0.8 - latent)
                for method, value in (("baseline", control), ("tta4", treatment)):
                    for metric, metric_value in (
                        ("dice", value),
                        ("hd95", (0.95 - value) * 40.0),
                    ):
                        rows.append(
                            {
                                "run_id": f"run-{seed}",
                                "seed": seed,
                                "fold": fold,
                                "split": "test",
                                "case_id": case,
                                "group_id": group,
                                "method": method,
                                "metric": metric,
                                "value": float(metric_value),
                                "empty_flag": "both_nonempty",
                                "missed_detection": 0,
                            }
                        )
    return pd.DataFrame(rows)


def _latent_model(name: str, *, effect: float, seed: int = 0) -> ModelRuns:
    frame = _latent_frames(seed=seed, effect=effect)
    return ModelRuns(name=name, frames_by_seed={seed: [frame]})


def test_zero_true_effect_is_not_significant_for_oldham_or_the_swap_null() -> None:
    table = residual_error_table(
        [_latent_model("unet", effect=0.0)], n_bootstrap=200, n_permutations=2000
    ).set_index("metric")

    assert set(table.index) == {"dice", "hd95"}
    # the naive statistic is negative for a purely statistical reason ...
    assert (table["rho_naive"] < -0.2).all()
    # ... while Oldham's correction and the paired-swap null stay centred
    assert (table["rho_oldham"].abs() < 0.15).all()
    assert (table["null_mean_oldham"].abs() < 0.15).all()
    assert (table["p_oldham"] > 0.05).all()
    assert (table["p_permutation_oldham"] > 0.05).all()
    assert (table["ci_low_oldham"] < 0.0).all()
    assert (table["ci_high_oldham"] > 0.0).all()
    # a single model has no pooled row: pooling needs at least two
    assert set(table.columns) >= {"rho_oldham", "rho_naive", "p_permutation_naive"}


def test_oldham_detects_a_real_gradient() -> None:
    table = residual_error_table(
        [_latent_model("unet", effect=0.5)], n_bootstrap=200, n_permutations=2000
    ).set_index("metric")
    assert (table["rho_oldham"] < -0.3).all()
    assert (table["ci_high_oldham"] < 0.0).all()
    assert (table["p_oldham"] < 0.001).all()
    assert (table["p_permutation_oldham"] < 0.01).all()
    assert (table["p_bootstrap_oldham"] < 0.01).all()


def test_gain_difference_table_pairs_models_by_video() -> None:
    # the same latent data with a stronger effect in one model: the gains differ
    models = [
        _latent_model("unet", effect=0.5),
        _latent_model("swin", effect=0.0),
    ]
    table = gain_difference_table(models, n_bootstrap=200).set_index("metric")
    assert set(table.index) == {"dice", "hd95"}
    dice = table.loc["dice"]
    assert dice["reference"] == "unet" and dice["compared"] == "swin"
    assert dice["n_units"] == 24  # videos, not frames
    # the treating model has the larger (positive) Dice gain
    assert dice["delta_gain"] < 0.0
    assert dice["favours"] == "unet"
    assert dice["p_value"] < 0.05
    assert dice["ci_low"] <= dice["delta_gain"] <= dice["ci_high"]
    # a "gain" is treatment - control, so it is negative for HD95 in both models
    assert table.loc["hd95", "gain_reference"] < 0.0
    assert table.loc["hd95", "delta_gain"] > 0.0  # the treated model improves more

    identical = gain_difference_table([models[0], models[0]], n_bootstrap=100)
    assert np.allclose(identical["delta_gain"], 0.0)
    assert set(identical["favours"]) == {"tie"}
    assert gain_difference_table(models[:1]).empty

    rng = np.random.default_rng(7)
    control = 0.6 + rng.normal(scale=0.1, size=40)
    treatment = control + rng.normal(scale=0.05, size=40)
    groups = np.repeat([f"v{index}" for index in range(10)], 4)
    result = paired_swap_null(control, treatment, groups, n_permutations=300, seed=3)

    delta = treatment - control
    assert result["naive_observed"] == pytest.approx(spearman(control, delta)[0], abs=1e-9)
    assert result["oldham_observed"] == pytest.approx(
        spearman((control + treatment) / 2.0, delta)[0], abs=1e-9
    )
    assert result["n_blocks"] == 10
    assert result["n_permutations"] == 300
    assert abs(result["oldham_null_mean"]) < 0.2
    assert 0.0 <= result["oldham_p_permutation"] <= 1.0
    # a NaN case is dropped, not propagated
    with_nan = control.astype(float).copy()
    with_nan[0] = np.nan
    dropped = paired_swap_null(with_nan, treatment, groups, n_permutations=50, seed=1)
    assert np.isfinite(dropped["oldham_observed"])


def test_cluster_bootstrap_correlation_matches_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    baseline = np.linspace(1.0, 10.0, 40)
    delta = -(baseline**2) * 0.1 + rng.normal(scale=0.5, size=baseline.size)
    groups = np.repeat([f"v{index}" for index in range(8)], 5)

    result = cluster_bootstrap_correlation(
        baseline, delta, groups, n_bootstrap=200, ci=0.95, seed=3
    )
    assert result["spearman_rho"] == pytest.approx(spearman(baseline, delta)[0])
    assert result["n_cases"] == 40
    assert result["n_clusters"] == 8
    assert result["ci_low"] <= result["ci_high"]
    assert 0.0 <= result["p_bootstrap"] <= 1.0
    assert result["spearman_rho"] < 0

    # pairs with a missing value are dropped, not silently coerced
    with_nan = delta.copy()
    with_nan[0] = np.nan
    dropped = cluster_bootstrap_correlation(baseline, with_nan, groups, n_bootstrap=50, seed=1)
    assert dropped["n_cases"] == 39

    # a constant axis has an undefined correlation and must not raise
    flat = cluster_bootstrap_correlation(np.ones(10), np.arange(10.0), np.repeat("v", 10))
    assert np.isnan(flat["spearman_rho"])
    assert np.isnan(flat["ci_low"])


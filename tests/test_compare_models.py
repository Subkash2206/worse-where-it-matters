"""Cross-model tables and the residual-error test, on synthetic per-case frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melba.eval.compare import (
    ModelRuns,
    cluster_bootstrap_correlation,
    deltas_table,
    method_table,
    missed_detection_table,
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
        [_model("unet", 1.0), _model("swin", 0.4)], n_bootstrap=200
    )
    assert set(table["scope"]) == {
        "model:unet",
        "model:swin",
        "pooled_models",
    }
    assert set(table["metric"]) == {"dice", "hd95"}
    # synthetic data: the gain grows as the baseline gets worse, for both metrics
    assert (table["spearman_rho"] < 0).all()
    assert (table["ci_high"] < 0).all()
    assert (table["ci_low"] <= table["spearman_rho"]).all()
    assert (table["spearman_rho"] <= table["ci_high"]).all()

    pooled = table.loc[table["scope"] == "pooled_models"].set_index("metric")
    # both models contribute their cases to the pooled analysis
    assert pooled.loc["dice", "n_cases"] == 2 * N_UNITS
    assert pooled.loc["dice", "n_clusters"] == N_UNITS


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


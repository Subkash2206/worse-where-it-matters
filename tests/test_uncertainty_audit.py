"""Uncertainty-descriptor audit: ROI semantics, AUROC definitions, CIs and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melba.eval.uncertainty_audit import (
    auroc_ci,
    auroc_difference,
    auroc_nan_bounds,
    build_case_table,
    map_statistics,
    roi_diagnostics,
    run_uncertainty_audit,
)


def _map(
    block: tuple[int, int, int, int] | None,
    *,
    foreground: float = 0.9,
    background: float = 0.0,
    size: int = 16,
) -> np.ndarray:
    probability = np.full((size, size), background, dtype=np.float32)
    if block is not None:
        top, bottom, left, right = block
        probability[top:bottom, left:right] = foreground
    return probability


def test_map_statistics_empty_prediction_has_no_roi() -> None:
    stats = map_statistics(_map(None, foreground=0.9, background=0.1))
    assert stats.area_px == 0
    assert stats.roi_px == 0
    assert np.isnan(stats.entropy_roi)
    assert np.isnan(stats.variance_roi)
    assert np.isfinite(stats.entropy_image)
    assert np.isnan(stats.confidence)


def test_map_statistics_tiny_prediction_has_near_zero_entropy() -> None:
    stats = map_statistics(_map((4, 5, 4, 5), foreground=0.99))
    assert stats.area_px == 1
    # the ROI is the pixel plus the 3-px band, and it is confidently painted
    assert stats.roi_px > stats.area_px
    assert 0.0 <= stats.entropy_roi < 0.05
    assert stats.confidence == pytest.approx(0.99, abs=1e-6)


def test_map_statistics_variance_across_seeds_is_roi_restricted() -> None:
    first = _map((6, 14, 6, 14), foreground=0.9)
    second = _map((6, 14, 6, 14), foreground=0.6)
    stats = map_statistics(np.mean([first, second], axis=0), [first, second])
    assert stats.variance_image > 0
    assert stats.variance_roi > stats.variance_image  # disagreement concentrates in the lesion


def test_auroc_ci_reports_used_and_undefined_cases() -> None:
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, np.nan]
    labels = [False, False, True, True, True, True]
    groups = ["v0", "v1", "v2", "v0", "v1", "v2"]
    result = auroc_ci(scores, labels, groups, n_bootstrap=200)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["n_used"] == 5
    assert result["n_undefined"] == 1
    assert result["n_failures"] == 3  # the undefined case is dropped from both classes
    assert result["ci_low"] <= 1.0 <= result["ci_high"]


def test_auroc_nan_bounds_bracket_the_dropping_estimate() -> None:
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, np.nan]
    labels = [False, False, True, True, True, True]
    certain, uncertain = auroc_nan_bounds(scores, labels)
    dropped = auroc_ci(scores, labels, list(range(6)), n_bootstrap=50)["auroc"]
    assert certain <= dropped <= uncertain
    assert certain < 1.0 and uncertain == pytest.approx(1.0)


def test_auroc_difference_detects_a_better_predictor() -> None:
    labels = [False, False, False, False, True, True, True, True]
    strong = [0.1, 0.2, 0.15, 0.25, 0.8, 0.9, 0.85, 0.95]
    weak = [0.4, 0.5, 0.3, 0.45, 0.5, 0.55, 0.4, 0.6]
    groups = ["v0", "v0", "v1", "v1", "v2", "v2", "v3", "v3"]
    result = auroc_difference(strong, weak, labels, groups, n_bootstrap=200)
    assert result["delta_auroc"] > 0.1
    assert result["ci_low"] >= 0.0
    assert np.isfinite(result["ci_high"])
    assert result["ci_low"] <= result["ci_high"]
    identical = auroc_difference(strong, strong, labels, groups, n_bootstrap=50)
    assert identical["delta_auroc"] == pytest.approx(0.0)
    assert identical["p_value"] == pytest.approx(1.0)


def _audit_fixture(tmp_path):
    """Two seeds x two methods of 16x16 maps for four cases (one missed, one tiny)."""
    from types import SimpleNamespace

    cases = {
        "c_missed": (_map(None, foreground=0.9, background=0.1), _map(None, foreground=0.9, background=0.1)),
        "c_tiny": (_map((4, 5, 4, 5), foreground=0.99), _map((4, 5, 4, 5), foreground=0.9)),
        "c_clean1": (_map((2, 12, 2, 12), foreground=0.95), _map((2, 12, 2, 12), foreground=0.95)),
        "c_clean2": (_map((3, 11, 3, 11), foreground=0.9), _map((3, 11, 3, 11), foreground=0.9)),
    }
    runs = []
    for seed in (0, 1):
        map_dir = tmp_path / f"maps/fold0_seed{seed}"
        map_dir.mkdir(parents=True)
        for case_id, (tta4_map, baseline_map) in cases.items():
            np.save(map_dir / f"{case_id}__tta4__mean.npy", tta4_map.astype(np.float16))
            np.save(map_dir / f"{case_id}__baseline__mean.npy", baseline_map.astype(np.float16))
        runs.append(SimpleNamespace(fold=0, seed=seed, map_dir=map_dir))

    dice_by_method = {
        "tta4": {"c_missed": 0.0, "c_tiny": 0.3, "c_clean1": 0.9, "c_clean2": 0.8},
        "baseline": {"c_missed": 0.0, "c_tiny": 0.55, "c_clean1": 0.9, "c_clean2": 0.8},
    }
    missed = {"c_missed": 1, "c_tiny": 0, "c_clean1": 0, "c_clean2": 0}
    rows = []
    for index, case_id in enumerate(missed):
        for method in ("tta4", "baseline"):
            empty = case_id == "c_missed"
            values = {
                "dice": dice_by_method[method][case_id],
                "hd95": 200.0 if empty else (60.0 if case_id == "c_tiny" else 5.0),
                "entropy_mean": 0.2,
                "entropy_mean_roi": float("nan") if empty else 0.1,
                "variance_mean": 0.01,
                "variance_mean_roi": float("nan") if empty else 0.02,
                "view_disagreement": 0.0 if empty else 0.05,
            }
            for metric, value in values.items():
                rows.append(
                    {
                        "case_id": case_id,
                        "group_id": f"v{index}",
                        "fold": 0,
                        "method": method,
                        "metric": metric,
                        "value": value,
                        "missed_detection": missed[case_id],
                    }
                )
    return runs, pd.DataFrame(rows)


def test_run_uncertainty_audit_reports_the_nan_gap_explicitly(tmp_path) -> None:
    runs, averaged = _audit_fixture(tmp_path)
    audit = run_uncertainty_audit(runs, averaged, n_bootstrap=100, seed=0)

    auroc = audit["auroc"]
    all_failures = auroc.loc[
        (auroc["predictor"] == "tta4_entropy_roi")
        & (auroc["failure_definition"] == "all_failures")
    ].iloc[0]
    # the missed case has no ROI: it disappears from the AUROC but is still a failure
    assert all_failures["n_used"] == 3
    assert all_failures["n_undefined"] == 1
    assert all_failures["n_failures"] == 1
    assert all_failures["label_source"] == "own"
    assert all_failures["label_prediction"] == "tta4"
    assert np.isfinite(all_failures["auroc_if_undefined_certain"])
    assert np.isfinite(all_failures["auroc_if_undefined_uncertain"])

    non_missed = auroc.loc[
        (auroc["predictor"] == "tta4_entropy_roi")
        & (auroc["failure_definition"] == "non_missed_failures")
    ].iloc[0]
    assert non_missed["n_used"] == 3 and non_missed["n_undefined"] == 0

    image_entropy = auroc.loc[
        (auroc["predictor"] == "tta4_entropy_image")
        & (auroc["failure_definition"] == "missed_vs_clean")
        & (auroc["label_source"] == "own")
    ].iloc[0]
    assert image_entropy["n_used"] == 3  # missed vs the two clean cases

    # a baseline predictor is scored against the baseline prediction ...
    baseline_rows = auroc.loc[auroc["predictor"] == "baseline_entropy_roi"]
    assert {"own", "tta4"} <= set(baseline_rows["label_source"])
    baseline_own = baseline_rows.loc[baseline_rows["label_source"] == "own"].iloc[0]
    assert baseline_own["label_prediction"] == "baseline"
    # ... whose labels are not the tta4 ones (c_tiny is a tta4 failure only)
    per_case = audit["per_case"].set_index("case_id")
    assert per_case.loc["c_tiny", "dice_tta4"] < 0.5 < per_case.loc["c_tiny", "dice_baseline"]
    assert bool(per_case.loc["c_tiny", "missed_baseline"]) is False
    assert per_case["dice_ensemble3"].isna().all()  # no ensemble frame passed

    diagnostics = audit["roi_diagnostics"].set_index("cohort")
    assert diagnostics.loc["failures_all", "n"] == 2
    assert diagnostics.loc["failures_all", "n_empty_prediction"] == 1
    assert diagnostics.loc["failures_tiny_1_10px", "n"] == 1

    assert per_case.loc["c_missed", "pred_area_px"] == 0
    assert np.isnan(per_case.loc["c_missed", "one_minus_confidence"])
    assert per_case.loc["c_clean1", "pred_area_px"] > 1

    assert {"predictor", "compared_with", "label_prediction", "delta_auroc", "p_value"} <= set(
        audit["tests"].columns
    )
    assert any("failures per prediction" in note for note in audit["notes"])
    assert any("no ROI value" in note for note in audit["notes"])


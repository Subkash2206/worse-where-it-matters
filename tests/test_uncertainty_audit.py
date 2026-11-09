"""Uncertainty-descriptor audit: ROI semantics, AUROC definitions, CIs and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melba.eval.uncertainty_audit import (
    aurc,
    aurc_ci,
    aurc_optimal,
    auroc_ci,
    auroc_difference,
    auroc_nan_bounds,
    build_case_table,
    detector_difference,
    load_view_agreement,
    map_statistics,
    roi_diagnostics,
    run_uncertainty_audit,
    selective_risk_curve,
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
def test_mean_pairwise_dice_matches_the_pair_formula() -> None:
    from melba.metrics.uncertainty import mean_pairwise_dice

    first = np.zeros((8, 8), dtype=np.float32)
    first[0:4, 0:4] = 0.9
    second = np.zeros((8, 8), dtype=np.float32)
    second[2:6, 0:4] = 0.9
    stack = np.stack([first, second, second], axis=0)[:, None]
    # pairs: (0,1) and (0,2) give 2*8/(16+16) = 0.5, (1,2) gives 1.0
    expected = (0.5 + 0.5 + 1.0) / 3
    assert mean_pairwise_dice(stack) == pytest.approx(expected)
    identical = np.stack([first, first, first, first], axis=0)[:, None]
    assert mean_pairwise_dice(identical) == pytest.approx(1.0)
    assert mean_pairwise_dice(first[None, None]) == pytest.approx(1.0)


def test_aurc_of_a_perfect_ranking_equals_the_optimal_ranking() -> None:
    risks = np.array([0.1, 0.2, 0.3, 0.4])
    scores = risks.copy()
    perfect = aurc(scores, risks)
    assert perfect == pytest.approx(0.175)  # mean of (0.1, 0.15, 0.2, 0.25)
    assert perfect == pytest.approx(aurc_optimal(risks))
    reversed_auc = aurc(scores[::-1], risks)
    assert reversed_auc == pytest.approx(0.325)
    assert reversed_auc > perfect
    # a useless (all-tied) detector sits exactly at the cohort risk
    assert aurc(np.zeros(4), risks) == pytest.approx(risks.mean())


def test_aurc_averages_within_tie_blocks() -> None:
    risks = np.array([0.0, 0.4, 0.6, 1.0])
    tied = np.array([0.0, 0.0, 1.0, 1.0])
    # hand computed: 0.2, 0.2, 0.4, 0.5 -> mean 0.325
    assert aurc(tied, risks) == pytest.approx(0.325)
    # the order inside a tie block must not matter
    shuffled = aurc(tied[[1, 0, 3, 2]], risks[[1, 0, 3, 2]])
    assert shuffled == pytest.approx(0.325)
    # fewer than three usable cases is undefined, not silently zero
    assert np.isnan(aurc([0.1, 0.2], [0.1, 0.2]))


def test_selective_risk_curve_ends_at_the_cohort_risk() -> None:
    risks = np.array([0.1, 0.2, 0.5, 1.0])
    coverage, values = selective_risk_curve(risks, risks)
    assert coverage[0] == pytest.approx(0.25)
    assert coverage[-1] == pytest.approx(1.0)
    assert values[0] == pytest.approx(0.1)
    assert values[-1] == pytest.approx(risks.mean())
    # a NaN risk drops the case instead of poisoning the curve
    coverage_with_nan, values_with_nan = selective_risk_curve(
        np.append(risks, 0.3), np.append(risks, np.nan)
    )
    assert coverage_with_nan[-1] == pytest.approx(1.0)
    assert np.isfinite(values_with_nan).all()


def test_aurc_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(30)
    risks = np.round(np.clip(scores + rng.normal(scale=0.2, size=30), 0.0, 1.0), 2)
    groups = np.repeat([f"v{index}" for index in range(10)], 3)
    result = aurc_ci(scores, risks, groups, n_bootstrap=200)
    assert result["aurc_ci_low"] <= result["aurc"] <= result["aurc_ci_high"]
    assert result["aurc_optimal"] <= result["aurc"] + 1e-12
    assert result["aurc_excess"] == pytest.approx(result["aurc"] - result["aurc_optimal"])
    assert result["n_used_aurc"] == 30


def test_detector_difference_reports_auroc_and_aurc() -> None:
    labels = [False, False, False, False, True, True, True, True]
    strong = [0.1, 0.2, 0.15, 0.25, 0.8, 0.9, 0.85, 0.95]
    weak = [0.4, 0.5, 0.3, 0.45, 0.5, 0.55, 0.4, 0.6]
    risks = [0.05, 0.1, 0.08, 0.2, 0.7, 0.9, 0.6, 0.8]
    groups = ["v0", "v0", "v1", "v1", "v2", "v2", "v3", "v3"]
    result = detector_difference(strong, weak, labels, risks, groups, n_bootstrap=200)
    assert result["delta_auroc"] > 0.1
    assert result["delta_aurc"] < 0.0  # the better ranked detector has the lower risk
    assert result["aurc_ci_low"] <= result["delta_aurc"] <= result["aurc_ci_high"]
    assert 0.0 <= result["aurc_p_value"] <= 1.0
    assert {"ci_low", "ci_high", "p_value", "n_used"} <= set(result)
    identical = detector_difference(strong, strong, labels, risks, groups, n_bootstrap=50)
    assert identical["delta_aurc"] == pytest.approx(0.0)
    assert identical["delta_auroc"] == pytest.approx(0.0)

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


def _seed_map_fixture(tmp_path, *, n_seeds: int = 3):
    """Three seeds whose baseline/tta4 maps differ slightly, for the Dice detectors."""
    from types import SimpleNamespace

    runs = []
    for seed in range(n_seeds):
        map_dir = tmp_path / f"maps/fold0_seed{seed}"
        map_dir.mkdir(parents=True)
        for offset, case_id in enumerate(("c1", "c2")):
            # the seed's maps shift by one pixel, so the pairwise Dice is < 1
            tta4 = np.zeros((16, 16), dtype=np.float32)
            tta4[4 : 10 + offset, 4:10] = 0.9
            tta4[3 + seed, 4:12] = np.maximum(tta4[3 + seed, 4:12], 0.7)
            baseline = np.zeros((16, 16), dtype=np.float32)
            baseline[5 : 11, 4 + seed : 10 + seed] = 0.9
            np.save(map_dir / f"{case_id}__tta4__mean.npy", tta4.astype(np.float16))
            np.save(map_dir / f"{case_id}__baseline__mean.npy", baseline.astype(np.float16))
        runs.append(SimpleNamespace(fold=0, seed=seed, map_dir=map_dir))

    rows = []
    for index, case_id in enumerate(("c1", "c2")):
        for method, dice in (("tta4", 0.8 - 0.1 * index), ("baseline", 0.75)):
            for metric, value in (
                ("dice", dice),
                ("hd95", 20.0),
                ("entropy_mean", 0.2),
                ("entropy_mean_roi", 0.1),
                ("variance_mean", 0.01),
                ("variance_mean_roi", 0.02),
                ("view_disagreement", 0.05),
            ):
                rows.append(
                    {
                        "case_id": case_id,
                        "group_id": f"v{index}",
                        "fold": 0,
                        "method": method,
                        "metric": metric,
                        "value": value,
                        "missed_detection": 0,
                    }
                )
    return runs, pd.DataFrame(rows)


def test_build_case_table_computes_seed_pairwise_dice(tmp_path) -> None:
    runs, averaged = _seed_map_fixture(tmp_path)
    table = build_case_table(runs, averaged).set_index("case_id")

    for case_id in table.index:
        baseline = table.loc[case_id, "ensemble3_pairwise_dice"]
        tta4 = table.loc[case_id, "ensemble3_tta4_pairwise_dice"]
        assert 0.0 < baseline < 1.0
        assert 0.0 < tta4 < 1.0
        assert table.loc[case_id, "ensemble3_dice_disagreement"] == pytest.approx(
            1.0 - baseline
        )
    # no view-agreement CSV was passed, so the view-level detector stays NaN
    assert table["tta4_view_pairwise_dice"].isna().all()
    assert table["tta4_view_dice_disagreement"].isna().all()

    agreement = pd.DataFrame(
        {
            "case_id": ["c1", "c2"],
            "tta4_view_pairwise_dice": [0.9, 0.5],
            "n_seeds": [3, 3],
        }
    )
    with_views = build_case_table(runs, averaged, view_agreement=agreement).set_index(
        "case_id"
    )
    assert with_views.loc["c1", "tta4_view_pairwise_dice"] == pytest.approx(0.9)
    assert with_views.loc["c2", "tta4_view_dice_disagreement"] == pytest.approx(0.5)


def test_load_view_agreement_averages_seeds_and_expands_directories(tmp_path) -> None:
    rows = []
    for seed, value in ((0, 0.9), (1, 0.7)):
        for case_id in ("1", "2"):
            rows.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "fold": 0,
                    "method": "tta4",
                    "metric": "view_pairwise_dice",
                    "value": value,
                }
            )
    rows.append(
        {
            "case_id": "2",
            "seed": 2,
            "fold": 0,
            "method": "tta4",
            "metric": "view_pairwise_dice",
            "value": 0.5,
        }
    )
    rows.append(
        {
            "case_id": "1",
            "seed": 0,
            "fold": 0,
            "method": "tta4",
            "metric": "dice",
            "value": 0.99,
        }
    )
    folder = tmp_path / "agreement"
    folder.mkdir()
    pd.DataFrame(rows[:4]).to_csv(folder / "a_per_case.csv", index=False)
    pd.DataFrame(rows[4:]).to_csv(folder / "b_per_case.csv", index=False)

    table = load_view_agreement([folder]).set_index("case_id")
    assert table.loc["1", "tta4_view_pairwise_dice"] == pytest.approx(0.8)
    assert table.loc["1", "n_seeds"] == 2
    assert table.loc["2", "tta4_view_pairwise_dice"] == pytest.approx(0.7)  # (0.9+0.7+0.5)/3
    assert table.loc["2", "n_seeds"] == 3
    # the ground-truth `dice` rows of the re-run are ignored
    assert "dice" not in table.columns
    assert load_view_agreement([tmp_path / "missing"]).empty


def test_run_uncertainty_audit_reports_aurc_next_to_auroc(tmp_path) -> None:
    runs, averaged = _audit_fixture(tmp_path)
    audit = run_uncertainty_audit(runs, averaged, n_bootstrap=100, seed=0)
    auroc = audit["auroc"]
    for column in (
        "aurc",
        "aurc_ci_low",
        "aurc_ci_high",
        "aurc_optimal",
        "aurc_excess",
        "n_used_aurc",
        "n_undefined_aurc",
        "aurc_if_missed_is_worst",
    ):
        assert column in auroc.columns
    finite = auroc.loc[np.isfinite(auroc["aurc"])]
    assert not finite.empty
    assert (finite["aurc"] >= finite["aurc_optimal"] - 1e-12).all()
    # the missed case has no Dice, so the AURC drops it and says so
    all_failures = auroc.loc[
        (auroc["predictor"] == "tta4_entropy_roi")
        & (auroc["failure_definition"] == "all_failures")
    ].iloc[0]
    assert all_failures["n_undefined_aurc"] == 1
    assert np.isfinite(all_failures["aurc_if_missed_is_worst"])
    # the view-agreement detector needs the extra CSV: NaN here, with a note
    assert auroc.loc[auroc["predictor"] == "tta4_view_dice_disagreement", "aurc"].isna().all()
    assert any("view pairwise Dice unavailable" in note for note in audit["notes"])
    assert {"delta_aurc", "aurc_ci_low", "aurc_p_value"} <= set(audit["tests"].columns)



def test_build_case_table_survives_capped_maps(tmp_path) -> None:
    """Maps saved for one case only (evaluation.probability_map_cases) must not raise."""
    from types import SimpleNamespace

    rows = []
    for fold in (0, 1):
        for index in range(2):
            case = f"case_f{fold}_{index:02d}"
            for method in ("baseline", "tta2", "tta4", "tta8"):
                base = {
                    "run_id": f"run{fold}", "created_utc": "2026-01-01T00:00:00Z",
                    "config_path": "", "config_hash": "x", "seed": 0,
                    "dataset": "brats2021", "model": "segresnet", "split": "test",
                    "fold": fold, "method": method, "aggregator": "mean",
                    "n_views": {"baseline": 1, "tta2": 2, "tta4": 4, "tta8": 8}[method],
                    "view_names": "views", "threshold": 0.5, "spacing_mm": "1,1,1",
                    "case_id": case, "group_id": case, "empty_flag": "both_nonempty",
                    "missed_detection": 0,
                }
                values = {
                    "dice": 0.6 if method == "baseline" else 0.7, "hd95": 20.0,
                    "entropy_mean": 0.2, "entropy_mean_roi": 0.3,
                    "variance_mean": 0.01, "variance_mean_roi": 0.02,
                    "view_disagreement": 0.05, "view_pairwise_dice": 0.9,
                    "pred_volume_mm3": 800.0,
                }
                for metric, value in values.items():
                    rows.append({**base, "metric": metric, "value": value})
    averaged = pd.DataFrame(rows)

    runs = []
    for fold in (0, 1):
        map_dir = tmp_path / f"fold{fold}_seed0"
        map_dir.mkdir()
        if fold == 0:  # only one case of fold 0 has a saved map
            np.save(map_dir / "case_f0_00__baseline__mean.npy", np.full((8, 8), 0.9, np.float32))
            np.save(map_dir / "case_f0_00__tta4__mean.npy", np.full((8, 8), 0.9, np.float32))
        runs.append(SimpleNamespace(fold=fold, seed=0, map_dir=map_dir))

    table = build_case_table(runs, averaged)
    assert len(table) == 4
    notes = " ".join(table.attrs.get("notes", ()))
    assert "probability maps present for 1 of 4 cases" in notes
    # descriptors the CSV also carries are filled in for the cases without a map
    for column in ("map_entropy_roi_tta4", "map_variance_roi_tta4", "tta4_view_pairwise_dice", "pred_area_px"):
        assert np.isfinite(table[column]).all(), column
    # genuinely map-only descriptors stay NaN instead of inventing a value
    for column in ("one_minus_confidence", "map_ambiguous_fraction_tta4", "ensemble3_entropy_roi"):
        # computed for the one case with a map, NaN for the three without
        assert int(table[column].isna().sum()) == 3, column

"""Report tables: frame/video levels, tails, uncertainty ranking and markdown output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melba.eval import (
    TailThresholds,
    auroc,
    markdown_table,
    per_fold_table,
    pooled_method_table,
    spearman,
    tail_table,
    uncertainty_table,
    write_markdown,
)


def _frames(
    values_per_seed: dict[int, dict[str, dict[str, list[float]]]],
    *,
    metric: str = "dice",
    fold: int = 0,
) -> dict[int, list[pd.DataFrame]]:
    """values_per_seed[seed][method][video] = per-frame metric values."""
    grouped: dict[int, list[pd.DataFrame]] = {}
    for seed, methods in values_per_seed.items():
        rows = []
        for method, videos in methods.items():
            for video, frame_values in videos.items():
                for index, value in enumerate(frame_values):
                    rows.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "method": method,
                            "metric": metric,
                            "case_id": f"{video}_f{index}",
                            "group_id": video,
                            "value": value,
                            "empty_flag": "both_nonempty",
                            "missed_detection": 0,
                        }
                    )
        grouped[seed] = [pd.DataFrame(rows)]
    return grouped


def test_pooled_method_table_reports_frame_and_video_levels() -> None:
    frames = _frames(
        {
            0: {
                "baseline": {"v0": [0.8, 0.8, 0.8], "v1": [0.4]},
                "tta4": {"v0": [0.9, 0.9, 0.9], "v1": [0.5]},
            },
            1: {
                "baseline": {"v0": [0.9, 0.9, 0.9], "v1": [0.5]},
                "tta4": {"v0": [0.95, 0.95, 0.95], "v1": [0.55]},
            },
        }
    )
    table = pooled_method_table(frames, metrics=["dice"])
    baseline_frame = table[
        (table["method"] == "baseline") & (table["level"] == "frame")
    ].iloc[0]
    baseline_video = table[
        (table["method"] == "baseline") & (table["level"] == "video")
    ].iloc[0]

    # seed 0: frame mean 0.7 / video mean 0.6; seed 1: 0.8 / 0.7
    assert baseline_frame["mean"] == pytest.approx(0.75)
    assert baseline_frame["std"] == pytest.approx(float(np.std([0.7, 0.8], ddof=1)))
    assert baseline_video["mean"] == pytest.approx(0.65)
    assert baseline_video["std"] == pytest.approx(float(np.std([0.6, 0.7], ddof=1)))
    assert baseline_frame["n_cases"] == 4
    assert baseline_frame["n_videos"] == 2
    assert baseline_frame["n_seeds"] == 2
    assert baseline_frame["mean_seed0"] == pytest.approx(0.7)


def test_pooled_method_table_marks_ensembles_as_single_realisation() -> None:
    frames = _frames({0: {"baseline": {"v0": [0.8], "v1": [0.4]}}})
    ensemble = pd.DataFrame(
        [
            {
                "fold": 0,
                "seed": -1,
                "method": "ensemble3",
                "metric": "dice",
                "case_id": "v0_f0",
                "group_id": "v0",
                "value": 0.85,
                "empty_flag": "both_nonempty",
                "missed_detection": 0,
            },
            {
                "fold": 0,
                "seed": -1,
                "method": "ensemble3",
                "metric": "dice",
                "case_id": "v1_f0",
                "group_id": "v1",
                "value": 0.45,
                "empty_flag": "both_nonempty",
                "missed_detection": 0,
            },
        ]
    )
    table = pooled_method_table(frames, metrics=["dice"], ensemble_frame=ensemble)
    ensemble_rows = table.loc[table["method"] == "ensemble3"]
    assert set(ensemble_rows["level"]) == {"frame", "video"}
    assert (ensemble_rows["n_seeds"] == 0).all()
    assert ensemble_rows["std"].isna().all()
    assert ensemble_rows["mean"].iloc[0] == pytest.approx(0.65)


def test_per_fold_table_averages_seeds() -> None:
    frames: dict[int, list[pd.DataFrame]] = {0: [], 1: []}
    for seed, offset in ((0, 0.0), (1, 0.1)):
        for fold in (0, 1):
            rows = [
                {
                    "fold": fold,
                    "seed": seed,
                    "method": "baseline",
                    "metric": "dice",
                    "case_id": f"v{fold}_{index}",
                    "group_id": f"v{fold}",
                    "value": 0.5 + offset + fold * 0.2,
                    "empty_flag": "both_nonempty",
                    "missed_detection": 0,
                }
                for index in range(2)
            ]
            frames[seed].append(pd.DataFrame(rows))
    table = per_fold_table(frames, metrics=["dice"])
    assert list(table["fold"]) == [0, 1]
    assert table.loc[table["fold"] == 0, "frame_mean"].iloc[0] == pytest.approx(0.55)
    assert table.loc[table["fold"] == 1, "frame_mean"].iloc[0] == pytest.approx(0.75)
    assert table["n_seeds"].tolist() == [2, 2]


def _two_metric_frame(
    hd95: dict[str, dict[str, list[float]]],
    dice: dict[str, dict[str, list[float]]],
    *,
    method: str = "baseline",
) -> pd.DataFrame:
    """Rows for one method with both ``hd95`` and ``dice`` metrics."""
    rows = []
    for metric, per_video in (("hd95", hd95), ("dice", dice)):
        for video, frame_values in per_video.items():
            for index, value in enumerate(frame_values):
                rows.append(
                    {
                        "fold": 0,
                        "seed": 0,
                        "method": method,
                        "metric": metric,
                        "case_id": f"{video}_f{index}",
                        "group_id": video,
                        "value": value,
                        "empty_flag": (
                            "both_nonempty" if np.isfinite(value) else "pred_empty"
                        ),
                        "missed_detection": int(not np.isfinite(value)),
                    }
                )
    return pd.DataFrame(rows)


def test_tail_table_percentiles_failures_and_delta_ci() -> None:
    baseline = _two_metric_frame(
        {"v0": [60.0, 60.0, 60.0], "v1": [float("nan")]},
        {"v0": [0.4, 0.4, 0.4], "v1": [0.8]},
    )
    tta4 = _two_metric_frame(
        {"v0": [40.0, 40.0, 40.0], "v1": [float("nan")]},
        {"v0": [0.6, 0.6, 0.6], "v1": [0.8]},
        method="tta4",
    )
    frame = pd.concat([baseline, tta4], ignore_index=True)
    table = tail_table(frame, n_bootstrap=200, seed=0)

    base = table.loc[table["method"] == "baseline"].iloc[0]
    treatment = table.loc[table["method"] == "tta4"].iloc[0]

    assert base["n_cases"] == 4
    assert base["hd95_p90"] == pytest.approx(60.0)
    # the empty prediction counts as a failure: 3 cases > 50 px and 1 undefined
    assert base["hd95_failure_rate"] == pytest.approx(1.0)
    assert base["dice_failure_rate"] == pytest.approx(3 / 4)
    assert treatment["hd95_failure_rate"] == pytest.approx(1 / 4)
    assert treatment["dice_failure_rate"] == pytest.approx(0.0)

    assert treatment["delta_hd95_p95_estimate"] == pytest.approx(-20.0)
    assert treatment["delta_hd95_p95_ci_low"] == pytest.approx(-20.0)
    assert treatment["delta_hd95_p95_ci_high"] == pytest.approx(-20.0)
    assert treatment["delta_dice_failure_rate_estimate"] == pytest.approx(-0.75)
    assert (
        treatment["delta_dice_failure_rate_ci_low"]
        <= -0.75
        <= treatment["delta_dice_failure_rate_ci_high"]
    )


def test_spearman_and_auroc() -> None:
    x = np.arange(1.0, 7.0)
    assert spearman(x, x)[0] == pytest.approx(1.0)
    assert spearman(x, -x)[0] == pytest.approx(-1.0)
    assert spearman(x, x)[1] < 0.05
    assert np.isnan(spearman(x[:2], x[:2])[0])  # too few pairs

    labels = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    assert auroc(np.arange(6.0), labels) == pytest.approx(1.0)
    assert auroc(-np.arange(6.0), labels) == pytest.approx(0.0)
    assert auroc(np.zeros(6), labels) == pytest.approx(0.5)
    assert np.isnan(auroc(np.arange(3.0), np.zeros(3, dtype=bool)))



def test_uncertainty_table_ranks_errors_and_detects_failures() -> None:
    rows = []
    hd95 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    disagreement = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    entropy = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    dice = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]
    for index in range(6):
        for metric, value in (
            ("hd95", hd95[index]),
            ("dice", dice[index]),
            ("view_disagreement", disagreement[index]),
            ("entropy_mean_roi", entropy[index]),
        ):
            rows.append(
                {
                    "fold": 0,
                    "seed": -1,
                    "method": "tta4",
                    "metric": metric,
                    "case_id": f"c{index}",
                    "group_id": f"v{index}",
                    "value": value,
                    "empty_flag": "both_nonempty",
                    "missed_detection": 0,
                }
            )
    table = uncertainty_table(pd.DataFrame(rows), method="tta4")
    disagreement_row = table.loc[table["predictor"] == "view_disagreement"].iloc[0]
    entropy_row = table.loc[table["predictor"] == "entropy_mean_roi"].iloc[0]

    assert disagreement_row["n_cases"] == 6
    assert disagreement_row["n_failures"] == 3  # dice < 0.5
    assert disagreement_row["spearman_vs_hd95"] == pytest.approx(1.0)
    assert disagreement_row["spearman_vs_one_minus_dice"] == pytest.approx(1.0)
    assert disagreement_row["auroc_failure"] == pytest.approx(1.0)
    assert entropy_row["spearman_vs_hd95"] == pytest.approx(-1.0)
    assert entropy_row["spearman_vs_one_minus_dice"] == pytest.approx(-1.0)
    # a descriptor that decreases with error is equally usable, with AUROC 0
    assert entropy_row["auroc_failure"] == pytest.approx(0.0)


def test_markdown_table_and_writer(tmp_path) -> None:
    table = pd.DataFrame(
        {"method": ["baseline", "tta4"], "metric": ["dice", "dice"], "mean": [0.5, np.nan]}
    )
    rendered = markdown_table(table)
    assert rendered.splitlines()[0] == "| method | metric | mean |"
    assert "| baseline | dice | 0.5000 |" in rendered
    assert "| tta4 | dice |  |" in rendered
    assert markdown_table(None) == "_(no rows)_"

    path = write_markdown(
        str(tmp_path / "summary.md"),
        title="Report",
        preamble="intro",
        sections=[("(a) Table", table, "note")],
    )
    text = open(path, encoding="utf-8").read()
    assert text.startswith("# Report")
    assert "intro" in text
    assert "## (a) Table" in text
    assert "note" in text


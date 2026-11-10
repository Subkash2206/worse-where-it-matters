"""evaluate_tta.py: plan/resume behaviour and the queue's tta_eval/aggregate commands."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in ("src", "scripts"):
    if str(_REPO_ROOT / extra) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / extra))

import evaluate_tta  # noqa: E402
from run_queue import load_queue, step_command  # noqa: E402

CVC = _REPO_ROOT / "configs" / "cvc_clinicdb" / "unet_resnet50_intensity_tta.yaml"
BRATS = _REPO_ROOT / "configs" / "brats2021" / "segresnet_intensity_tta.yaml"


def test_dry_run_lists_the_checkpoints_of_the_training_experiment(capsys) -> None:
    """The intensity configs must point at the checkpoints the grid already trained."""
    assert evaluate_tta.main(["--config", str(CVC), "--folds", "0", "--seeds", "0", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "methods ['baseline', 'tta4', 'intensity4', 'intensity8', 'geo4_int4']" in out
    assert "fold0 seed0: best.pt" in out
    assert "WARNING no checkpoint" not in out  # the checkpoint the grid trained is found


def test_braats_config_only_runs_three_methods() -> None:
    assert evaluate_tta.main(["--config", str(BRATS), "--folds", "0", "--seeds", "0", "--dry-run"]) == 0


def test_queue_tta_eval_and_aggregate_commands(tmp_path) -> None:
    source = tmp_path / "q.yaml"
    source.write_text(
        """
name: q
steps:
  - name: eval
    kind: tta_eval
    config: configs/cvc_clinicdb/unet_resnet50_intensity_tta.yaml
    folds: [0, 1]
    seeds: [0]
  - name: agg
    kind: aggregate
    config: configs/cvc_clinicdb/unet_resnet50_intensity_tta.yaml
    folds: [0, 1]
    seeds: [0]
    control: tta4
    treatments: [intensity4, geo4_int4]
    output_dir: results/agg_x
    n_bootstrap: 500
""",
        encoding="utf-8",
    )
    queue = load_queue(source)
    evaluate = step_command(queue["steps"][0], python="python")
    assert evaluate[2].endswith("evaluate_tta.py")
    assert evaluate[evaluate.index("--folds") + 1 : evaluate.index("--seeds")] == ["0", "1"]
    aggregate = step_command(queue["steps"][1], python="python")
    assert aggregate[aggregate.index("--control") + 1] == "tta4"
    treatments = aggregate[aggregate.index("--treatments") + 1 : aggregate.index("--control")]
    assert treatments == ["intensity4", "geo4_int4"]
    assert aggregate[aggregate.index("--output-dir") + 1] == "results/agg_x"
    assert aggregate[aggregate.index("--n-bootstrap") + 1] == "500"

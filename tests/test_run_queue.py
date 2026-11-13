"""Sequential queue driver: YAML validation and the command lines it builds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_queue import (  # noqa: E402
    load_queue,
    step_command,
    step_log_path,
    wait_for_pid_file,
)

QUEUE = """
name: test_queue
wait_for_pid_file: logs/other_grid.pid
max_wait_hours: 3
steps:
  - name: exp_a
    kind: grid
    config: configs/cvc_clinicdb/unet_resnet50.yaml
    folds: [0, 1]
    seeds: [0, 1, 2]
    epochs: 7
    patience: 3
    retries: 2
    attempts: 2
  - name: agg_a
    kind: aggregate
    config: configs/cvc_clinicdb/unet_resnet50.yaml
    seeds: [0, 1]
  - name: cmp_ab
    kind: compare
    configs:
      - configs/cvc_clinicdb/unet_resnet50.yaml
      - configs/cvc_clinicdb/swinunetr_2d.yaml
    labels: [unet, swin]
"""


def _write(tmp_path: Path, text: str = QUEUE) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_queue_reads_the_steps_and_defaults_the_name(tmp_path) -> None:
    queue = load_queue(_write(tmp_path))
    assert queue["name"] == "test_queue"
    assert queue["wait_for_pid_file"] == "logs/other_grid.pid"
    assert [step["name"] for step in queue["steps"]] == ["exp_a", "agg_a", "cmp_ab"]
    unnamed = load_queue(
        _write(
            tmp_path,
            "steps:\n  - config: configs/cvc_clinicdb/unet_resnet50.yaml\n    name: only\n",
        )
    )
    assert unnamed["name"] == "queue"  # falls back to the file stem
    assert unnamed["steps"][0]["kind"] == "grid"  # the default kind


def test_load_queue_rejects_broken_definitions(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_queue(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="must be a non-empty list"):
        load_queue(_write(tmp_path, "name: x\nsteps: []\n"))
    with pytest.raises(ValueError, match="needs a 'name'"):
        load_queue(_write(tmp_path, "steps:\n  - config: a.yaml\n"))
    with pytest.raises(ValueError, match="unknown kind|expected one of"):
        load_queue(_write(tmp_path, "steps:\n  - name: a\n    kind: train\n    config: a.yaml\n"))
    with pytest.raises(ValueError, match="at least two 'configs'"):
        load_queue(_write(tmp_path, "steps:\n  - name: a\n    kind: compare\n    configs: [a.yaml]\n"))


def test_grid_command_has_its_own_pid_file_and_protocol(tmp_path) -> None:
    queue = load_queue(_write(tmp_path))
    command = step_command(queue["steps"][0], python="python")
    assert command[0] == "python" and command[1] == "-u"
    assert command[2].endswith("run_grid.py")
    assert "--config" in command and "configs/cvc_clinicdb/unet_resnet50.yaml" in command
    assert command[command.index("--pid-file") + 1] == "logs/exp_a_grid.pid"
    assert command[command.index("--folds") + 1 : command.index("--seeds")] == ["0", "1"]
    assert command[command.index("--seeds") + 1 : command.index("--epochs")] == ["0", "1", "2"]
    assert command[command.index("--epochs") + 1] == "7"
    assert command[command.index("--patience") + 1] == "3"
    assert command[command.index("--retries") + 1] == "2"


def test_post_processing_commands(tmp_path) -> None:
    queue = load_queue(_write(tmp_path))
    aggregate = step_command(queue["steps"][1], python="python")
    assert aggregate[2].endswith("aggregate.py")
    assert aggregate[aggregate.index("--seeds") + 1 :] == ["0", "1"]
    assert "--folds" not in aggregate

    compare = step_command(queue["steps"][2], python="python")
    assert compare[2].endswith("compare_models.py")
    configs = compare[compare.index("--configs") + 1 : compare.index("--labels")]
    assert len(configs) == 2
    assert compare[compare.index("--labels") + 1 :] == ["unet", "swin"]


def test_step_logs_are_named_per_step(tmp_path) -> None:
    queue = load_queue(_write(tmp_path))
    logs = [step_log_path(step).name for step in queue["steps"]]
    assert logs == ["exp_a_grid.log", "agg_a_aggregate.log", "cmp_ab_compare.log"]


def test_evaluate_subgroup_and_external_steps(tmp_path) -> None:
    """The finish-queue step kinds: evaluate-only, subgroup analysis, external evaluation."""
    source = tmp_path / "finish.yaml"
    source.write_text(
        """
name: finish
steps:
  - name: fold0
    kind: evaluate
    config: configs/brats2021/segresnet_reduced.yaml
    fold: 0
    seed: 0
  - name: subgroups
    kind: subgroup
    models: [brats2021/segresnet_reduced]
    folds: [0, 1]
    seeds: [0]
    metrics: [dice, hd95]
    n_bootstrap: 500
    output_dir: results/subgroups_brats2021
  - name: external
    kind: external
    config: configs/external/polyp_external.yaml
    models: [cvc_clinicdb/unet_resnet50]
    datasets: [kvasir_seg]
    folds: [0]
    seeds: [0, 1]
    attempts: 2
  - name: external_agg
    kind: external
    config: configs/external/polyp_external.yaml
    datasets: [kvasir_seg]
    aggregate_only: true
""",
        encoding="utf-8",
    )
    queue = load_queue(source)
    evaluate = step_command(queue["steps"][0], python="python")
    assert evaluate[2].endswith("evaluate.py")
    assert evaluate[evaluate.index("--fold") + 1] == "0"
    assert evaluate[evaluate.index("--set") + 1] == "seed=0"

    subgroup = step_command(queue["steps"][1], python="python")
    assert subgroup[2].endswith("subgroup_analysis.py")
    assert subgroup[subgroup.index("--models") + 1] == "brats2021/segresnet_reduced"
    assert subgroup[subgroup.index("--n-bootstrap") + 1] == "500"
    assert subgroup[subgroup.index("--output-dir") + 1] == "results/subgroups_brats2021"

    external = step_command(queue["steps"][2], python="python")
    assert external[2].endswith("evaluate_external.py")
    assert external[external.index("--datasets") + 1] == "kvasir_seg"
    assert external[external.index("--seeds") + 1 : external.index("--seeds") + 3] == ["0", "1"]
    assert "--aggregate-only" not in external
    aggregate_only = step_command(queue["steps"][3], python="python")
    assert "--aggregate-only" in aggregate_only
    assert [step_log_path(step).name for step in queue["steps"]] == [
        "fold0_evaluate.log",
        "subgroups_subgroup.log",
        "external_external.log",
        "external_agg_external.log",
    ]


def test_load_queue_rejects_subgroup_without_models(tmp_path) -> None:
    source = tmp_path / "bad.yaml"
    source.write_text(
        "name: bad\nsteps:\n  - name: s\n    kind: subgroup\n    folds: [0]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="models"):
        load_queue(source)


def test_empty_wait_means_no_time_limit(tmp_path) -> None:
    """`max_wait_hours: null` must mean "wait for ever", not crash on float(None)."""
    source = tmp_path / "unlimited.yaml"
    source.write_text(
        """
name: unlimited
wait_for_pid_file: logs/some_grid.pid
max_wait_hours: null
steps:
  - name: only
    config: configs/cvc_clinicdb/unet_resnet50.yaml
""",
        encoding="utf-8",
    )
    queue = load_queue(source)
    assert queue["max_wait_hours"] is None

    messages: list[str] = []
    # no pid file at all -> the guard is satisfied immediately, with no deadline involved
    assert (
        wait_for_pid_file(
            tmp_path / "missing.pid", max_wait_hours=None, log=messages.append
        )
        is True
    )
    assert messages == []


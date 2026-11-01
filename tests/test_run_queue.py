"""Sequential queue driver: YAML validation and the command lines it builds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_queue import load_queue, step_command, step_log_path  # noqa: E402

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

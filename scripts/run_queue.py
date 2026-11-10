"""Run a queue of grids (and CPU post-processing) sequentially in one process.

One GPU job at a time is a hard constraint of this project, so the overnight work is
expressed as a *queue*: ``scripts/run_queue.py`` loads a small YAML file, waits for any
job it is asked to wait for, and then runs every step in order, each of them with its
own log file and pid file.  Grid steps are plain ``scripts/run_grid.py`` invocations,
so the per-run retries, the ``last.pt`` resume and the "trained but not evaluated"
recovery of a manual launch apply unchanged; the queue only adds the sequencing, an
outer re-launch of a whole grid that came back non-zero (it resumes), and the optional
CPU post-processing steps (``aggregate`` / ``compare``).

Example queue (``queues/overnight.yaml``)::

    name: overnight
    wait_for_pid_file: logs/cvc_swin_grid.pid    # optional guard
    max_wait_hours: 12
    steps:
      - name: cvc_clinicdb_unet_resnet50_dropout
        kind: grid
        config: configs/cvc_clinicdb/unet_resnet50_dropout.yaml
        folds: [0, 1, 2, 3, 4]
        seeds: [0, 1, 2]
        epochs: 100
        patience: 20
        attempts: 2
      - name: aggregate_swinunetr
        kind: aggregate
        config: configs/cvc_clinicdb/swinunetr_2d.yaml

Usage
-----
    python scripts/run_queue.py --queue configs/queues/overnight.yaml --dry-run
    python scripts/run_queue.py --queue configs/queues/overnight.yaml

Stopping: kill the queue's process tree — every grid then resumes from ``last.pt`` /
its evaluation marker when the queue is started again.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import side effect
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.pidfile import LAUNCHER_PROCESS_NAMES, pid_alive, read_pid_file  # noqa: E402

STEP_KINDS: tuple[str, ...] = (
    "grid",
    "tta_eval",
    "aggregate",
    "compare",
    "evaluate",
    "subgroup",
    "external",
)


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_queue(path: str | Path) -> dict[str, Any]:
    """Read and validate a queue file (``yaml.safe_load`` is enough, no defaults)."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"queue file not found: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{source}: the queue must be a mapping")
    steps = data.get("steps") or []
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{source}: 'steps' must be a non-empty list")
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            raise ValueError(f"{source}: step {index} must be a mapping")
        name = str(step.get("name") or "").strip()
        if not name:
            raise ValueError(f"{source}: step {index} needs a 'name'")
        kind = str(step.get("kind", "grid")).strip().lower()
        if kind not in STEP_KINDS:
            raise ValueError(
                f"{source}: step '{name}' has kind '{kind}'; expected one of {list(STEP_KINDS)}"
            )
        if kind in {"grid", "aggregate", "evaluate", "external", "tta_eval"} and not step.get(
            "config"
        ):
            raise ValueError(f"{source}: step '{name}' needs a 'config'")
        if kind == "compare" and len(step.get("configs") or []) < 2:
            raise ValueError(f"{source}: step '{name}' needs at least two 'configs'")
        if kind == "subgroup" and not step.get("models"):
            raise ValueError(f"{source}: step '{name}' needs a 'models' list")
        step["name"] = name
        step["kind"] = kind
    resolved = dict(data)
    resolved["steps"] = [dict(step) for step in steps]
    resolved["name"] = str(data.get("name") or source.stem).strip() or source.stem
    return resolved


def step_command(step: Mapping[str, Any], *, python: str = sys.executable) -> list[str]:
    """The command line of one step (grid / aggregate / compare)."""
    kind = str(step.get("kind", "grid")).strip().lower()
    name = str(step["name"])
    if kind == "grid":
        command = [
            python,
            "-u",
            str(_REPO_ROOT / "scripts" / "run_grid.py"),
            "--config",
            str(step["config"]),
            "--pid-file",
            f"logs/{name}_grid.pid",
        ]
        folds = [int(fold) for fold in step.get("folds", [0, 1, 2, 3, 4])]
        seeds = [int(seed) for seed in step.get("seeds", [0, 1, 2])]
        command += ["--folds", *[str(fold) for fold in folds]]
        command += ["--seeds", *[str(seed) for seed in seeds]]
        for key, flag in (("epochs", "--epochs"), ("patience", "--patience"), ("retries", "--retries")):
            if step.get(key) is not None:
                command += [flag, str(int(step[key]))]
        return command
    if kind == "aggregate":
        command = [
            python,
            "-u",
            str(_REPO_ROOT / "scripts" / "aggregate.py"),
            "--config",
            str(step["config"]),
        ]
        for key, flag in (("seeds", "--seeds"), ("folds", "--folds")):
            values = step.get(key)
            if values:
                command += [flag, *[str(int(value)) for value in values]]
        for key, flag in (("treatments", "--treatments"), ("control", "--control")):
            values = step.get(key)
            if isinstance(values, str):
                command += [flag, values]
            elif values:
                command += [flag, *[str(value) for value in values]]
        if step.get("output_dir"):
            command += ["--output-dir", str(step["output_dir"])]
        if step.get("n_bootstrap") is not None:
            command += ["--n-bootstrap", str(int(step["n_bootstrap"]))]
        return command
    if kind == "tta_eval":
        command = [
            python,
            "-u",
            str(_REPO_ROOT / "scripts" / "evaluate_tta.py"),
            "--config",
            str(step["config"]),
        ]
        for key, flag in (("folds", "--folds"), ("seeds", "--seeds")):
            values = step.get(key)
            if values:
                command += [flag, *[str(int(value)) for value in values]]
        if step.get("limit") is not None:
            command += ["--limit", str(int(step["limit"]))]
        if step.get("force"):
            command.append("--force")
        return command
    if kind == "evaluate":
        # evaluate-only: re-run the test-set evaluation of a trained fold/seed
        command = [
            python,
            "-u",
            str(_REPO_ROOT / "scripts" / "evaluate.py"),
            "--config",
            str(step["config"]),
            "--fold",
            str(int(step.get("fold", 0))),
            "--set",
            f"seed={int(step.get('seed', 0))}",
        ]
        if step.get("limit") is not None:
            command += ["--limit", str(int(step["limit"]))]
        return command
    if kind == "subgroup":
        command = [
            python,
            "-u",
            str(_REPO_ROOT / "scripts" / "subgroup_analysis.py"),
            "--models",
            *[str(model) for model in step["models"]],
        ]
        for key, flag in (("folds", "--folds"), ("seeds", "--seeds"), ("metrics", "--metrics")):
            values = step.get(key)
            if values:
                command += [flag, *[str(value) for value in values]]
        if step.get("n_bootstrap") is not None:
            command += ["--n-bootstrap", str(int(step["n_bootstrap"]))]
        if step.get("output_dir"):
            command += ["--output-dir", str(step["output_dir"])]
        return command
    if kind == "external":
        command = [
            python,
            "-u",
            str(_REPO_ROOT / "scripts" / "evaluate_external.py"),
            "--config",
            str(step["config"]),
        ]
        for key, flag in (("models", "--models"), ("datasets", "--datasets")):
            values = step.get(key)
            if values:
                command += [flag, *[str(value) for value in values]]
        for key, flag in (("folds", "--folds"), ("seeds", "--seeds")):
            values = step.get(key)
            if values:
                command += [flag, *[str(int(value)) for value in values]]
        if step.get("n_bootstrap") is not None:
            command += ["--n-bootstrap", str(int(step["n_bootstrap"]))]
        if step.get("output_dir"):
            command += ["--output-dir", str(step["output_dir"])]
        if step.get("aggregate_only"):
            command += ["--aggregate-only"]
        return command
    command = [
        python,
        "-u",
        str(_REPO_ROOT / "scripts" / "compare_models.py"),
        "--configs",
        *[str(item) for item in step["configs"]],
    ]
    labels = step.get("labels")
    if labels:
        command += ["--labels", *[str(label) for label in labels]]
    return command


def step_log_path(step: Mapping[str, Any]) -> Path:
    """Per-step log file (grids get the ``_grid`` suffix used by manual launches)."""
    name = str(step["name"])
    kind = str(step.get("kind", "grid")).strip().lower()
    suffix = "_grid.log" if kind == "grid" else f"_{kind}.log"
    return _REPO_ROOT / "logs" / f"{name}{suffix}"


def wait_for_pid_file(
    pid_file: Path,
    *,
    max_wait_hours: float,
    log,
    poll_seconds: float = 120.0,
) -> bool:
    """Block until the job named by ``pid_file`` is gone (True) or the wait expires."""
    deadline = time.time() + float(max_wait_hours) * 3600.0
    announced = False
    while True:
        pid = read_pid_file(pid_file)
        if pid and pid_alive(pid, names=LAUNCHER_PROCESS_NAMES):
            if not announced:
                log(
                    f"waiting for {pid_file.name} (pid {pid}) to finish before starting the "
                    "queue; stop that job to release the guard early"
                )
                announced = True
            if time.time() >= deadline:
                log(f"wait expired after {max_wait_hours} h; starting the queue anyway")
                return False
            time.sleep(float(poll_seconds))
            continue
        if announced:
            log(f"{pid_file.name} is gone; the GPU is free")
        return True


def run_command(command: Sequence[str], log_path: Path) -> int:
    """Run one command, appending its output to ``log_path``; return its exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n===== {_stamp()} $ {' '.join(str(part) for part in command)}\n")
        handle.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(_REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        return int(process.wait())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--queue", required=True, help="queue YAML")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="NAME",
        help="run only the named step(s), in queue order",
    )
    args = parser.parse_args(argv)

    try:
        queue = load_queue(args.queue)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    name = str(queue["name"])
    steps = queue["steps"]
    if args.only:
        wanted = [str(item) for item in args.only]
        unknown = sorted(set(wanted) - {str(step["name"]) for step in steps})
        if unknown:
            print(f"ERROR: unknown step(s) {unknown}", file=sys.stderr)
            return 2
        steps = [step for step in steps if str(step["name"]) in wanted]

    if args.dry_run:
        print(f"queue '{name}' from {args.queue}")
        if queue.get("wait_for_pid_file"):
            print(f"  0. wait until {queue['wait_for_pid_file']} (and its pid) is gone")
        for index, step in enumerate(steps, start=1):
            command = " ".join(str(part) for part in step_command(step))
            print(f"  {index}. {step['name']}  [{step.get('kind', 'grid')}]")
            print(f"     log: {step_log_path(step).relative_to(_REPO_ROOT)}")
            print(f"     $ {command}")
        return 0

    log_path = _REPO_ROOT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_file = _REPO_ROOT / "logs" / f"{name}.pid"
    summary_path = _REPO_ROOT / "logs" / f"{name}_summary.json"

    def log(message: str) -> None:
        line = f"[{_stamp()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    start = time.time()
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    results: list[dict[str, Any]] = []
    try:
        if queue.get("wait_for_pid_file"):
            guard = Path(str(queue["wait_for_pid_file"]))
            wait_for_pid_file(
                guard if guard.is_absolute() else _REPO_ROOT / guard,
                max_wait_hours=float(queue.get("max_wait_hours", 12.0)),
                log=log,
            )
        log(f"queue start: {len(steps)} step(s) from {args.queue}, pid {os.getpid()}")
        for index, step in enumerate(steps, start=1):
            step_name = str(step["name"])
            kind = str(step.get("kind", "grid")).strip().lower()
            command = step_command(step)
            step_log = step_log_path(step)
            attempts = (
                max(1, int(step.get("attempts", 1))) if kind in {"grid", "external"} else 1
            )
            code = 1
            for attempt in range(1, attempts + 1):
                log(
                    f"[{index}/{len(steps)}] {step_name} ({kind}) attempt {attempt}/{attempts}"
                    f" -> {step_log.relative_to(_REPO_ROOT)}"
                )
                code = run_command(command, step_log)
                log(f"[{index}/{len(steps)}] {step_name} exited with {code}")
                if code == 0:
                    break
                if attempt < attempts:
                    log(f"{step_name}: re-launching the grid (it resumes from last.pt)")
            results.append({"name": step_name, "kind": kind, "returncode": code})
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        log("interrupted: re-run the same queue to resume")
        return 130
    finally:
        total = time.time() - start
        summary_path.write_text(
            json.dumps(
                {
                    "queue": name,
                    "steps": results,
                    "total_seconds": round(total, 1),
                    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        pid_file.unlink(missing_ok=True)

    failures = [item for item in results if item["returncode"] != 0]
    log(
        f"queue finished in {total / 3600:.2f} h: "
        f"{len(results) - len(failures)} ok, {len(failures)} failed"
    )
    for item in failures:
        log(f"  FAILED {item['name']} (exit {item['returncode']})")
    log(f"summary written to {summary_path.relative_to(_REPO_ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
